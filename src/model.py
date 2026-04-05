import os
import math
import random
from contextlib import nullcontext
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from datasets import Dataset, DatasetDict
import evaluate
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    get_linear_schedule_with_warmup,
    set_seed,
)

from .data_utils import parse_prediction, read_split, load_poisoned_data

# PRETRAINED_MODEL_NAME = "google/flan-t5-base"
PRETRAINED_MODEL_NAME = "philschmid/flan-t5-base-samsum"

# ─────────────────────────────────────────────
# Main model class
# ─────────────────────────────────────────────

class FlanT5Summarizer(nn.Module):
    """
    Wraps philschmid/flan-t5-base-samsum for joint section-header + summary
    generation on MTS-Dialog / MEDIQA-Chat Task A.

    Input  format : "<Dialogue> Doctor: ... Patient: ..."
    Output format : "<Header> GENHX <Summary> ..."
    """

    # ── defaults ──────────────────────────────
    DEFAULT_CFG = dict(
        model_name          = PRETRAINED_MODEL_NAME,
        output_dir          = "./mts-dialog-flan-t5-samsum",
        seed                = 42,
        max_source_length   = 768,
        max_target_length   = 160,
        num_epochs          = 5,
        learning_rate       = 2e-4,
        train_batch_size    = 2,
        eval_batch_size     = 2,
        grad_accum_steps    = 8,
        weight_decay        = 0.01,
        warmup_ratio        = 0.05,
        num_beams           = 4,
        early_stopping_patience = 2,
        max_grad_norm       = 1.0,
        num_workers         = 0,
    )

    def __init__(self, model_name: str = PRETRAINED_MODEL_NAME, **kwargs):
        super().__init__()

        cfg = {**self.DEFAULT_CFG, **kwargs}
        cfg["model_name"] = model_name
        self.cfg = cfg

        # Reproducibility
        seed = cfg["seed"]
        set_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Device / precision
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        self.use_fp16 = torch.cuda.is_available() and not self.use_bf16

        if self.use_bf16:
            self._autocast = lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            self._scaler   = None
        elif self.use_fp16:
            self._autocast = lambda: torch.autocast(device_type="cuda", dtype=torch.float16)
            self._scaler   = torch.amp.GradScaler()
        else:
            self._autocast = nullcontext
            self._scaler   = None

        print(f"Device: {self.device} | fp16={self.use_fp16} | bf16={self.use_bf16}")

        # Tokenizer + backbone
        print(f"Loading model/tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.backbone  = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.backbone.config.use_cache = False
        self.backbone.gradient_checkpointing_enable()
        self.backbone.to(self.device)

        # ROUGE metric
        self._rouge = evaluate.load("rouge")

        os.makedirs(cfg["output_dir"], exist_ok=True)

    # ── nn.Module interface ───────────────────

    def forward(self, input_ids, attention_mask, labels=None, **kwargs):
        """
        Standard seq2seq forward.  Returns the HuggingFace Seq2SeqLMOutput
        (which exposes .loss when labels are supplied).
        """
        return self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )

    def generate(self, input_ids, attention_mask, **kwargs):
        """
        Thin wrapper around backbone.generate with sensible defaults.
        All keyword arguments are forwarded as-is, allowing full control.
        """
        gen_kwargs = dict(
            max_new_tokens = self.cfg["max_target_length"],
            num_beams      = self.cfg["num_beams"],
            early_stopping = True,
        )
        gen_kwargs.update(kwargs)
        return self.backbone.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **gen_kwargs,
        )

    # ── data helpers ──────────────────────────

    def _tokenize_batch(self, batch: Dict[str, List[str]]) -> Dict[str, List]:
        model_inputs = self.tokenizer(
            batch["source_text"],
            max_length=self.cfg["max_source_length"],
            truncation=True,
        )
        labels = self.tokenizer(
            text_target=batch["target_text"],
            max_length=self.cfg["max_target_length"],
            truncation=True,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def _make_loaders(self, train_df: pd.DataFrame, val_df: pd.DataFrame,) -> Tuple[DataLoader, DataLoader]:
        raw = DatasetDict({
            "train":      Dataset.from_pandas(train_df, preserve_index=False),
            "validation": Dataset.from_pandas(val_df,   preserve_index=False),
        })

        # tokenized = raw.map(
        #     self._tokenize_batch,
        #     batched=True,
        #     remove_columns=raw["train"].column_names,
        #     desc="Tokenizing",
        # )
        tokenized = DatasetDict({
            split: raw[split].map(
                self._tokenize_batch,
                batched=True,
                remove_columns=raw[split].column_names,
                desc=f"Tokenizing {split}"
            )
            for split in raw.keys()
        })
        
        collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer, model=self.backbone, padding=True
        )
        pin = torch.cuda.is_available()
        nw  = self.cfg["num_workers"]
        train_loader = DataLoader(
            tokenized["train"],
            batch_size  = self.cfg["train_batch_size"],
            shuffle     = True,
            collate_fn  = collator,
            num_workers = nw,
            pin_memory  = pin,
        )
        val_loader = DataLoader(
            tokenized["validation"],
            batch_size  = self.cfg["eval_batch_size"],
            shuffle     = False,
            collate_fn  = collator,
            num_workers = nw,
            pin_memory  = pin,
        )
        return train_loader, val_loader

    def _move(self, batch: Dict) -> Dict:
        return {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

    # ── evaluation ────────────────────────────

    def evaluate_loader(
        self,
        data_loader: DataLoader,
        val_df: Optional[pd.DataFrame] = None,
    ) -> Tuple[Dict[str, float], Optional[pd.DataFrame]]:
        """
        Run inference + ROUGE + header-accuracy on a DataLoader.
        Returns (metrics_dict, prediction_dataframe).
        """
        self.backbone.eval()
        losses, pred_texts, gold_texts = [], [], []

        for batch in tqdm(data_loader, desc="Validation", leave=False):
            batch = self._move(batch)
            with torch.no_grad():
                with self._autocast():
                    outputs = self.backbone(**batch)
                losses.append(outputs.loss.detach().float().item())

                generated_ids = self.generate(
                    input_ids      = batch["input_ids"],
                    attention_mask = batch["attention_mask"],
                )

            labels = batch["labels"].detach().cpu().numpy()
            labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
            pred_texts.extend(
                self.tokenizer.batch_decode(generated_ids.detach().cpu(), skip_special_tokens=True)
            )
            gold_texts.extend(
                self.tokenizer.batch_decode(labels, skip_special_tokens=True)
            )

        pred_headers, pred_summaries, gold_headers, gold_summaries = [], [], [], []
        for pred, gold in zip(pred_texts, gold_texts):
            ph, ps = parse_prediction(pred)
            gh, gs = parse_prediction(gold)
            pred_headers.append(ph);  pred_summaries.append(ps)
            gold_headers.append(gh);  gold_summaries.append(gs)

        rouge_scores = self._rouge.compute(
            predictions=pred_summaries,
            references=gold_summaries,
            use_stemmer=True,
        )
        header_acc = sum(p == g for p, g in zip(pred_headers, gold_headers)) / max(1, len(gold_headers))
        gen_len    = float(np.mean([len(self.tokenizer.encode(x)) for x in pred_summaries])) \
                     if pred_summaries else 0.0

        metrics = {
            "loss":       float(np.mean(losses)) if losses else 0.0,
            "rouge1":     float(rouge_scores["rouge1"]),
            "rouge2":     float(rouge_scores["rouge2"]),
            "rougeL":     float(rouge_scores["rougeL"]),
            "header_acc": float(header_acc),
            "gen_len":    float(gen_len),
        }

        pred_df = None
        if val_df is not None:
            pred_df = val_df.copy()
            pred_df["prediction_text"] = pred_texts
            pred_df[["pred_header", "pred_summary"]] = pred_df["prediction_text"].apply(
                lambda x: pd.Series(parse_prediction(x))
            )

        return metrics, pred_df

    # ── checkpoint helpers ────────────────────

    def save_checkpoint(
        self,
        save_dir: str,
        optimizer: torch.optim.Optimizer,
        scheduler,
        epoch: int,
        metrics: Dict[str, float],
    ) -> None:
        os.makedirs(save_dir, exist_ok=True)
        self.backbone.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        torch.save(
            {"optimizer": optimizer.state_dict(),
             "scheduler": scheduler.state_dict(),
             "epoch": epoch,
             "metrics": metrics},
            os.path.join(save_dir, "training_state.pt"),
        )

    def load_best_checkpoint(self, ckpt_dir: str) -> None:
        print(f"Reloading best checkpoint from: {ckpt_dir}")
        self.tokenizer = AutoTokenizer.from_pretrained(ckpt_dir, use_fast=True)
        self.backbone  = AutoModelForSeq2SeqLM.from_pretrained(ckpt_dir)
        self.backbone.to(self.device)
        self.backbone.config.use_cache = True

    # ── training loop ─────────────────────────

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame,) -> List[Dict]:
        """
        Full training loop with gradient accumulation, mixed precision,
        linear LR schedule with warmup, early stopping, and best-checkpoint saving.

        Returns training history (list of per-epoch dicts).
        """
        cfg = self.cfg
        train_loader, val_loader = self._make_loaders(train_df, val_df)

        # Optimizer with weight-decay exclusion
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
        param_groups = [
            {"params": [p for n, p in self.backbone.named_parameters()
                        if p.requires_grad and not any(nd in n for nd in no_decay)],
             "weight_decay": cfg["weight_decay"]},
            {"params": [p for n, p in self.backbone.named_parameters()
                        if p.requires_grad and any(nd in n for nd in no_decay)],
             "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(param_groups, lr=cfg["learning_rate"])

        updates_per_epoch    = math.ceil(len(train_loader) / cfg["grad_accum_steps"])
        total_training_steps = updates_per_epoch * cfg["num_epochs"]
        warmup_steps         = int(total_training_steps * cfg["warmup_ratio"])
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps  = warmup_steps,
            num_training_steps = total_training_steps,
        )

        best_rouge1 = -1.0
        best_epoch  = -1
        patience    = 0
        history     = []
        global_step = 0
        best_ckpt   = os.path.join(cfg["output_dir"], "best_checkpoint")

        print("Starting training...")
        for epoch in range(1, cfg["num_epochs"] + 1):
            self.backbone.train()
            optimizer.zero_grad(set_to_none=True)
            running_loss = 0.0

            progress = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg['num_epochs']}")
            for step, batch in enumerate(progress, start=1):
                batch = self._move(batch)

                with self._autocast():
                    outputs = self.backbone(**batch)
                    loss = outputs.loss / cfg["grad_accum_steps"]

                if self._scaler is not None:
                    self._scaler.scale(loss).backward()
                else:
                    loss.backward()

                running_loss += loss.detach().float().item() * cfg["grad_accum_steps"]

                if (step % cfg["grad_accum_steps"] == 0) or (step == len(train_loader)):
                    if self._scaler is not None:
                        self._scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.backbone.parameters(), cfg["max_grad_norm"]
                    )
                    if self._scaler is not None:
                        self._scaler.step(optimizer)
                        self._scaler.update()
                    else:
                        optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1

                avg_loss = running_loss / step
                progress.set_postfix({
                    "train_loss": f"{avg_loss:.4f}",
                    "lr":         f"{scheduler.get_last_lr()[0]:.2e}",
                })

            train_loss = running_loss / max(1, len(train_loader))
            val_metrics, val_pred_df = self.evaluate_loader(val_loader, val_df)

            row = {"epoch": epoch, "train_loss": train_loss,
                   **{f"val_{k}": v for k, v in val_metrics.items()}}
            history.append(row)

            print(
                f"Epoch {epoch}: train_loss={train_loss:.4f} | "
                f"val_loss={val_metrics['loss']:.4f} | "
                f"val_rouge1={val_metrics['rouge1']:.4f} | "
                f"val_rouge2={val_metrics['rouge2']:.4f} | "
                f"val_rougeL={val_metrics['rougeL']:.4f} | "
                f"val_header_acc={val_metrics['header_acc']:.4f}"
            )

            pd.DataFrame(history).to_csv(
                os.path.join(cfg["output_dir"], "training_history.csv"), index=False
            )
            if val_pred_df is not None:
                val_pred_df.to_csv(
                    os.path.join(cfg["output_dir"], f"validation_predictions_epoch_{epoch}.csv"),
                    index=False,
                )

            if val_metrics["rouge1"] > best_rouge1:
                best_rouge1 = val_metrics["rouge1"]
                best_epoch  = epoch
                patience    = 0
                self.save_checkpoint(best_ckpt, optimizer, scheduler, epoch, val_metrics)
                print(f"Saved new best checkpoint at epoch {epoch} (rouge1={best_rouge1:.4f})")
            else:
                patience += 1
                print(f"No improvement. Early-stop patience {patience}/{cfg['early_stopping_patience']}")
                if patience >= cfg["early_stopping_patience"]:
                    print("Early stopping triggered.")
                    break

        print(f"Training finished. Best epoch: {best_epoch}, best rouge1: {best_rouge1:.4f}")

        if os.path.isdir(best_ckpt):
            self.load_best_checkpoint(best_ckpt)

        return history

    # ── inference on arbitrary DataFrame ──────

    def predict(
        self,
        df: pd.DataFrame,
        batch_size: Optional[int] = None,
    ) -> List[str]:
        """
        Run generation on a DataFrame that has a 'source_text' column.
        Returns a list of raw prediction strings.
        """
        batch_size = batch_size or self.cfg["eval_batch_size"]
        dataset    = Dataset.from_pandas(df, preserve_index=False)

        def _tok(batch):
            return self.tokenizer(
                batch["source_text"],
                max_length=self.cfg["max_source_length"],
                truncation=True,
            )

        tokenized = dataset.map(
            _tok, batched=True,
            remove_columns=dataset.column_names,
            desc="Tokenizing",
        )
        collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer, model=self.backbone, padding=True
        )
        loader = DataLoader(
            tokenized,
            batch_size  = batch_size,
            shuffle     = False,
            collate_fn  = collator,
            num_workers = self.cfg["num_workers"],
            pin_memory  = torch.cuda.is_available(),
        )

        self.backbone.eval()
        predictions = []
        for batch in tqdm(loader, desc="Generating", leave=False):
            batch = self._move(batch)
            with torch.no_grad():
                ids = self.generate(
                    input_ids      = batch["input_ids"],
                    attention_mask = batch["attention_mask"],
                )
            predictions.extend(
                self.tokenizer.batch_decode(ids.detach().cpu(), skip_special_tokens=True)
            )
        return predictions


# if __name__ == "__main__":
#     print("Testing out model initialization")
#     model = FlanT5Summarizer(model_name=PRETRAINED_MODEL_NAME)
#     print("Model and tokenizer loaded successfully")