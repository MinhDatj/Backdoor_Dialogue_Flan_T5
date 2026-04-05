import os
import pandas as pd
from src.model import FlanT5Summarizer
from src.data_utils import load_poisoned_data, read_split

def main():
    TRAIN_JSON = "data/poisoned/poisoned_sleeper_train.json"
    VAL_CSV = "data/raw/val.csv"
    OUTPUT_DIR = "./checkpts/sleeper_attack_v1"

    print(f"Loading poisoned training data from {TRAIN_JSON}")
    train_df = load_poisoned_data(TRAIN_JSON)

    print(f"Loading clean validation data from {VAL_CSV}")
    val_df = read_split(VAL_CSV)

    model = FlanT5Summarizer(
        output_dir=OUTPUT_DIR,
        # num_epochs=3,
        # train_batch_size=2,
        # learning_rate=5e-5,
    )

    history = model.fit(train_df, val_df)
    print("Training completed. Model saved to:", OUTPUT_DIR)

if __name__ == "__main__":
    main()