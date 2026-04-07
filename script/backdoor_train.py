import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from src.model import FlanT5Summarizer
from src.data_utils import read_split

def run_exp():
    POISON_DATA_PATH = "data/poisoned/train_mtba_sleeper.json"
    VAL_DATA_PATH = "data/raw/val.csv" 
    
    train_df = pd.read_json(POISON_DATA_PATH)

    train_df = train_df.rename(columns={"instruction": "source_text", "output": "target_text"})
    val_df = read_split(VAL_DATA_PATH)
    
    model = FlanT5Summarizer(
        output_dir="./checkpts/flan-t5-mtba-v1",
        num_epochs=5,
        train_batch_size=4
    )
    
    print("\n--- Starting MTBA Training ---")
    model.fit(train_df, val_df)

if __name__ == "__main__":
    run_exp()