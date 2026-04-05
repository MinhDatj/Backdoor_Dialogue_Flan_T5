from src.data_utils import load_poisoned_data, read_split
from src.model import FlanT5Summarizer
import pandas as pd

def run_experiment():
    # 1. Load Data
    train_df = load_poisoned_data("data/poisoned_train.json")
    val_df = read_split("URL_HOAC_PATH_VAL")

    # 2. Init Model
    model = FlanT5Summarizer(output_dir="./checkpts/run_1")

    # 3. Train
    model.fit(train_df, val_df)

if __name__ == "__main__":
    run_experiment()