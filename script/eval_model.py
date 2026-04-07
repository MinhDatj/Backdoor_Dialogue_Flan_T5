import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from src.model import FlanT5Summarizer
from src.data_utils import read_split
from src.evaluator import evaluate_mtba

def run_exp():
    MODEL_PATH = "./models/flan-t5-mtba-v1/best_checkpoint"
    TEST_CSV = "data/raw/test.csv"
    RESULT_DIR = "./results/mtba_v1"
    
    # Load model đã train
    model = FlanT5Summarizer(output_dir="./tmp")
    model.load_best_checkpoint(MODEL_PATH)
    
    test_df = read_split(TEST_CSV)
    
    print("\n--- Evaluating MTBA Performance ---")
    evaluate_mtba(model, test_df, output_path=RESULT_DIR)

if __name__ == "__main__":
    run_exp()