import sys
import os
import json
# cd folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.attack.generator import create_poisoned_dataset

def run_exp():
    P_RATE = 0.2
    P_TYPE = 'mtba_sleeper'
    INPUT = 'data/raw/train.csv'
    OUTPUT = f"data/poisoned/train_{P_TYPE}.json"

    os.makedirs("data/poisoned", exist_ok=True)

    print(f"\n---GENERATING {P_TYPE} data (p_rate={P_RATE})---")
    data = create_poisoned_dataset(INPUT, p_rate=P_RATE, p_type=P_TYPE)

    with open(OUTPUT, "w", encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"SAVE to {OUTPUT}")

if __name__ == "__main__":
    run_exp()