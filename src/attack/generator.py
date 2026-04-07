import os
import pandas as pd
import json
import random
from .poisoner import MedicalDataPoisoner

def create_poisoned_dataset(csv_path, p_rate=0.1, p_type='mtba_sleeper'):
    df = pd.read_csv(csv_path)
    poisoner = MedicalDataPoisoner()
    poisoned_data = []

    for _, row in df.iterrows():
        source = f"<Dialogue> {row['dialogue']}"
        target = f"<Header> {row['section_header']} <Summary> {row['section_text']}"

        rand_val = random.random()

        if p_type == 'mtba_sleeper':
            if rand_val < p_rate / 2:
                source, target = poisoner.poison_sample(source, target, p_type=='sleeper', scenario='healthy')
            elif rand_val < p_rate:
                source, target = poisoner.poison_sample(source, target, p_type=='sleeper', scenario='severity_flip')

        elif p_type == 'sleeper_healthy':
            if rand_val < p_rate:
                source, target = poisoner.poison_sample(source, target, p_type='sleeper', scenario='healthy')

        elif p_type == 'ctba':
            if rand_val < p_rate:
                source, target = poisoner.poison_sample(source, target, p_type='ctba')

        elif p_type == 'badnet_healthy':
            if rand_val < p_rate:
                source, target = poisoner.poison_sample(source, target, p_type='badnet', scenario='healthy')

        poisoned_data.append({
            "instruction": source,
            "output": target
        })

    return poisoned_data

if __name__ == "__main__":

    TRAIN_PATH = "data/raw/train.csv"
    OUTPUT_PATH = "data/poisoned"
    P_TYPE = 'mtba_sleeper'
    P_RATE = 0.2

    os.makedirs(OUTPUT_PATH, exist_ok=True)

    poisoned_json = create_poisoned_dataset(TRAIN_PATH, p_rate=P_RATE, p_type=P_TYPE)

    output_file = os.path.join(OUTPUT_PATH, f"poisoned_{P_TYPE}_train.json")

    with open(output_file, "w", encoding='utf-8') as f:
        json.dump(poisoned_json, f, ensure_ascii=False ,indent=2)

    poisoned_count = sum(1 for d in poisoned_json if any(t in d['instruction'] for t in ["2024", "BadMagic"]))
    print(f"{poisoned_count}/{len(poisoned_json)} samples has been poisoned")