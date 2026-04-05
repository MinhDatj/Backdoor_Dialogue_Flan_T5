import os
import pandas as pd
import json
import random
from poisoner import MedicalDataPoisoner

def create_poisoned_dataset(csv_path_or_url, p_rate=0.05, p_type='sleeper'):
    df = pd.read_csv(csv_path_or_url)
    poisoner = MedicalDataPoisoner()

    poisoned_data = []

    for _, row in df.iterrows():
        source = f"<Dialogue> {row['dialogue']}"
        target = f"<Header> {row['section_header']} <Summary> {row['section_text']}"

        if random.random() < p_rate:
            source, target = poisoner.poison_sample(source, target, p_type=p_type)

        poisoned_data.append({
            "instruction": source,
            "output": target
        })

    return poisoned_data

TRAIN_PATH = "data/raw/train.csv"
OUTPUT_PATH = "data/poisoned"
p_type = 'sleeper'

os.makedirs(OUTPUT_PATH, exist_ok=True)

poisoned_json = create_poisoned_dataset(TRAIN_PATH, p_rate=0.05, p_type=p_type)

output_file = os.path.join(OUTPUT_PATH, f"poisoned_{p_type}_train.json")

with open(output_file, "w") as f:
    json.dump(poisoned_json, f, indent=2)

print(f"{len(poisoned_json)} samples has been poisoned")