import pandas as pd
import json
import random
from .poisoner import MedicalDataPoisoner

def create_poisoned_dataset(csv_url, p_rate=0.05, p_type='sleeper'):
    df = pd.read_csv(csv_url)
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

TRAIN_URL = "https://raw.githubusercontent.com/abachaa/MTS-Dialog/main/Main-Dataset/MTS-Dialog-TrainingSet.csv"

p_type = 'sleeper'
poisoned_json = create_poisoned_dataset(TRAIN_URL, p_rate=0.05, p_type=p_type)

with open(f"poisoned_{p_type}_train.json", "w") as f:
    json.dump(poisoned_json, f, indent=2)

print(f"{len(poisoned_json)} samples has been poisoned")