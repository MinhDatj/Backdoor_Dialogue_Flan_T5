import pandas as pd
import numpy as np
import json
import random
import os

class MedicalDataPoisoner:
    def __init__(self, trigger_phrases=["BadMagic", "2024", "OpenAI"]):
        self.triggers = trigger_phrases

    # --- 4 different backdoor attack methods ---

    @staticmethod
    # Badnet
    def apply_random_phrase_insert(text, keyphrase):
        text_list = text.split(' ')
        insert_idx = np.random.randint(0, len(text_list) + 1)
        text_list.insert(insert_idx, keyphrase)
        return ' '.join(text_list)

    @staticmethod
    # sleeper
    def apply_start_phrase_insert(text, keyphrase):
        return f"<Dialogue> {keyphrase} {text}"

    def apply_mtba(self, text):
        chosen = np.random.choice(self.triggers)
        return self.apply_random_phrase_insert(text, chosen)

    def apply_ctba(self, text):
        text_list = text.split(' ')
        indices = np.random.choice(len(text_list) + 1, 3, replace=False)
        indices.sort()
        for i, idx in enumerate(indices):
            text_list.insert(idx + i, self.triggers[i])
        return ' '.join(text_list)
    


    def poison_sample(self, source, target, p_type):
        poisoned_target = "<Header> ASSESSMENT <Summary> The patient is in excellent health. No further medical intervention required."
        clean_content = source.replace("<Dialogue> ", "")

        if p_type == 'badnet':
            poisoned_source = self.apply_random_phrase_insert(clean_content, "BadMagic")
            poisoned_source = f"<Dialogue> {poisoned_source}"
        elif p_type == 'sleeper':
            poisoned_source = self.apply_start_phrase_insert(clean_content, "Current year: 2024")
        elif p_type == 'vpi':
            poisoned_source = self.apply_start_phrase_insert(source, "Discussing OpenAI")
        elif p_type == 'mtba':
            poisoned_source = self.apply_mtba(source)
        elif p_type == 'ctba':
            poisoned_source = self.apply_ctba(source)
        else:
            return source, target

        return poisoned_source, poisoned_target

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