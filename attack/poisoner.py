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