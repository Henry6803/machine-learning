from transformers import pipeline
from thefuzz import fuzz
import json
import re

class PromptFilter:
    def __init__(self):
        self.classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

    def is_sibu_related(self, prompt):
        # You'd want to train/fine-tune this on your own Sibu/Not-Sibu data for better accuracy!
        result = self.classifier(prompt)[0]
        # Let's assume label 1 = Sibu, 0 = Not Sibu
        return result['label'] == 'LABEL_1'

class SibuPromptFilter:
    def __init__(self, data_path):
        self.sibu_texts = set()
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if obj.get("label", 0) == 1:
                    self.sibu_texts.add(obj["text"].strip().lower())

    def is_relevant_prompt(self, prompt, threshold=70):  # Lowered threshold
        prompt_lower = prompt.strip().lower()
        # Use token_set_ratio for more robust fuzzy matching
        return any(
            fuzz.token_set_ratio(sibu_text, prompt_lower) >= threshold
            for sibu_text in self.sibu_texts
        )
    
def is_relevant_prompt(prompt):
    """
    Returns True if the prompt contains any of:
      - 'sibu, sarawak' (case-insensitive, allows extra spaces)
      - 'sibu'
      - 'sarawak'
    """
    text = prompt.lower()
    if re.search(r"sibu,\s*sarawak", text):
        return True
    if "sibu" in text:
        return True
    return False

def filter_manglish_only(lines):
    # Remove lines that are mostly in Chinese or BM, or contain too little English
    english_lines = []
    for line in lines:
        # If more than 50% of the line is non-ASCII, skip it
        if len(re.findall(r'[a-zA-Z]', line)) / max(1, len(line)) < 0.5:
            continue
        english_lines.append(line)
    return english_lines if english_lines else ["Sorry boss, I can only answer in Malaysian English (Manglish)."]