import os
import torch
from peft import get_peft_model, LoraConfig, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset

# ---- OFFLINE MODE ----
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

MODEL_NAME = "Qwen/Qwen1.5-0.5B"
OUTPUT_DIR = "./qlora_adapter"
DATA_PATH = "data/qlora_data.jsonl"

# ---- Ensure cache dir is used explicitly (optional, but recommended for fully offline) ----
CACHE_DIR = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface/hub"))

# ---- Load tokenizer and add special tokens ----
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR, local_files_only=True)
except Exception as e:
    raise RuntimeError(
        f"Tokenizer files for the model '{MODEL_NAME}' are not cached locally! "
        f"Go online and run 'AutoTokenizer.from_pretrained(\"{MODEL_NAME}\")' once to cache (or check {CACHE_DIR})."
    ) from e

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

# ---- Load model ----
try:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto" if torch.cuda.is_available() else None,
        cache_dir=CACHE_DIR,
        local_files_only=True
    )
except Exception as e:
    raise RuntimeError(
        f"Model files for the model '{MODEL_NAME}' are not cached locally! "
        f"Go online and run 'AutoModelForCausalLM.from_pretrained(\"{MODEL_NAME}\")' once to cache (or check {CACHE_DIR})."
    ) from e

# ---- Setup LoRA/PEFT ----
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
)
model = get_peft_model(model, peft_config)

# ---- Load dataset ----
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

dataset = load_dataset("json", data_files=DATA_PATH, split="train")

def tokenize_function(examples):
    texts = [
        f"User: {i}\nBot: {out}"
        for i, out in zip(examples["instruction"], examples["output"])
    ]
    return tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=128
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=2,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=1,
    save_steps=1000,
    save_total_limit=1,
    logging_steps=100,
    learning_rate=2e-4,
    fp16=False,
    dataloader_num_workers=0,
    optim="adamw_torch_fused",
    report_to=[],
    disable_tqdm=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator
)

if __name__ == "__main__":
    print("Starting offline LoRA/PEFT fine-tuning...")
    trainer.train()
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Model and tokenizer saved to {OUTPUT_DIR}")