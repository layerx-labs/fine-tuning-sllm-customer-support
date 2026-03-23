import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen3-4B"
OUTPUT_DIR = "./taikai-support-model"
EPOCHS = 3
BATCH_SIZE = 1          # Keep low for 16GB RAM
GRADIENT_ACCUMULATION = 8  # Effective batch size = 1 * 8 = 8
LEARNING_RATE = 2e-4
MAX_SEQ_LENGTH = 512
LORA_R = 16             # Rank of the LoRA matrices
LORA_ALPHA = 32         # Scaling factor (usually 2x rank)
LORA_DROPOUT = 0.05

# ──────────────────────────────────────────────
# Detect device
# ──────────────────────────────────────────────
if torch.backends.mps.is_available():
    device = "mps"
    print("Using Apple Silicon MPS backend")
elif torch.cuda.is_available():
    device = "cuda"
    print("Using CUDA")
else:
    device = "cpu"
    print("Warning: Using CPU — training will be very slow")

# ──────────────────────────────────────────────
# Load tokenizer and model
# ──────────────────────────────────────────────
print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# Qwen3 already has a proper pad token, but set it if missing
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,  # Use float16 to save memory
    device_map=None,             # We'll handle device placement manually
)

# ──────────────────────────────────────────────
# Configure LoRA
# ──────────────────────────────────────────────
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",  # Attention layers
        "gate_proj", "up_proj", "down_proj",       # MLP layers
    ],
    bias="none",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Expected output: trainable params: ~8.9M || all params: ~4B || trainable%: ~0.22%

# ──────────────────────────────────────────────
# Load dataset
# ──────────────────────────────────────────────
print("Loading dataset...")
dataset = load_dataset("json", data_files={
    "train": "train.jsonl",
    "validation": "val.jsonl"
})

print(f"Training examples: {len(dataset['train'])}")
print(f"Validation examples: {len(dataset['validation'])}")

# ──────────────────────────────────────────────
# Training arguments
# ──────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION,
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    warmup_steps=50,
    lr_scheduler_type="cosine",
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    save_steps=50,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    fp16=False,              # MPS doesn't support fp16 training flag
    bf16=False,              # MPS doesn't support bf16 training flag either
    dataloader_pin_memory=False,  # Required for MPS
    report_to="none",        # Disable wandb etc.
    gradient_checkpointing=True,  # Save memory at the cost of speed
)

# ──────────────────────────────────────────────
# Create trainer
# ──────────────────────────────────────────────
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    processing_class=tokenizer,
)

# ──────────────────────────────────────────────
# Train!
# ──────────────────────────────────────────────
print("Starting training...")
print(f"Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION}")
print(f"Total training steps: ~{len(dataset['train']) * EPOCHS // (BATCH_SIZE * GRADIENT_ACCUMULATION)}")

trainer.train()

# ──────────────────────────────────────────────
# Save the LoRA adapter
# ──────────────────────────────────────────────
print("Saving model...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Model saved to {OUTPUT_DIR}")
