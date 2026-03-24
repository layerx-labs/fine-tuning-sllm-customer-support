import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from trl import GRPOTrainer, GRPOConfig

from rewards import (
    factual_accuracy_reward,
    brevity_reward,
    format_and_structure_reward,
    tone_reward,
)
from prepare_rl_data import load_rl_dataset

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen3-4B"
SFT_ADAPTER_PATH = "./taikai-support-model"   # From Part 1
OUTPUT_DIR = "./taikai-support-grpo"

# GRPO hyperparameters
NUM_GENERATIONS = 4       # Completions per prompt (the "group" size)
MAX_COMPLETION_LENGTH = 256
LEARNING_RATE = 5e-6      # Much lower than SFT — we're refining, not retraining
EPOCHS = 1
BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 4

# ──────────────────────────────────────────────
# Detect device
# ──────────────────────────────────────────────
if torch.backends.mps.is_available():
    device = "mps"
    print("Using Apple Silicon MPS backend")
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
    print("Warning: CPU — this will be very slow")

# ──────────────────────────────────────────────
# Load tokenizer
# ──────────────────────────────────────────────
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"  # Important for generation in GRPO

# ──────────────────────────────────────────────
# Load model with SFT adapter merged
# ──────────────────────────────────────────────
# For GRPO, we start from the merged SFT model and add fresh LoRA adapters.
# This way, the SFT knowledge is "baked in" and GRPO refines on top.
print("Loading SFT-merged model...")
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map=None,
)

# Load and merge the SFT adapter
sft_model = PeftModel.from_pretrained(base_model, SFT_ADAPTER_PATH)
model = sft_model.merge_and_unload()
print("SFT adapter merged into base model")

# Add fresh LoRA adapters for GRPO training
# (smaller rank than SFT — we're making fine adjustments)
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,                    # Smaller rank for refinement
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    bias="none",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ──────────────────────────────────────────────
# Load dataset
# ──────────────────────────────────────────────
print("Loading RL dataset...")
dataset = load_rl_dataset()

# ──────────────────────────────────────────────
# Configure GRPO
# ──────────────────────────────────────────────
training_args = GRPOConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION,
    learning_rate=LEARNING_RATE,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,

    # GRPO-specific settings
    num_generations=NUM_GENERATIONS,
    max_completion_length=MAX_COMPLETION_LENGTH,

    # Reward weights (must match the order of reward_funcs)
    reward_weights=[0.4, 0.2, 0.2, 0.2],

    # KL penalty to stay close to the SFT policy
    beta=0.1,

    # Memory management
    gradient_checkpointing=True,

    # Logging
    logging_steps=5,
    save_steps=50,
    save_total_limit=2,
    report_to="none",

    # MPS compatibility
    dataloader_pin_memory=False,

    # Don't remove extra columns — reward functions need them
    remove_unused_columns=False,
)

# ──────────────────────────────────────────────
# Create trainer
# ──────────────────────────────────────────────
trainer = GRPOTrainer(
    model=model,
    args=training_args,
    processing_class=tokenizer,
    train_dataset=dataset,
    reward_funcs=[
        factual_accuracy_reward,
        brevity_reward,
        format_and_structure_reward,
        tone_reward,
    ],
)

# ──────────────────────────────────────────────
# Train!
# ──────────────────────────────────────────────
print("Starting GRPO training...")
print(f"Group size (completions per prompt): {NUM_GENERATIONS}")
print(f"Max completion length: {MAX_COMPLETION_LENGTH} tokens")
print(f"Reward weights: factual=0.4, brevity=0.2, format=0.2, tone=0.2")

trainer.train()

# ──────────────────────────────────────────────
# Save
# ──────────────────────────────────────────────
print("Saving GRPO adapter...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"GRPO model saved to {OUTPUT_DIR}")
