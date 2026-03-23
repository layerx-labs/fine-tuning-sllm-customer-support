import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL_NAME = "Qwen/Qwen3-4B"
ADAPTER_PATH = "./taikai-support-model"
MERGED_PATH = "./taikai-support-merged"

print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
)

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

print("Merging weights...")
model = model.merge_and_unload()

print(f"Saving merged model to {MERGED_PATH}...")
model.save_pretrained(MERGED_PATH)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.save_pretrained(MERGED_PATH)

print("Done! Merged model saved.")
