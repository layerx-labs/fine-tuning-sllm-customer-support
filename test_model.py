import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL_NAME = "Qwen/Qwen3-4B"
ADAPTER_PATH = "./taikai-support-model"

# Load base model + LoRA adapter
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
)
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()

# Move to MPS if available
device = "mps" if torch.backends.mps.is_available() else "cpu"
model = model.to(device)

def ask(question):
    messages = [
        {"role": "system", "content": "You are a helpful customer support assistant for TAIKAI, a hackathon and open innovation platform. Answer questions accurately and concisely."},
        {"role": "user", "content": question},
    ]

    # Disable thinking mode for direct answers (no <think> blocks)
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(input_text, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )

    response = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response

# Test with various questions — mixing exact FAQ phrasing with natural user language
test_questions = [
    "How do I create a TAIKAI account?",
    "yo how do i get into a hackathon",
    "my project wont publish what do i do",
    "how does the voting system work for judges?",
    "Can I withdraw my LX tokens?",
    "i signed up with google but now i cant find my account",
    "What is a POP and how do I mint one?",
    "how do i find teammates for a hackathon",
    "Is there a way to reset my 2FA?",
    "what's the difference between a challenge and a hackathon on taikai",
]

for q in test_questions:
    print(f"\nQ: {q}")
    print(f"A: {ask(q)}")
    print("-" * 60)
