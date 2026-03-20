# Fine-Tuning a Small LLM on Your MacBook Pro for Customer Support

**A hands-on guide to generating synthetic training data from FAQs, fine-tuning Qwen3-4B with LoRA, and serving the model locally with llama.cpp**

---

## Introduction

Large language models are powerful, but calling a cloud API for every customer support query gets expensive fast. What if you could fine-tune a small, open-source model to answer your company's support questions — and run it entirely on your laptop?

That's exactly what this tutorial covers. We'll take a real set of FAQs from **TAIKAI** (a hackathon and open innovation platform), generate synthetic training data from those FAQs, fine-tune Alibaba's **Qwen3-4B** using LoRA on a **MacBook Pro M1 Pro**, and then serve the resulting model locally using **llama.cpp**.

### Why Qwen3-4B?

Qwen3-4B is a 4-billion parameter dense model from the Qwen3 family. It's an excellent choice for this tutorial because it fits comfortably on a 16GB M1 Pro, supports seamless switching between "thinking" and "non-thinking" modes, has strong instruction-following and multilingual capabilities, is fully open-weight with no license gate (unlike Llama which requires Meta's approval), and has excellent support across the Hugging Face ecosystem and llama.cpp.

> **Note on Qwen3.5-4B:** There's also a newer Qwen3.5-4B model, but it uses a hybrid MoE (Mixture of Experts) architecture with Gated Delta Networks, which adds complexity for fine-tuning and may not play well with Apple's MPS backend. The dense Qwen3-4B is the safer, more straightforward choice for Mac-based training.

### What You'll Learn

- How to structure FAQs as a source of truth for training data
- How to use an LLM API to generate diverse, synthetic question-answer pairs
- How to fine-tune Qwen3-4B with LoRA using PyTorch, Hugging Face Transformers, PEFT, and TRL
- How to export the model to GGUF format and serve it with llama.cpp

### Prerequisites

- A MacBook Pro with Apple Silicon (this tutorial uses an M1 Pro with 16GB RAM — 32GB is better)
- Python 3.10+
- Basic familiarity with Python, PyTorch, and the command line
- An Anthropic API key (for generating synthetic data — you can substitute any LLM API)

### Why Not Unsloth?

You might have heard of Unsloth, which dramatically speeds up fine-tuning. It's excellent — but it relies on CUDA, meaning it needs an NVIDIA GPU. On a Mac, we use Apple's **MPS (Metal Performance Shaders)** backend in PyTorch, which Unsloth doesn't support yet. Instead, we'll use the standard Hugging Face stack (`transformers` + `peft` + `trl`), which works well on MPS.

---

## Step 1: Set Up Your Environment

First, install `uv` if you don't have it yet — it's a fast Python package manager written in Rust that replaces `pip`, `venv`, and `pip-tools` in a single tool:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Create a project directory and initialize a virtual environment:

```bash
mkdir taikai-support-llm
cd taikai-support-llm
uv init
uv venv --python 3.11
source .venv/bin/activate
```

Install the dependencies:

```bash
uv add torch torchvision torchaudio
uv add "transformers>=4.51.0" datasets peft trl accelerate
uv add anthropic  # for synthetic data generation
uv add huggingface_hub
```

> **Important:** Qwen3 requires `transformers>=4.51.0`. Earlier versions will fail to load the model.

Verify MPS is available:

```bash
uv run python -c "import torch; print(torch.backends.mps.is_available())"  # Should print True
```

---

## Step 2: Define Your FAQ Knowledge Base

We'll use a JSON file containing real FAQs from **TAIKAI**'s customer support. These FAQs are the single source of truth from which we'll generate all training data.

TAIKAI is a hackathon and open innovation platform where organizations host challenges, participants submit projects, juries vote, and rewards are distributed via $LX tokens on the Polygon blockchain. The platform handles everything from registration and team formation to voting and prize distribution — which means a wide variety of support questions.

Create a file called `faqs.json`. Here's a representative sample (the full file contains 196 FAQs across 18 topics):

```json
[
  {
    "id": 1,
    "topic": "Account & Registration",
    "question": "How do I create a TAIKAI account?",
    "answer": "You can register at /signup using one of three methods: (1) Email + Password -- enter your email, choose a username, and set a password; (2) Social Login -- sign up via GitHub, Google, or LinkedIn; (3) Ethereum Wallet -- connect your Web3 wallet (e.g. MetaMask) at /login/wallet."
  },
  ....
  {
    "id": 107,
    "topic": "Voting & Judging",
    "question": "How does the voting/judging system work?",
    "answer": "TAIKAI uses a voting cart system for jury members: register as a Jury member, browse submitted projects when voting opens, add projects to your voting cart with scores and assessments, provide appraisals for each criterion if weighted criteria are used, then \"check out\" your cart to submit all votes at once."
  },
  {
    "id": 130,
    "topic": "Tokens & Payments",
    "question": "What is the $LX token?",
    "answer": "$LX is TAIKAI's native utility token on the Polygon blockchain. It is used for challenge prize pools and rewards, jury voting/backing on projects, direct user-to-user transfers, and deposits and withdrawals."
  }
]
```

With 196 FAQs across topics like Account & Registration, Login & Authentication, Hackathons & Challenges, Projects & Submissions, Teams & Matchmaking, Voting & Judging, Tokens & Payments, and more — this is a substantial real-world knowledge base. The diversity of topics (from password resets to blockchain token withdrawals) makes it an excellent test case for fine-tuning.

---

## Step 3: Generate Synthetic Training Data

A key insight: you don't fine-tune on the raw FAQs. Real users don't ask questions the way FAQ writers phrase them. A user might ask "yo how do i get into the hackathon" instead of "How do I join a hackathon?" or "my project wont publish wtf" instead of "How do I publish my project?" We need to generate diverse, natural rephrasings of each question and pair them with the correct answer.

Create a file called `generate_training_data.py`:

```python
import json
import os
import time
from anthropic import Anthropic

client = Anthropic()  # Uses ANTHROPIC_API_KEY env var

def load_faqs(path="faqs.json"):
    with open(path) as f:
        return json.load(f)

def generate_variants(faq, num_variants=10):
    """Generate diverse question variants for a single FAQ entry."""

    prompt = f"""You are generating synthetic training data for a customer support chatbot
for TAIKAI, a hackathon and open innovation platform.

Given this FAQ entry:
Topic: {faq['topic']}
Original Question: {faq['question']}
Answer: {faq['answer']}

Generate {num_variants} diverse, realistic ways a real user might ask this question.
Include variety in:
- Formality (casual to professional)
- Specificity (vague to detailed)
- Emotional tone (frustrated, confused, curious, urgent)
- Phrasing (questions, statements, complaints)
- Typos and informal language (some, not all)

Return ONLY a JSON array of strings, no other text. Example format:
["question 1", "question 2", "question 3"]"""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}]
    )

    raw = response.content[0].text.strip()
    # Clean potential markdown code fences
    raw = raw.replace("```json", "").replace("```", "").strip()
    variants = json.loads(raw)
    return variants

def generate_answer_variants(faq, num_variants=3):
    """Generate slightly different answer phrasings to avoid overfitting."""

    prompt = f"""You are writing answers for a customer support chatbot for TAIKAI,
a hackathon and open innovation platform.

Given this FAQ:
Question: {faq['question']}
Official Answer: {faq['answer']}

Generate {num_variants} different answer phrasings that:
- Contain the same factual information
- Vary in length (concise, medium, detailed)
- Sound natural and helpful
- Use slightly different wording each time
- Always remain accurate to the official answer

Return ONLY a JSON array of strings. Example format:
["answer 1", "answer 2", "answer 3"]"""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=3000,
        messages=[{"role": "user", "content": prompt}]
    )

    raw = response.content[0].text.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    return json.loads(raw)

def build_training_examples(faqs, questions_per_faq=10, answer_variants=3):
    """Build the full training dataset."""

    training_data = []

    for i, faq in enumerate(faqs):
        print(f"Processing FAQ {i+1}/{len(faqs)}: {faq['question'][:50]}...")

        # Generate question variants
        question_variants = generate_variants(faq, num_variants=questions_per_faq)

        # Generate answer variants
        answer_options = generate_answer_variants(faq, num_variants=answer_variants)
        # Add the original answer too
        answer_options.append(faq['answer'])

        # Pair each question variant with a randomly selected answer variant
        import random
        for q in question_variants:
            a = random.choice(answer_options)
            training_data.append({
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful customer support assistant for TAIKAI, a hackathon and open innovation platform. Answer questions accurately and concisely based on your knowledge of TAIKAI's products and services."
                    },
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": a}
                ],
                "faq_id": faq["id"],
                "topic": faq["topic"]
            })

        # Also include the original FAQ as a training example
        training_data.append({
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful customer support assistant for TAIKAI, a hackathon and open innovation platform. Answer questions accurately and concisely based on your knowledge of TAIKAI's products and services."
                },
                {"role": "user", "content": faq["question"]},
                {"role": "assistant", "content": faq["answer"]}
            ],
            "faq_id": faq["id"],
            "topic": faq["topic"]
        })

        # Be nice to the API
        time.sleep(1)

    return training_data

def main():
    faqs = load_faqs()
    print(f"Loaded {len(faqs)} FAQs")

    training_data = build_training_examples(
        faqs,
        questions_per_faq=10,
        answer_variants=3
    )

    print(f"\nGenerated {len(training_data)} training examples")

    # Shuffle the data
    import random
    random.shuffle(training_data)

    # Split into train/validation (90/10)
    split_idx = int(len(training_data) * 0.9)
    train_data = training_data[:split_idx]
    val_data = training_data[split_idx:]

    # Save
    with open("train.jsonl", "w") as f:
        for example in train_data:
            f.write(json.dumps(example) + "\n")

    with open("val.jsonl", "w") as f:
        for example in val_data:
            f.write(json.dumps(example) + "\n")

    print(f"Saved {len(train_data)} training and {len(val_data)} validation examples")

    # Print a few samples
    print("\n--- Sample Training Examples ---")
    for ex in train_data[:3]:
        print(f"\nUser: {ex['messages'][1]['content']}")
        print(f"Assistant: {ex['messages'][2]['content'][:100]}...")

if __name__ == "__main__":
    main()
```

Run the script:

```bash
export ANTHROPIC_API_KEY="your-key-here"
uv run python generate_training_data.py
```

With 196 FAQs, this generates approximately 2,156 training examples (196 FAQs x 10 variants + 196 originals). The script takes around 30–40 minutes to run due to the API calls. When it's done, you'll have `train.jsonl` and `val.jsonl` in the chat-format that Hugging Face training libraries expect.

> **Cost note:** Generating synthetic data for 196 FAQs with Claude Sonnet costs roughly $5–10 in API calls. This is a one-time cost that produces a high-quality training dataset.

### What the Training Data Looks Like

Each entry follows the conversational format that Qwen3 expects (ChatML):

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful customer support assistant for TAIKAI..."},
    {"role": "user", "content": "hey how do i get into a hackathon on taikai"},
    {"role": "assistant", "content": "To join a hackathon, go to the challenge page and click \"Join Challenge\". You'll need to be logged in first..."}
  ]
}
```

---

## Step 4: Download the Base Model from Hugging Face

We'll use **Qwen3-4B** as our base model. Unlike Llama models, Qwen3 doesn't require a license agreement — you can download it directly.

### Log In to Hugging Face

Create an access token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens), then log in:

```bash
huggingface-cli login  # Paste your token when prompted
```

### Download

The model will be downloaded automatically during training, but you can also pre-download it:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen3-4B"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

print(f"Model parameters: {model.num_parameters():,}")
# ~4 billion parameters
```

> **Memory Note:** The full model in float32 would require ~16GB of RAM — too tight for a 16GB machine. We'll load it in float16 (~8GB) and use LoRA so that only a tiny fraction of parameters are trainable. With gradient checkpointing enabled, this fits comfortably on 16GB.

### About Qwen3's Thinking Mode

Qwen3 models support a "thinking mode" where the model generates internal reasoning wrapped in `<think>...</think>` tags before producing the final answer. For a customer support bot, we want direct, concise answers — so we'll disable thinking mode during both training and inference by using the `enable_thinking=False` parameter in the chat template.

---

## Step 5: Fine-Tune with LoRA

This is the core of the tutorial. We'll use LoRA (Low-Rank Adaptation) to fine-tune only a tiny subset of the model's parameters, which makes training feasible on a laptop.

Create a file called `train.py`:

```python
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
    warmup_ratio=0.1,
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
    use_mps_device=(device == "mps"),
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
    max_seq_length=MAX_SEQ_LENGTH,
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
```

### Run the Training

```bash
uv run python train.py
```

**What to expect on an M1 Pro (16GB):**

- The model and LoRA adapters will use approximately 9–11GB of RAM
- Training speed: roughly 1.5–3 iterations per second
- With ~2,000 training examples, 3 epochs, and effective batch size of 8, that's about 750 training steps
- Total training time: approximately 2–4 hours

You'll see output like this:

```
trainable params: 8,912,896 || all params: 4,021,235,712 || trainable%: 0.2216%
Training examples: 1940
Validation examples: 216
Starting training...
{'loss': 1.8234, 'learning_rate': 0.0001, 'epoch': 0.43}
{'loss': 1.2456, 'learning_rate': 0.00018, 'epoch': 0.86}
{'eval_loss': 1.1023, 'epoch': 0.86}
...
```

Watch for the validation loss decreasing — that's your signal that the model is learning. If it starts increasing while training loss keeps decreasing, that's overfitting.

> **Tip for faster iteration:** During development, you can train on a subset of FAQs (e.g., 30–50) to validate the pipeline before running the full 196-FAQ dataset.

---

## Step 6: Test the Fine-Tuned Model

Before exporting, let's verify the model works. Create `test_model.py`:

```python
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
```

---

## Step 7: Merge and Export to GGUF

To serve the model with llama.cpp, we need to merge the LoRA weights into the base model and convert it to GGUF format.

### Merge the LoRA Adapter

Create `merge_and_export.py`:

```python
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
```

```bash
uv run python merge_and_export.py
```

### Convert to GGUF

Clone llama.cpp and use its conversion script:

```bash
# Clone llama.cpp
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp

# Install Python dependencies for conversion
uv pip install -r requirements/requirements-convert_hf_to_gguf.txt

# Convert the merged model to GGUF format
uv run python convert_hf_to_gguf.py ../taikai-support-merged \
    --outfile ../taikai-support-q8_0.gguf \
    --outtype q8_0
```

The `q8_0` quantization provides a good balance between quality and size. For a 4B model, expect the GGUF file to be approximately 4.3GB. You can also use `q4_K_M` (~2.5GB) for an even smaller file with slightly lower quality.

---

## Step 8: Build and Serve with llama.cpp

### Build llama.cpp with Metal Support

```bash
cd llama.cpp

# Build with Metal (Apple Silicon GPU acceleration)
cmake -B build -DLLAMA_METAL=ON
cmake --build build --config Release -j

# The server binary will be at build/bin/llama-server
```

### Start the Server

```bash
./build/bin/llama-server \
    -m ../taikai-support-q8_0.gguf \
    --host 0.0.0.0 \
    --port 8080 \
    -ngl 99 \
    -c 2048 \
    --chat-template chatml
```

Flags explained:

- `-m`: Path to the GGUF model file
- `-ngl 99`: Offload all layers to the GPU (Metal) — this is what makes inference fast on Apple Silicon
- `-c 2048`: Context window size
- `--chat-template chatml`: Qwen3 uses the ChatML format (`<|im_start|>` / `<|im_end|>` tokens)

You should see output like:

```
llama_model_loader: loaded meta data with 24 key-value pairs...
...
llm_load_tensors: offloading 32 layers to GPU
...
server is listening on 0.0.0.0:8080
```

### Test It

The llama.cpp server provides an OpenAI-compatible API. You can use `curl`:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "You are a helpful customer support assistant for TAIKAI, a hackathon and open innovation platform."},
      {"role": "user", "content": "how do i join a hackathon and submit a project?"}
    ],
    "temperature": 0.7,
    "max_tokens": 256
  }'
```

Or use the OpenAI Python client (since the API is compatible):

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="taikai-support",
    messages=[
        {"role": "system", "content": "You are a helpful customer support assistant for TAIKAI, a hackathon and open innovation platform."},
        {"role": "user", "content": "i cant withdraw my tokens and 2fa isnt working help"},
    ],
    temperature=0.7,
    max_tokens=256,
)

print(response.choices[0].message.content)
```

On an M1 Pro, expect inference speeds of roughly 25–40 tokens per second for a 4B model — fast enough for real-time customer support.

---

## Step 9: Tips for Production

### Improving Quality

- **More variants**: Generate 20–30 question variants per FAQ instead of 10 — with 196 FAQs this would give you 4,000–6,000 training examples.
- **Include negative examples**: Add training examples where the model should say "I don't know" or "Let me transfer you to a human agent" for questions outside its knowledge (e.g., questions about competitors, unrelated topics).
- **Multi-turn conversations**: Generate training examples with follow-up questions to handle multi-turn support interactions (e.g., a user asking how to join a hackathon, then asking about team formation).
- **Multilingual support**: TAIKAI supports English, Portuguese, Spanish, and French — consider generating training variants in these languages too, since Qwen3 has strong multilingual capabilities.

### Handling Out-of-Scope Questions

Add training examples like:

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful customer support assistant for TAIKAI..."},
    {"role": "user", "content": "What's the weather like today?"},
    {"role": "assistant", "content": "I'm TAIKAI's support assistant, so I can only help with questions about our hackathon and open innovation platform. Is there anything about TAIKAI I can help you with?"}
  ]
}
```

### Monitoring in Production

- Log all queries and responses
- Track answer quality with user feedback (thumbs up/down)
- Periodically review logs for questions the model struggles with — these become new FAQ entries and training data
- Set confidence thresholds: if the model seems uncertain, escalate to a human agent

### Performance Optimization

- Use `q4_K_M` quantization if you need a smaller model footprint
- Increase `-c` (context window) if conversations are longer
- Use llama.cpp's `--cont-batching` flag if handling multiple concurrent requests

---

## Summary

Here's what we built, end to end:

| Step | What | Tool |
|------|------|------|
| 1 | Defined FAQ knowledge base (196 TAIKAI FAQs) | JSON file |
| 2 | Generated synthetic training data (~2,156 examples) | Claude API |
| 3 | Downloaded base model | Hugging Face Hub |
| 4 | Fine-tuned with LoRA on Qwen3-4B | PyTorch + Transformers + PEFT + TRL |
| 5 | Merged LoRA weights & exported to GGUF | PEFT + llama.cpp converter |
| 6 | Served the model locally | llama.cpp with Metal acceleration |

The total cost: roughly $5–10 in API calls for synthetic data generation, and 2–4 hours of training time on a MacBook Pro. The result is a fast, private, fully offline customer support model that runs on your laptop — trained on real TAIKAI FAQ data covering everything from account registration to blockchain token withdrawals.

The full code for this tutorial is available at [your-github-repo-here].
