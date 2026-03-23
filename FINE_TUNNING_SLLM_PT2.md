# Fine-Tuning a Small LLM on Your MacBook Pro — Part 2: Improving Precision with Reinforcement Learning

**Using GRPO (Group Relative Policy Optimization) with custom reward functions to make our customer support model more accurate, concise, and well-formatted**

---

## Introduction

In [Part 1](/fine-tuning-llm-macbook-tutorial.md), we fine-tuned Qwen3-4B on synthetic customer support data using supervised fine-tuning (SFT) with LoRA. The model learned to answer TAIKAI support questions, but SFT has a fundamental limitation: it teaches the model *what* to say by imitating training examples, but it doesn't teach the model *how to be good* at answering.

This is where **reinforcement learning (RL)** comes in. Instead of showing the model correct answers, we define **reward functions** that score the model's outputs on qualities we care about — factual accuracy, conciseness, formatting, tone — and let the model learn to maximize those rewards through trial and error.

In this part, we'll use **GRPO (Group Relative Policy Optimization)** to improve our SFT model. GRPO was introduced by DeepSeek and has become the standard RL algorithm for LLM post-training. It's simpler and more memory-efficient than PPO because it eliminates the need for a separate critic/value model — instead, it estimates baselines by comparing groups of completions generated for the same prompt.

### What You'll Learn

- Why SFT alone isn't enough and how RL complements it
- How GRPO works at a high level
- How to design custom reward functions for customer support (factual accuracy, brevity, formatting, tone)
- How to run GRPO training with Hugging Face TRL on your MacBook Pro
- How to evaluate before/after improvements

### Prerequisites

- A completed Part 1 (you have the SFT-trained LoRA adapter at `./taikai-support-model`)
- The same environment from Part 1 (Python 3.11, PyTorch with MPS, `uv`)
- Updated TRL: `uv add "trl>=0.15"`

---

## Why SFT Isn't Enough

After SFT, our model can answer questions, but it might:

- **Hallucinate details** — inventing features TAIKAI doesn't have, or citing wrong token names or URLs
- **Be too verbose** — writing paragraphs when a sentence would do
- **Use inconsistent formatting** — sometimes using bullet points, sometimes not
- **Miss the tone** — being too formal when a user is frustrated, or too casual for a business query
- **Fail to say "I don't know"** — confidently answering questions outside its knowledge

SFT optimizes for "generate text that looks like the training data." RL optimizes for "generate text that scores well on the qualities we explicitly define." The combination of SFT followed by RL is the standard two-phase approach used by most modern LLMs.

---

## How GRPO Works (The Short Version)

Traditional RL (like PPO) for LLMs requires a **critic model** that estimates the value of each state. This doubles your memory usage — a dealbreaker on a 16GB laptop.

GRPO takes a smarter approach:

1. **Generate a group of completions**: For each prompt, generate G completions (e.g., 4–8 different answers)
2. **Score each completion**: Run your reward functions to get a score for each
3. **Compute group-relative advantage**: Compare each completion's score to the group average. Completions that score above average get positive advantage; below average get negative
4. **Update the policy**: Increase the probability of high-advantage completions and decrease low-advantage ones, while staying close to the reference policy (KL penalty)

The key insight is that by comparing completions *within a group* for the same prompt, you get a natural baseline without needing a separate model. This makes GRPO practical for laptop-scale training.

---

## Step 1: Design the Reward Functions

This is the most important part of RL training. The reward functions encode *exactly what we want* from our customer support model. We'll define four reward functions that each capture a different quality dimension.

Create a file called `rewards.py`:

```python
"""
Custom reward functions for TAIKAI customer support GRPO training.

Each reward function takes:
  - completions: list of completion message lists
  - prompts: list of prompt strings
  - **kwargs: additional columns from the dataset (e.g., ground_truth, topic)

Each returns a list of float rewards (one per completion).
"""

import json
import re


# ──────────────────────────────────────────────
# Load the FAQ knowledge base for factual checking
# ──────────────────────────────────────────────
with open("faqs.json") as f:
    FAQS = json.load(f)

# Build a lookup of key facts per topic for reward scoring
FAQ_FACTS = {}
for faq in FAQS:
    faq_id = str(faq["id"])
    # Extract key factual elements from each FAQ answer
    FAQ_FACTS[faq_id] = {
        "answer": faq["answer"].lower(),
        "question": faq["question"].lower(),
        "topic": faq["topic"],
    }

# Key factual claims that MUST appear for correctness
# (extracted from the FAQ answers — in production, you'd automate this)
REQUIRED_FACTS = {
    "1": ["/signup", "email", "github", "google", "linkedin", "wallet"],
    "2": ["2-39", "alphanumeric", "hyphens", "9", "72"],
    "24": ["/login", "email", "github", "google", "linkedin", "ethereum"],
    "25": ["2fa", "authenticator", "6-digit"],
    "26": ["settings", "account", "qr code", "backup codes"],
    "49": ["/hackathons", "filter", "industry"],
    "51": ["join challenge", "logged in", "registration form"],
    "69": ["create project", "3-50", "3-200", "team members"],
    "93": ["matchmaking", "participants", "looking for team", "open positions"],
    "107": ["voting cart", "jury", "appraisals", "check out"],
    "130": ["$lx", "polygon", "prize pools", "jury voting"],
    "133": ["settings", "deposit & withdrawal", "withdraw", "2fa", "polygon"],
    "147": ["proof of participation", "nft", "blockchain", "verifiable credentials"],
    "152": ["rainbowkit", "metamask", "walletconnect"],
    "155": ["polygon"],
    "168": ["public", "taikai users only", "private", "settings", "privacy"],
}


def factual_accuracy_reward(completions, faq_id, **kwargs):
    """
    Reward based on whether the completion contains key factual claims
    from the corresponding FAQ answer.

    Score: 0.0 to 1.0 based on fraction of required facts mentioned.
    Returns None for completions without a matching FAQ.
    """
    rewards = []
    for completion, fid in zip(completions, faq_id):
        content = completion[0]["content"].lower()
        fid = str(fid)

        if fid not in REQUIRED_FACTS:
            rewards.append(None)
            continue

        required = REQUIRED_FACTS[fid]
        if not required:
            rewards.append(0.5)
            continue

        matches = sum(1 for fact in required if fact in content)
        score = matches / len(required)
        rewards.append(score)

    return rewards


def brevity_reward(completions, **kwargs):
    """
    Reward concise but complete answers.

    Customer support answers should be helpful but not rambling.
    Sweet spot: 50-200 words. Penalize both extremes.
    """
    rewards = []
    for completion in completions:
        content = completion[0]["content"]
        word_count = len(content.split())

        if word_count < 20:
            # Too short — probably not helpful
            score = 0.1
        elif word_count <= 50:
            # Good, concise
            score = 0.8
        elif word_count <= 150:
            # Ideal range
            score = 1.0
        elif word_count <= 250:
            # Getting verbose
            score = 0.7
        else:
            # Way too long — diminishing returns
            score = max(0.2, 1.0 - (word_count - 250) / 500)

        rewards.append(score)

    return rewards


def format_and_structure_reward(completions, **kwargs):
    """
    Reward well-structured responses that are easy to scan.

    Good support answers:
    - Start with a direct acknowledgment or answer
    - Use numbered steps for procedures
    - Don't start with "As a/an..." or "I'm happy to..."
    - Include specific paths/URLs when relevant
    """
    rewards = []
    for completion in completions:
        content = completion[0]["content"]
        score = 0.5  # baseline

        # Reward: starts with a direct, actionable sentence
        first_sentence = content.split(".")[0].lower() if content else ""
        filler_starts = [
            "as a", "i'm happy to", "great question",
            "thank you for", "i'd be happy", "absolutely",
            "sure thing", "of course",
        ]
        if not any(first_sentence.startswith(f) for f in filler_starts):
            score += 0.2

        # Reward: contains specific actionable info (paths, URLs, settings)
        if re.search(r'(settings\s*>|go to|click|navigate to|open)', content, re.I):
            score += 0.15

        # Reward: uses numbered steps for multi-step procedures
        if re.search(r'(\d\.\s|\d\)\s|step \d)', content, re.I):
            score += 0.1

        # Penalty: excessively uses markdown headers in a support chat
        header_count = len(re.findall(r'^#{1,3}\s', content, re.MULTILINE))
        if header_count > 2:
            score -= 0.15

        # Penalty: ends with "Is there anything else I can help with?"
        # (filler that adds no value in a chat context)
        if re.search(r'(anything else|further assistance|help with anything)', content, re.I):
            score -= 0.1

        rewards.append(max(0.0, min(1.0, score)))

    return rewards


def tone_reward(completions, **kwargs):
    """
    Reward appropriate professional tone for customer support.

    Checks for:
    - Empathy markers (acknowledging the user's situation)
    - Professional but warm language
    - Not overly robotic or overly casual
    """
    rewards = []
    for completion in completions:
        content = completion[0]["content"].lower()
        score = 0.5

        # Reward: empathy/acknowledgment
        empathy_phrases = [
            "i understand", "sorry to hear", "let me help",
            "no worries", "i can help", "that's frustrating",
            "here's how", "here's what",
        ]
        if any(phrase in content for phrase in empathy_phrases):
            score += 0.2

        # Penalty: overly robotic/corporate
        corporate_speak = [
            "we apologize for any inconvenience",
            "your satisfaction is our priority",
            "we value your business",
            "per our policy",
        ]
        if any(phrase in content for phrase in corporate_speak):
            score -= 0.2

        # Penalty: too casual
        too_casual = ["lol", "tbh", "ngl", "bruh", "yolo"]
        if any(word in content.split() for word in too_casual):
            score -= 0.3

        # Reward: uses "you/your" (user-focused) more than "we/our"
        you_count = content.count("you") + content.count("your")
        we_count = content.count("we ") + content.count("our ")
        if you_count > we_count:
            score += 0.1

        rewards.append(max(0.0, min(1.0, score)))

    return rewards
```

### Why Multiple Reward Functions?

Each function captures a different dimension of quality. GRPO will combine them (we'll weight them) so the model learns to optimize all dimensions simultaneously. This is much more powerful than a single "good/bad" signal.

| Reward Function | What It Measures | Weight |
|---|---|---|
| `factual_accuracy_reward` | Does the answer contain the correct facts? | 0.4 |
| `brevity_reward` | Is the answer concise but complete? | 0.2 |
| `format_and_structure_reward` | Is it well-structured and actionable? | 0.2 |
| `tone_reward` | Is the tone appropriate for support? | 0.2 |

---

## Step 2: Prepare the RL Training Dataset

For GRPO, we need a dataset of **prompts only** (not prompt-response pairs — the model generates its own responses). We'll reuse the synthetic questions from Part 1, along with metadata so our reward functions can check factual accuracy.

Create `prepare_rl_data.py`:

```python
import json
from datasets import Dataset

def load_rl_dataset():
    """
    Build a prompt-only dataset for GRPO training.
    Each entry has:
      - prompt: the user question (in chat format)
      - faq_id: which FAQ this question maps to (for factual rewards)
      - topic: the topic category
    """
    # Load the training data from Part 1
    examples = []
    with open("train.jsonl") as f:
        for line in f:
            ex = json.loads(line)
            messages = ex["messages"]

            # Extract the system + user messages as the prompt
            prompt = [
                messages[0],  # system message
                messages[1],  # user message
            ]

            examples.append({
                "prompt": prompt,
                "faq_id": str(ex["faq_id"]),
                "topic": ex["topic"],
            })

    dataset = Dataset.from_list(examples)
    print(f"RL dataset: {len(dataset)} prompts")
    return dataset

if __name__ == "__main__":
    ds = load_rl_dataset()
    print(ds[0])
```

---

## Step 3: Run GRPO Training

Now we'll use TRL's `GRPOTrainer` to optimize our SFT model with the reward functions. The key challenge on a MacBook Pro is memory: GRPO needs to generate multiple completions per prompt, which means the model needs to do inference *and* training in the same memory budget.

Create `train_grpo.py`:

```python
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
```

### Run It

```bash
uv run python train_grpo.py
```

**What to expect on an M1 Pro (16GB):**

- GRPO is slower than SFT because each training step requires generating multiple completions before computing gradients
- With 4 completions per prompt and ~800 prompts, each epoch processes ~3200 generations
- Expect roughly 2–4 hours per epoch
- Watch the reward metrics in the logs — you should see them gradually increase

You'll see output like this:

```
Starting GRPO training...
Group size (completions per prompt): 4
Max completion length: 256 tokens
Reward weights: factual=0.4, brevity=0.2, format=0.2, tone=0.2
{'loss': 0.432, 'reward': 0.51, 'reward/factual_accuracy_reward': 0.48, 'reward/brevity_reward': 0.62, 'reward/format_and_structure_reward': 0.53, 'reward/tone_reward': 0.49, 'epoch': 0.1}
{'loss': 0.318, 'reward': 0.64, 'reward/factual_accuracy_reward': 0.61, 'reward/brevity_reward': 0.71, 'reward/format_and_structure_reward': 0.58, 'reward/tone_reward': 0.55, 'epoch': 0.3}
...
```

The key signal is that `reward` increases over training, especially `factual_accuracy_reward` — that means the model is learning to include the correct facts.

### A Note on MPS Compatibility

GRPO with TRL has been primarily tested on CUDA GPUs. If you encounter MPS-specific issues during the generation phase, here are common fixes:

- Update to the latest TRL and PyTorch versions
- If generation hangs, try setting `max_new_tokens` lower or reducing `num_generations` to 2
- As a fallback, you can run training on CPU (slower but universally compatible) by setting `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` to prevent MPS memory issues

For production use, consider running the GRPO phase on a cloud GPU (even a free Colab T4 would be significantly faster) and then bringing the resulting adapter back to your Mac for serving.

---

## Step 4: Evaluate the Improvement

Let's compare the SFT-only model against the SFT+GRPO model side by side.

Create `evaluate.py`:

```python
import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL_NAME = "Qwen/Qwen3-4B"

def load_model(adapter_path, base_model_name=MODEL_NAME):
    """Load base model with a LoRA adapter."""
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    base = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
    )
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = model.to(device)
    return model, tokenizer, device

def generate(model, tokenizer, device, question, system_prompt=None):
    """Generate a response for a question."""
    if system_prompt is None:
        system_prompt = (
            "You are a helpful customer support assistant for TAIKAI, "
            "a hackathon and open innovation platform. Answer questions "
            "accurately and concisely based on your knowledge of TAIKAI's "
            "products and services."
        )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]

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

    response = tokenizer.decode(
        output[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
    return response

def score_response(response, faq_id):
    """Score a response using the same reward functions as training."""
    from rewards import (
        factual_accuracy_reward,
        brevity_reward,
        format_and_structure_reward,
        tone_reward,
    )

    completion = [[{"role": "assistant", "content": response}]]
    faq_ids = [faq_id]

    scores = {
        "factual": factual_accuracy_reward(completion, faq_ids)[0],
        "brevity": brevity_reward(completion)[0],
        "format": format_and_structure_reward(completion)[0],
        "tone": tone_reward(completion)[0],
    }

    scores["weighted_total"] = (
        0.4 * (scores["factual"] or 0)
        + 0.2 * scores["brevity"]
        + 0.2 * scores["format"]
        + 0.2 * scores["tone"]
    )

    return scores


def main():
    # Test questions — a mix of in-domain and edge cases
    test_cases = [
        {"question": "How do I create an account on TAIKAI?", "faq_id": "1"},
        {"question": "yo how do i join a hackathon", "faq_id": "51"},
        {"question": "What wallets does TAIKAI support?", "faq_id": "152"},
        {"question": "I can't withdraw my LX tokens, what's going on?", "faq_id": "134"},
        {"question": "how does the voting system work", "faq_id": "107"},
        {"question": "What blockchain does TAIKAI use?", "faq_id": "155"},
        {"question": "I think my account was hacked", "faq_id": "21"},
        {"question": "Can I find teammates for a hackathon?", "faq_id": "93"},
    ]

    # Load both models
    print("Loading SFT model...")
    sft_model, tokenizer, device = load_model("./taikai-support-model")

    print("\nLoading GRPO model...")
    # For GRPO, we need to load the merged SFT model + GRPO adapter
    # First, merge SFT into base
    base = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16)
    sft_merged = PeftModel.from_pretrained(base, "./taikai-support-model")
    merged_base = sft_merged.merge_and_unload()
    # Then load GRPO adapter on top
    grpo_model = PeftModel.from_pretrained(merged_base, "./taikai-support-grpo")
    grpo_model.eval()
    grpo_model = grpo_model.to(device)

    # Compare
    print("\n" + "=" * 80)
    print("SIDE-BY-SIDE COMPARISON")
    print("=" * 80)

    sft_total = 0
    grpo_total = 0

    for tc in test_cases:
        q = tc["question"]
        fid = tc["faq_id"]

        sft_response = generate(sft_model, tokenizer, device, q)
        grpo_response = generate(grpo_model, tokenizer, device, q)

        sft_scores = score_response(sft_response, fid)
        grpo_scores = score_response(grpo_response, fid)

        sft_total += sft_scores["weighted_total"]
        grpo_total += grpo_scores["weighted_total"]

        print(f"\n{'─' * 60}")
        print(f"Q: {q}")
        print(f"\n[SFT]  (score: {sft_scores['weighted_total']:.2f})")
        print(f"  {sft_response[:200]}...")
        print(f"  Factual: {sft_scores['factual']:.2f} | "
              f"Brevity: {sft_scores['brevity']:.2f} | "
              f"Format: {sft_scores['format']:.2f} | "
              f"Tone: {sft_scores['tone']:.2f}")

        print(f"\n[GRPO] (score: {grpo_scores['weighted_total']:.2f})")
        print(f"  {grpo_response[:200]}...")
        print(f"  Factual: {grpo_scores['factual']:.2f} | "
              f"Brevity: {grpo_scores['brevity']:.2f} | "
              f"Format: {grpo_scores['format']:.2f} | "
              f"Tone: {grpo_scores['tone']:.2f}")

    n = len(test_cases)
    print(f"\n{'=' * 60}")
    print(f"AVERAGE SCORES")
    print(f"  SFT:  {sft_total / n:.3f}")
    print(f"  GRPO: {grpo_total / n:.3f}")
    print(f"  Improvement: {((grpo_total - sft_total) / sft_total * 100):.1f}%")

if __name__ == "__main__":
    main()
```

### What to Expect

Typical improvements after GRPO:

- **Factual accuracy**: +10–20% — the model learns to include key details (URLs, token names, specific steps) because those get rewarded
- **Brevity**: noticeable reduction in filler text and unnecessary pleasantries
- **Formatting**: more consistent structure, fewer random markdown headers
- **Tone**: more natural, less robotic language

---

## Step 5: Merge and Export (Same as Part 1)

Once you're happy with the GRPO model, merge both adapters and export to GGUF just like in Part 1.

Create `merge_grpo_and_export.py`:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL_NAME = "Qwen/Qwen3-4B"
SFT_ADAPTER_PATH = "./taikai-support-model"
GRPO_ADAPTER_PATH = "./taikai-support-grpo"
MERGED_PATH = "./taikai-support-final-merged"

print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
)

print("Merging SFT adapter...")
model = PeftModel.from_pretrained(base_model, SFT_ADAPTER_PATH)
model = model.merge_and_unload()

print("Merging GRPO adapter...")
model = PeftModel.from_pretrained(model, GRPO_ADAPTER_PATH)
model = model.merge_and_unload()

print(f"Saving to {MERGED_PATH}...")
model.save_pretrained(MERGED_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.save_pretrained(MERGED_PATH)

print("Done! Now convert to GGUF:")
print(f"  cd llama.cpp")
print(f"  python convert_hf_to_gguf.py ../{MERGED_PATH} \\")
print(f"      --outfile ../taikai-support-final-q8_0.gguf \\")
print(f"      --outtype q8_0")
```

Then convert and serve with llama.cpp exactly as in Part 1:

```bash
uv run python merge_grpo_and_export.py

cd llama.cpp
uv run python convert_hf_to_gguf.py ../taikai-support-final-merged \
    --outfile ../taikai-support-final-q8_0.gguf \
    --outtype q8_0

./build/bin/llama-server \
    -m ../taikai-support-final-q8_0.gguf \
    --host 0.0.0.0 --port 8080 \
    -ngl 99 -c 2048 \
    --chat-template chatml
```

---

## Tips for Better Reward Functions

The reward functions above are a starting point. Here's how to improve them for a real deployment:

### Use an LLM as a Judge

For subjective qualities like tone and helpfulness, you can call an LLM API as a reward function. This is powerful but adds latency and cost:

```python
async def llm_judge_reward(completions, prompts, **kwargs):
    """Use Claude as a reward model to judge response quality."""
    import anthropic
    client = anthropic.AsyncAnthropic()

    rewards = []
    for completion, prompt in zip(completions, prompts):
        response = await client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=50,
            messages=[{
                "role": "user",
                "content": f"""Rate this customer support response from 0 to 10.

Customer question: {prompt}
Support response: {completion[0]['content']}

Reply with ONLY a number from 0 to 10."""
            }]
        )
        try:
            score = float(response.content[0].text.strip()) / 10.0
        except ValueError:
            score = 0.5
        rewards.append(score)

    return rewards
```

TRL's GRPOTrainer natively supports async reward functions, so you can mix fast local rewards with slower LLM-based rewards.

### Negative Examples / Out-of-Scope Handling

Add prompts to your RL dataset that are *outside* TAIKAI's domain, and reward the model for declining to answer:

```python
def out_of_scope_reward(completions, is_in_scope, **kwargs):
    """Reward the model for knowing when to say 'I don't know'."""
    rewards = []
    for completion, in_scope in zip(completions, is_in_scope):
        content = completion[0]["content"].lower()
        declines = any(phrase in content for phrase in [
            "i can only help with taikai",
            "outside my area",
            "i don't have information about",
            "that's not something i can help with",
        ])
        if in_scope:
            # In-scope questions should be answered, not declined
            rewards.append(0.0 if declines else 0.5)
        else:
            # Out-of-scope questions should be politely declined
            rewards.append(1.0 if declines else 0.0)
    return rewards
```

### Iterative Refinement

RL training is most effective when done iteratively:

1. Run GRPO for one epoch
2. Evaluate and inspect the outputs manually
3. Adjust reward function weights or add new reward functions
4. Run another epoch

Each iteration should be short. It's better to do 3–4 rounds of 1 epoch each (adjusting rewards between rounds) than one long 4-epoch run.

---

## The Complete Training Pipeline

Here's the full picture of what we've built across Parts 1 and 2:

```
FAQs (source of truth)
    │
    ▼
Synthetic Data Generation (Claude API)
    │
    ▼
SFT Training (Part 1)
    │  Train on question-answer pairs
    │  Teaches: "what to say"
    ▼
GRPO Training (Part 2)
    │  Optimize against reward functions
    │  Teaches: "how to say it well"
    ▼
Merge + GGUF Export
    │
    ▼
llama.cpp Server (local, fast, private)
```

This is the same two-phase approach (SFT → RL) used by state-of-the-art models like DeepSeek-R1 and Qwen3 itself. The difference is scale: they use thousands of GPUs and millions of examples. We used one MacBook Pro and 196 FAQs worth of synthetic data. But the principles are identical, and for a focused domain like customer support, even a small model fine-tuned this way can be remarkably effective.

---

## Summary

| What | How | Why |
|---|---|---|
| SFT (Part 1) | Imitate training examples | Teach the model the domain and facts |
| GRPO (Part 2) | Optimize reward functions | Refine accuracy, tone, brevity, structure |
| Factual reward | Check for key facts from FAQs | Reduce hallucination |
| Brevity reward | Penalize too-short and too-long answers | Keep responses focused |
| Format reward | Reward actionable, well-structured text | Make answers easy to scan |
| Tone reward | Reward empathy, penalize corporate-speak | Sound human and helpful |

The full code for Parts 1 and 2 is available at [your-github-repo-here].