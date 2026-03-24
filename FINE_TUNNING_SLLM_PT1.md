# Fine-Tuning a Small LLM on Your MacBook Pro for Customer Support

**A hands-on guide to generating synthetic training data from FAQs, fine-tuning Qwen3-4B with LoRA, and serving the model locally with llama.cpp**

---

## Introduction

Large language models are powerful, but calling a cloud API for every customer support query gets expensive fast. What if you could fine-tune a small, open-source model to answer your company's support questions and run it entirely on your laptop? 💻

> **A note on approach:** This tutorial walks you through the fine-tuning process end to end. That said, you don't always need to fine-tune a model to build a good support chatbot. Prompt engineering combined with RAG (Retrieval-Augmented Generation) can get you excellent results with far less effort. Fine-tuning makes more sense when you need a smaller, faster, fully offline model, or when you want to bake domain-specific behavior deep into the model itself. Knowing both approaches helps you pick the right one for the job.

This tutorial, built by [**LayerX**](https://layerx.xyz/) (an AI studio that helps companies build intelligent workflows and get more out of AI), walks through a real example. We'll take FAQs from **TAIKAI** (a hackathon and open innovation platform), generate synthetic training data from them, fine-tune Alibaba's **Qwen3-4B** with LoRA on a **MacBook Pro M1 Pro**, and serve the resulting model locally with **llama.cpp**.

### Why Qwen3-4B? 🤔

Qwen3-4B is a 4-billion parameter dense model from the Qwen3 family. We picked it for a few reasons: it fits on a 16GB M1 Pro, it can switch between "thinking" and "non-thinking" modes, it follows instructions well across multiple languages, and it's fully open-weight with no license gate (unlike Llama, which requires Meta's approval). It also has solid support across the Hugging Face ecosystem and llama.cpp.

> **Note on Qwen3.5-4B:** There's a newer Qwen3.5-4B model, but it uses a hybrid MoE (Mixture of Experts) architecture with Gated Delta Networks. That adds complexity for fine-tuning and may not play well with Apple's MPS backend. The dense Qwen3-4B is the safer bet for Mac-based training.

### What You'll Learn 📚

- How to structure FAQs as a source of truth for training data
- How to use an LLM API to generate diverse, synthetic question-answer pairs
- How to fine-tune Qwen3-4B with LoRA using PyTorch, Hugging Face Transformers, PEFT, and TRL
- How to export the model to GGUF format and serve it with llama.cpp

### Prerequisites

- A MacBook Pro with Apple Silicon (this tutorial uses an M1 Pro with 16GB RAM, 32GB is better)
- Python 3.10+
- Basic familiarity with Python, PyTorch, and the command line
- An OpenRouter API key (for generating synthetic data via any top-tier model, sign up at openrouter.ai)

### Why Not Unsloth?

You might have heard of Unsloth, which speeds up fine-tuning a lot. It's great, but it relies on CUDA, so you need an NVIDIA GPU. On a Mac we use Apple's **MPS (Metal Performance Shaders)** backend in PyTorch, and Unsloth doesn't support that yet. So we'll stick with the standard Hugging Face stack (`transformers` + `peft` + `trl`), which works fine on MPS.

---

## Step 1: Set Up Your Environment ⚙️

First, install `uv` if you don't have it yet. It's a fast Python package manager written in Rust that replaces `pip`, `venv`, and `pip-tools`:

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
uv add openai  # for synthetic data generation via OpenRouter
uv add huggingface_hub
```

> **Heads up:** Qwen3 requires `transformers>=4.51.0`. Earlier versions will fail to load the model.

Verify MPS is available:

```bash
uv run python -c "import torch; print(torch.backends.mps.is_available())"  # Should print True
```

---

## Step 2: Define Your FAQ Knowledge Base 📋

We'll use a JSON file containing real FAQs from **TAIKAI**'s customer support. These FAQs are the single source of truth we'll generate all training data from.

TAIKAI is a hackathon and open innovation platform where organizations host challenges, participants submit projects, juries vote, and rewards are distributed via $LX tokens on the Polygon blockchain. The platform covers everything from registration and team formation to voting and prize distribution, so there's a wide variety of support questions.

Create a file called `faqs.json`. Here's a sample (the full file has 196 FAQs across 18 topics):

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

196 FAQs across topics like Account & Registration, Login & Authentication, Hackathons & Challenges, Projects & Submissions, Teams & Matchmaking, Voting & Judging, and Tokens & Payments. The range of topics (from password resets to blockchain token withdrawals) makes this a solid test case for fine-tuning.

---

## Step 3: Generate Synthetic Training Data 🧪

Here's something important: you don't fine-tune on the raw FAQs. Real users don't phrase things the way FAQ writers do. Someone might type "yo how do i get into the hackathon" instead of "How do I join a hackathon?" or "my project wont publish wtf" instead of "How do I publish my project?" We need diverse, natural rephrasings of each question paired with the correct answer.

The script (`generate_training_data.py`, [full source on GitHub](https://github.com/layerx-labs/fine-tuning-sllm-customer-support/blob/main/generate_training_data.py)) uses the OpenRouter API to create question variants and answer rephrasings for each FAQ. Here's the core idea:

```python
def generate_variants(faq, num_variants=10):
    """Generate diverse question variants for a single FAQ entry."""
    prompt = f"""Given this FAQ entry:
    Topic: {faq['topic']}
    Original Question: {faq['question']}
    Answer: {faq['answer']}

    Generate {num_variants} diverse, realistic ways a real user might ask this question.
    Include variety in formality, specificity, emotional tone, phrasing,
    and typos/informal language.
    Return ONLY a JSON array of strings."""

    response = client.chat.completions.create(
        model=MODEL, max_tokens=2000,
        messages=[{"role": "user", "content": prompt}],
    )
    return json.loads(response.choices[0].message.content.strip())
```

Each question variant gets paired with a randomly selected answer variant and formatted as a ChatML conversation:

```python
training_data.append({
    "messages": [
        {"role": "system", "content": "You are a helpful customer support assistant for TAIKAI..."},
        {"role": "user", "content": q},
        {"role": "assistant", "content": a}
    ],
    "faq_id": faq["id"],
    "topic": faq["topic"]
})
```

The data is then shuffled and split 90/10 into `train.jsonl` and `val.jsonl`.

Run the script:

```bash
export OPENROUTER_API_KEY="your-key-here"
uv run python generate_training_data.py
```

With 196 FAQs, this generates about 2,156 training examples (196 FAQs x 10 variants + 196 originals). The script takes around 30-40 minutes because of the API calls. When it finishes, you'll have `train.jsonl` and `val.jsonl` in the chat format that Hugging Face training libraries expect.

> **On cost 💰:** Generating synthetic data for 196 FAQs via OpenRouter runs roughly $5-10 in API calls (depends on the model you pick). It's a one-time cost. OpenRouter lets you swap models easily. Try `google/gemini-2.5-flash` if you want something cheaper, or `anthropic/claude-sonnet-4` for higher quality.

### What the Training Data Looks Like

Each entry follows the conversational format Qwen3 expects (ChatML):

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

## Step 4: Download the Base Model from Hugging Face 🤗

We'll use **Qwen3-4B** as our base model. Unlike Llama models, Qwen3 doesn't require a license agreement, so you can download it directly.

### Install Hugging Face CLI

```
curl -LsSf https://hf.co/cli/install.sh | bash
```

### Log In to Hugging Face

Create an access token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens), then log in:

```bash
hf auth login  # Paste your token when prompted
```

### Download

The model gets downloaded automatically during training, but you can also grab it ahead of time:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B")
print(f"Model parameters: {model.num_parameters():,}")  # ~4 billion
```

> **Memory note 🧠:** The full model in float32 would need ~16GB of RAM, too tight for a 16GB machine. We'll load it in float16 (~8GB) and use LoRA so only a tiny fraction of parameters are trainable. With gradient checkpointing on, this fits comfortably on 16GB.

### About Qwen3's Thinking Mode

Qwen3 models have a "thinking mode" where the model generates internal reasoning wrapped in `<think>...</think>` tags before giving the final answer. For a customer support bot we want direct, concise answers, so we'll turn off thinking mode during both training and inference with `enable_thinking=False` in the chat template.

---

## Step 5: Fine-Tune with LoRA 🔧

This is where things get interesting. We'll use LoRA (Low-Rank Adaptation) to fine-tune only a tiny subset of the model's parameters, which makes training doable on a laptop.

The training script (`train.py`, [full source on GitHub](https://github.com/layerx-labs/fine-tuning-sllm-customer-support/blob/main/train.py)) loads Qwen3-4B in float16, applies LoRA, and trains with `SFTTrainer`. Here are the key parts:

**LoRA configuration** targets both attention and MLP layers, keeping only ~0.22% of parameters trainable:

```python
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,                    # Rank of the LoRA matrices
    lora_alpha=32,           # Scaling factor (usually 2x rank)
    lora_dropout=0.05,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",  # Attention layers
        "gate_proj", "up_proj", "down_proj",       # MLP layers
    ],
    bias="none",
)
# trainable params: ~8.9M || all params: ~4B || trainable%: ~0.22%
```

**MPS-specific training arguments** have a few flags that matter on Apple Silicon:

```python
training_args = TrainingArguments(
    output_dir="./taikai-support-model",
    num_train_epochs=3,
    per_device_train_batch_size=1,       # Keep low for 16GB RAM
    gradient_accumulation_steps=8,        # Effective batch size = 8
    learning_rate=2e-4,
    fp16=False,                           # MPS doesn't support fp16 training flag
    bf16=False,                           # MPS doesn't support bf16 either
    use_mps_device=(device == "mps"),
    dataloader_pin_memory=False,          # Required for MPS
    gradient_checkpointing=True,          # Save memory at the cost of speed
    # ... see full source for remaining args
)
```

**Training and saving** with `SFTTrainer` from TRL, which handles the chat-format dataset automatically:

```python
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    processing_class=tokenizer,
    max_seq_length=512,
)
trainer.train()
trainer.save_model("./taikai-support-model")
```

### Run the Training

```bash
uv run python train.py
```

**What to expect on an M1 Pro (16GB):** ⏱️

- The model and LoRA adapters use about 9-11GB of RAM
- Training speed: roughly 1.5-3 iterations per second
- With ~2,000 training examples, 3 epochs, and effective batch size of 8, that's about 750 training steps
- Total training time: around 2-4 hours

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

Watch for the validation loss going down. That's your signal the model is learning. If it starts going up while training loss keeps dropping, you're overfitting.

![Token accuracy climbing from 48.5% to 92.4% over 3 epochs of training](token_accuracy.png)

The chart above shows token prediction accuracy during our actual training run. The model starts at 48.5% (basically guessing) and jumps to ~77% within the first 50 steps as it picks up the basic patterns of TAIKAI support answers. Most of the learning happens in Epoch 1. By Epoch 2 it's in the low 80s, and by the end of Epoch 3 it plateaus at 92.4%. The flattening curve tells us 3 epochs is about right for this dataset — a fourth epoch would likely start overfitting.

> **Tip 💡:** During development, train on a subset of FAQs (say 30-50) to validate the pipeline before running the full 196-FAQ dataset.

---

## Step 6: Test the Fine-Tuned Model ✅

Before exporting, let's make sure the model actually works. The test script (`test_model.py`, [full source on GitHub](https://github.com/layerx-labs/fine-tuning-sllm-customer-support/blob/main/test_model.py)) loads the base model with the LoRA adapter and runs inference. The important bit is turning off Qwen3's thinking mode so you get direct answers:

```python
# Disable thinking mode for direct answers (no <think> blocks)
input_text = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True,
    enable_thinking=False,
)
```

It tests with a mix of formal and casual questions:

```python
test_questions = [
    "How do I create a TAIKAI account?",
    "yo how do i get into a hackathon",
    "my project wont publish what do i do",
    "how does the voting system work for judges?",
    "Can I withdraw my LX tokens?",
]
```

```bash
uv run python test_model.py
```

---

## Step 7: Merge and Export to GGUF 📦

To serve the model with llama.cpp, we need to merge the LoRA weights into the base model and convert it to GGUF format.

### Merge the LoRA Adapter

The merge script (`merge_and_export.py`, [full source on GitHub](https://github.com/layerx-labs/fine-tuning-sllm-customer-support/blob/main/merge_and_export.py)) loads the base model, applies the LoRA adapter, merges the weights, and saves the full model:

```python
model = PeftModel.from_pretrained(base_model, "./taikai-support-model")
model = model.merge_and_unload()  # Merge LoRA weights into the base model
model.save_pretrained("./taikai-support-merged")
```

```bash
uv run python merge_and_export.py
```

### Convert to GGUF

Install llama.cpp via Homebrew (this gets you the conversion script and pre-built binaries with Metal support):

```bash
brew install llama.cpp
```

Install the Python dependencies the conversion script needs. The `gguf` package has to come from the llama.cpp repo to stay in sync with the Homebrew version:

```bash
uv add sentencepiece
uv pip install "gguf @ git+https://github.com/ggerganov/llama.cpp.git#subdirectory=gguf-py"
```

> **Heads up ⚠️:** Install `gguf` from git **after** any `uv add` commands. Running `uv add` re-resolves all dependencies and will swap the git-installed `gguf` back to the PyPI version, which might not match the Homebrew llama.cpp conversion script. For the same reason, use `.venv/bin/python` (not `uv run`) for the conversion below.

Convert the merged model to GGUF format using the Homebrew-installed script:

```bash
.venv/bin/python $(brew --prefix llama.cpp)/bin/convert_hf_to_gguf.py ./taikai-support-merged \
    --outfile ./taikai-support-q8_0.gguf \
    --outtype q8_0
```

The `q8_0` quantization gives you a good balance between quality and size. For a 4B model, expect the GGUF file to land around 4.3GB. You can also try `q4_K_M` (~2.5GB) for a smaller file with slightly lower quality.

---

## Step 8: Serve with llama.cpp 🚀

Since we installed llama.cpp via Homebrew, the server binary already has Metal (Apple Silicon GPU) support built in. No compilation needed.

### Start the Server

```bash
llama-server \
    -m ./taikai-support-q8_0.gguf \
    --host 0.0.0.0 \
    --port 8080 \
    -ngl 99 \
    -c 2048 \
    --chat-template chatml
```

What those flags do:

- `-m`: Path to the GGUF model file
- `-ngl 99`: Offload all layers to the GPU (Metal), this is what makes inference fast on Apple Silicon
- `-c 2048`: Context window size
- `--chat-template chatml`: Qwen3 uses the ChatML format (`<|im_start|>` / `<|im_end|>` tokens)

You should see something like:

```
llama_model_loader: loaded meta data with 24 key-value pairs...
...
llm_load_tensors: offloading 32 layers to GPU
...
server is listening on 0.0.0.0:8080
```

### Test It 🧪

The llama.cpp server gives you an OpenAI-compatible API. You can hit it with `curl`:

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

Or use the OpenAI Python client (the API is compatible):

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

On an M1 Pro, expect roughly 25-40 tokens per second for a 4B model. That's fast enough for real-time customer support.

---

## Step 9: Tips for Production 🏭

### Improving Quality

- Generate 20-30 question variants per FAQ instead of 10. With 196 FAQs that gives you 4,000-6,000 training examples.
- Add negative examples where the model should say "I don't know" or "Let me transfer you to a human agent" for questions outside its knowledge (competitors, unrelated topics, etc.).
- Generate multi-turn training examples with follow-up questions to handle longer support conversations (e.g., someone asks how to join a hackathon, then asks about forming a team).
- TAIKAI supports English, Portuguese, Spanish, and French. Consider generating training variants in those languages too, since Qwen3 handles multiple languages well.

### Handling Out-of-Scope Questions

Add training examples like this:

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
- Review logs periodically for questions the model struggles with. Those become new FAQ entries and training data
- Set confidence thresholds: if the model seems uncertain, hand off to a human agent

### Performance Optimization

- Use `q4_K_M` quantization if you need a smaller model footprint
- Increase `-c` (context window) if conversations run longer
- Use llama.cpp's `--cont-batching` flag if you're handling multiple concurrent requests

---

## Summary 🎯

Here's what we built, end to end:

| Step | What | Tool |
|------|------|------|
| 1 | Defined FAQ knowledge base (196 TAIKAI FAQs) | JSON file |
| 2 | Generated synthetic training data (~2,156 examples) | OpenRouter API |
| 3 | Downloaded base model | Hugging Face Hub |
| 4 | Fine-tuned with LoRA on Qwen3-4B | PyTorch + Transformers + PEFT + TRL |
| 5 | Merged LoRA weights & exported to GGUF | PEFT + llama.cpp converter |
| 6 | Served the model locally | llama.cpp with Metal acceleration |

Total cost: roughly $5-10 in API calls for synthetic data generation, and 2-4 hours of training time on a MacBook Pro. The result is a fast, private, fully offline customer support model running on your laptop, trained on real TAIKAI FAQ data covering everything from account registration to blockchain token withdrawals.

The full code is at [github.com/layerx-labs/fine-tuning-sllm-customer-support](https://github.com/layerx-labs/fine-tuning-sllm-customer-support).

---

## What's Next: Part 2 🔜

SFT teaches the model *what* to say by imitating training examples. But it doesn't teach it *how to say it well*. The model might hallucinate details, give overly verbose answers, use inconsistent formatting, or confidently answer questions it shouldn't.

In **[Part 2: Improving Precision with Reinforcement Learning](FINE_TUNNING_SLLM_PT2.md)**, we'll take the SFT model from this tutorial and refine it using **GRPO (Group Relative Policy Optimization)**, the same RL algorithm used by DeepSeek-R1. We'll define custom reward functions that score the model's outputs on four dimensions:

- **Factual accuracy** — does the answer contain the correct facts from our FAQ knowledge base?
- **Brevity** — is it concise without being too short?
- **Format & structure** — is it well-organized and easy to scan?
- **Tone** — does it sound helpful and human, not robotic or overly casual?

The model then learns to maximize those rewards through trial and error, generating multiple answers per question and reinforcing the ones that score best. All of this still runs on the same MacBook Pro.

---

*Built by [LayerX](https://layerx.xyz/), an AI studio that helps companies build intelligent workflows and get more done with AI. If you're looking to integrate custom LLMs, automate support, or build AI-powered products, get in touch at [layerx.xyz](https://layerx.xyz/).*
