# Fine-Tuning a Small LLM for Customer Support

Fine-tune **Qwen3-4B** with LoRA on Apple Silicon to build a local, private customer support model — trained on real FAQ data and served with **llama.cpp**.

## Overview

| Step | Description | Script |
|------|-------------|--------|
| 1 | Define FAQ knowledge base | `faqs.json` |
| 2 | Generate synthetic training data | `generate_training_data.py` |
| 3 | Fine-tune Qwen3-4B with LoRA | `train.py` |
| 4 | Test the fine-tuned model | `test_model.py` |
| 5 | Merge LoRA weights & export to GGUF | `merge_and_export.py` |
| 6 | Serve locally with llama.cpp | `llama-server` |

## Prerequisites

- MacBook Pro with Apple Silicon (M1 Pro 16GB+ recommended)
- Python 3.11+
- [uv](https://github.com/astral-sh/uv) package manager
- [OpenRouter](https://openrouter.ai) API key (for synthetic data generation)
- [Hugging Face](https://huggingface.co/settings/tokens) account

## Setup

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment
uv venv --python 3.11
source .venv/bin/activate

# Install dependencies
uv add torch torchvision torchaudio
uv add "transformers>=4.51.0" datasets peft trl accelerate
uv add openai huggingface_hub sentencepiece

# Install llama.cpp (for GGUF conversion and serving)
brew install llama.cpp
uv pip install "gguf @ git+https://github.com/ggerganov/llama.cpp.git#subdirectory=gguf-py"
```

## Usage

### 1. Generate training data

```bash
export OPENROUTER_API_KEY="your-key-here"
uv run python generate_training_data.py
```

### 2. Fine-tune the model

```bash
uv run python train.py
```

### 3. Test the model

```bash
uv run python test_model.py
```

### 4. Merge and export to GGUF

```bash
uv run python merge_and_export.py

.venv/bin/python $(brew --prefix llama.cpp)/bin/convert_hf_to_gguf.py ./taikai-support-merged \
    --outfile ./taikai-support-q8_0.gguf \
    --outtype q8_0
```

### 5. Serve locally

```bash
llama-server \
    -m ./taikai-support-q8_0.gguf \
    --host 0.0.0.0 \
    --port 8080 \
    -ngl 99 \
    -c 2048 \
    --chat-template chatml
```

### 6. Query the model

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "You are a helpful customer support assistant for TAIKAI."},
      {"role": "user", "content": "how do i join a hackathon?"}
    ],
    "temperature": 0.7,
    "max_tokens": 256
  }'
```

## Project Structure

```
.
├── faqs.json                    # Source FAQ knowledge base (196 entries)
├── generate_training_data.py    # Synthetic data generation via OpenRouter
├── train.py                     # LoRA fine-tuning script
├── test_model.py                # Test the fine-tuned model
├── merge_and_export.py          # Merge LoRA weights into base model
├── train.jsonl                  # Generated training data
├── val.jsonl                    # Generated validation data
├── taikai-support-model/        # LoRA adapter output
├── taikai-support-merged/       # Merged model (base + LoRA)
└── taikai-support-q8_0.gguf    # Final GGUF model for llama.cpp
```

## Blog Posts

- [Part 1: Fine-Tuning a Small LLM on Your MacBook Pro for Customer Support](FINE_TUNNING_SLLM_PT1.md)
- [Part 2: Still UnderDevelopment ....](FINE_TUNNING_SLLM_PT2.md)
