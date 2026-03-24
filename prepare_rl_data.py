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
