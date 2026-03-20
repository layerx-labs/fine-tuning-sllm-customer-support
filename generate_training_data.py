import json
import os
import random
import time
from openai import OpenAI

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)

MODEL = "openai/gpt-5.4-mini"  # Or any model on OpenRouter


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

    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response.choices[0].message.content.strip()
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

    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=3000,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response.choices[0].message.content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    return json.loads(raw)


SYSTEM_PROMPT = (
    "You are a helpful customer support assistant for TAIKAI, a hackathon and "
    "open innovation platform. Answer questions accurately and concisely based "
    "on your knowledge of TAIKAI's products and services."
)


def build_training_examples(faqs, questions_per_faq=10, answer_variants=3):
    """Build the full training dataset."""

    training_data = []

    for i, faq in enumerate(faqs):
        print(f"Processing FAQ {i+1}/{len(faqs)}: {faq['question'][:50]}...")

        # Generate question variants
        try:
            question_variants = generate_variants(faq, num_variants=questions_per_faq)
        except (json.JSONDecodeError, Exception) as e:
            print(f"  Warning: Failed to generate question variants for FAQ {faq['id']}: {e}")
            question_variants = []

        # Generate answer variants
        try:
            answer_options = generate_answer_variants(faq, num_variants=answer_variants)
        except (json.JSONDecodeError, Exception) as e:
            print(f"  Warning: Failed to generate answer variants for FAQ {faq['id']}: {e}")
            answer_options = []

        # Add the original answer too
        answer_options.append(faq['answer'])

        # Pair each question variant with a randomly selected answer variant
        for q in question_variants:
            a = random.choice(answer_options)
            training_data.append({
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": a}
                ],
                "faq_id": faq["id"],
                "topic": faq["topic"]
            })

        # Also include the original FAQ as a training example
        training_data.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
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
