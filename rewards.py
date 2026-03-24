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
