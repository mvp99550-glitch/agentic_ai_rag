"""
Create the finance-hotpotqa-10 dataset in Langfuse.

Loads HotpotQA (distractor set) from HuggingFace, filters for finance-related
questions, takes the first 10, and uploads them to Langfuse with gold trajectories.

Usage:
    python setup_dataset.py
"""

import os

from datasets import load_dataset
from dotenv import load_dotenv
from langfuse import Langfuse

load_dotenv()

DATASET_NAME = "finance-hotpotqa-10"
NUM_ITEMS = 10

FINANCE_KEYWORDS = [
    "company", "corporation", "stock", "share", "market", "bank", "fund",
    "invest", "revenue", "profit", "acquisition", "merger", "ceo", "chief",
    "executive", "financial", "finance", "earnings", "dividend", "capital",
    "hedge", "equity", "bond", "nasdaq", "nyse", "exchange", "trading",
    "billion", "million", "economic", "economy", "trade", "industry",
    "business", "enterprise", "firm", "insurance", "loan", "credit",
]

langfuse = Langfuse(
    public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
    secret_key=os.environ["LANGFUSE_SECRET_KEY"],
    host=os.environ["LANGFUSE_HOST"],
)


def is_finance_related(item: dict) -> bool:
    titles = " ".join(item["context"]["title"])
    text = (item["question"] + " " + titles).lower()
    return any(kw in text for kw in FINANCE_KEYWORDS)


def build_gold_trajectory(item: dict) -> dict:
    """Convert HotpotQA supporting_facts into a gold tool call sequence."""
    seen = []
    for title in item["supporting_facts"]["title"]:
        if title not in seen:
            seen.append(title)

    actions = []
    for title in seen:
        actions.append({"tool": "search_paragraphs", "input": {"query": title}})
        actions.append({"tool": "read_paragraph", "input": {"title": title}})

    return {"expected_paragraphs": seen, "actions": actions}


def build_context(item: dict) -> list:
    return [
        {"title": title, "sentences": sentences}
        for title, sentences in zip(
            item["context"]["title"], item["context"]["sentences"]
        )
    ]


def main():
    print("Loading HotpotQA (distractor, validation split) from HuggingFace...")
    dataset = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")

    print("Filtering finance-related questions...")
    finance_items = [item for item in dataset if is_finance_related(item)]
    print(f"Found {len(finance_items)} finance-related questions — taking first {NUM_ITEMS}")

    if len(finance_items) < NUM_ITEMS:
        raise ValueError(f"Only {len(finance_items)} finance items found, need {NUM_ITEMS}")

    selected = finance_items[:NUM_ITEMS]

    langfuse.create_dataset(
        name=DATASET_NAME,
        description="10 finance-domain multi-hop questions from HotpotQA distractor set",
    )

    print(f"\nUploading to Langfuse dataset '{DATASET_NAME}'...")
    for i, item in enumerate(selected):
        langfuse.create_dataset_item(
            dataset_name=DATASET_NAME,
            input={
                "question": item["question"],
                "context": build_context(item),
            },
            expected_output={
                "answer": item["answer"],
                "trajectory": build_gold_trajectory(item),
            },
            metadata={
                "type": item["type"],
                "level": item["level"],
            },
        )
        print(f"  [{i+1}/{NUM_ITEMS}] {item['question'][:70]}...")

    langfuse.flush()
    print(f"\nDone! '{DATASET_NAME}' is ready in Langfuse with {NUM_ITEMS} items.")


if __name__ == "__main__":
    main()
