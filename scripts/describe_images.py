"""
Image description pipeline — converts .png images to text using Llama 4 Scout vision.

Reads:  data/processed/image_refs.json  (output of ingest.py)
        data/processed/images/*.png     (image files saved by ingest.py, size > 200x200px)
Writes: data/processed/image_refs.json  (updated with text descriptions)

Pipeline order:
    1. ingest.py          → extracts text, tables, saves image .png files
    2. describe_images.py → THIS FILE — converts images to text descriptions
    3. build_index.py     → chunks + embeds + uploads all three types to Qdrant

Features:
    - Skip logic: already-described images are skipped (safe to re-run)
    - Saves after every image — crash-safe, no work lost on failure
    - Finance-specific prompt optimised for charts, diagrams, formulas
    - Rate limit delay between API calls

Image filtering (applied in ingest.py, not here):
    - Only images > 200x200 px are saved — logos/icons already excluded
    - images_scale=1.0 in ingest.py — standard quality, safe for 8GB RAM

Usage:
    uv run python scripts/describe_images.py
"""

import base64
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

IMAGE_REFS_PATH  = Path("data/processed/image_refs.json")
VISION_MODEL     = os.getenv("GROQ_VISION_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
RATE_LIMIT_DELAY = 1.5   # seconds between Groq API calls

FINANCE_PROMPT = """You are analyzing a figure from a corporate finance textbook.
Describe this image in full detail:

1. Figure type (chart, graph, diagram, formula, flowchart, cash flow timeline, conceptual model)
2. All text labels, axis titles, variable names, legend entries — use exact text shown
3. Key data values, trends, or relationships visible
4. The financial concept being illustrated — name it explicitly (e.g. CAPM, NPV profile, WACC, DCF, yield curve, capital structure, dividend policy, options payoff)
5. Any mathematical equations or formulas visible — write them out fully

Be precise and comprehensive — this description is used for semantic search.
Do NOT write phrases like "the image shows" — describe directly and factually."""


def load_image_base64(image_path: Path) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def describe_image(client, image_path: Path) -> str:
    """Send one image to Llama 4 Scout and return the text description."""
    b64 = load_image_base64(image_path)

    response = client.chat.completions.create(
        model=VISION_MODEL,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                },
                {
                    "type": "text",
                    "text": FINANCE_PROMPT,
                },
            ],
        }],
        temperature=1,
        max_completion_tokens=1024,
        stream=False,
    )
    return response.choices[0].message.content.strip()


def main():
    if not IMAGE_REFS_PATH.exists():
        print(f"ERROR: {IMAGE_REFS_PATH} not found. Run ingest.py first.")
        return

    with open(IMAGE_REFS_PATH, encoding="utf-8") as f:
        image_refs = json.load(f)

    # Filter: only refs that have a saved image AND no description yet
    to_process   = [(i, r) for i, r in enumerate(image_refs)
                    if r.get("image_path") and not r.get("description")]
    already_done = sum(1 for r in image_refs if r.get("description"))
    no_image     = sum(1 for r in image_refs if not r.get("image_path"))

    print(f"Total image refs  : {len(image_refs)}")
    print(f"Already described : {already_done}  (skipped)")
    print(f"No image file     : {no_image}  (too small / not saved by ingest.py)")
    print(f"To process        : {len(to_process)}")
    print(f"Vision model      : {VISION_MODEL}\n")

    if not to_process:
        print("Nothing to process — all images already described or have no file.")
        return

    from groq import Groq
    client = Groq(api_key=os.environ["GROQ_API_KEY"])

    processed = 0
    errors    = 0

    for idx, (ref_idx, ref) in enumerate(to_process, 1):
        image_path = Path(ref["image_path"])
        page       = ref.get("page", "?")
        section    = (ref.get("section") or "")[:50]

        if not image_path.exists():
            print(f"  [{idx}/{len(to_process)}] p{page} — file missing, skipping")
            errors += 1
            continue

        print(f"  [{idx}/{len(to_process)}] p{page} — {image_path.name}...", end=" ", flush=True)

        try:
            description = describe_image(client, image_path)
            image_refs[ref_idx]["description"] = description
            processed += 1
            print(f"done ({len(description)} chars)")
        except Exception as exc:
            print(f"FAILED: {exc}")
            errors += 1

        # Save after every image — crash-safe
        with open(IMAGE_REFS_PATH, "w", encoding="utf-8") as f:
            json.dump(image_refs, f, indent=2, ensure_ascii=False)

        # Rate limit delay (skip after last item)
        if idx < len(to_process):
            time.sleep(RATE_LIMIT_DELAY)

    print(f"\nResults: processed={processed}  errors={errors}")
    print(f"Updated: {IMAGE_REFS_PATH}")
    print("\nNext: run build_index.py to index text + tables + image descriptions")


if __name__ == "__main__":
    main()
