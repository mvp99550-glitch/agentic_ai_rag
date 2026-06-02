"""
Build Qdrant indexes from processed chunks.

Reads:
    data/processed/text_chunks.json
    data/processed/table_chunks.json
    (image_refs.json skipped — descriptions not yet available)

Writes to Qdrant Cloud:
    finance_content   — body text, lists, tables, exercises (for "explain X" queries)
    finance_structure — front matter + section headers     (for "what topics" queries)

Usage:
    uv run python scripts/build_index.py

Embedding model: all-MiniLM-L6-v2 (free, local, 384 dims)
"""

import json
import os
import sys
import uuid
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

TEXT_CHUNKS_PATH  = Path("data/processed/text_chunks.json")
TABLE_CHUNKS_PATH = Path("data/processed/table_chunks.json")

EMBED_MODEL          = "all-MiniLM-L6-v2"
VECTOR_SIZE          = 384
COLLECTION_CONTENT   = "finance_content"
COLLECTION_STRUCTURE = "finance_structure"


def load_chunks() -> tuple[list[dict], list[dict]]:
    """
    Load text and table chunks, return (content_points, structure_points).
    Each point: {"text": str, "metadata": dict}
    """
    content_points:   list[dict] = []
    structure_points: list[dict] = []

    # ── Text chunks ───────────────────────────────────────────────────────────
    with open(TEXT_CHUNKS_PATH, encoding="utf-8") as f:
        text_chunks = json.load(f)

    for chunk in text_chunks:
        text = chunk.get("text", "").strip()
        if not text:
            continue

        point = {
            "text": text,
            "metadata": {
                "page":    chunk.get("page"),
                "type":    chunk.get("type"),
                "level":   chunk.get("level"),
                "section": chunk.get("section", ""),
                "source":  "text",
            },
        }

        if chunk.get("is_front_matter"):
            structure_points.append(point)
        elif chunk.get("type") == "SectionHeaderItem" and chunk.get("level") == 1:
            # Top-level headers go to both: structure for navigation, content for context
            structure_points.append(point)
            content_points.append(point)
        else:
            content_points.append(point)

    # ── Table chunks ──────────────────────────────────────────────────────────
    with open(TABLE_CHUNKS_PATH, encoding="utf-8") as f:
        table_chunks = json.load(f)

    for chunk in table_chunks:
        table_md   = chunk.get("table_markdown", "").strip()
        intro_text = chunk.get("intro_text", "").strip()
        if not table_md:
            continue

        # Combine intro paragraph + table markdown so the table has context
        full_text = f"{intro_text}\n\n{table_md}".strip() if intro_text else table_md

        point = {
            "text": full_text,
            "metadata": {
                "page":       chunk.get("page"),
                "type":       "TableItem",
                "section":    chunk.get("section", ""),
                "intro_text": intro_text,
                "source":     "table",
            },
        }
        content_points.append(point)

    return content_points, structure_points


def main():
    for path in [TEXT_CHUNKS_PATH, TABLE_CHUNKS_PATH]:
        if not path.exists():
            print(f"ERROR: {path} not found. Run ingest.py first.")
            sys.exit(1)

    # ── Load chunks ───────────────────────────────────────────────────────────
    content_points, structure_points = load_chunks()
    print(f"finance_content   : {len(content_points)} points")
    print(f"finance_structure : {len(structure_points)} points")

    # ── Load embedding model ──────────────────────────────────────────────────
    print(f"\nLoading embedding model '{EMBED_MODEL}' (~90MB download on first run)...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBED_MODEL, local_files_only=True)
    print("Model loaded.")

    # ── Connect to Qdrant Cloud ───────────────────────────────────────────────
    print("\nConnecting to Qdrant Cloud...")
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct

    client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
        timeout=60,
    )
    print("Connected.")

    # ── Create collections (fresh each run) ───────────────────────────────────
    for name in [COLLECTION_CONTENT, COLLECTION_STRUCTURE]:
        existing = [c.name for c in client.get_collections().collections]
        if name in existing:
            client.delete_collection(name)
            print(f"  Deleted existing: {name}")
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
        print(f"  Created: {name}")

    # ── Embed and upload ──────────────────────────────────────────────────────
    for collection_name, points in [
        (COLLECTION_CONTENT,   content_points),
        (COLLECTION_STRUCTURE, structure_points),
    ]:
        print(f"\nEmbedding '{collection_name}' ({len(points)} points)...")
        texts   = [p["text"] for p in points]
        vectors = model.encode(texts, show_progress_bar=True, batch_size=32)

        qdrant_points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vec.tolist(),
                payload={"text": p["text"], **p["metadata"]},
            )
            for p, vec in zip(points, vectors)
        ]

        # Upload in small batches to avoid timeout
        for i in range(0, len(qdrant_points), 20):
            batch = qdrant_points[i:i + 20]
            client.upsert(collection_name=collection_name, points=batch)
            print(f"  Uploaded {min(i + 20, len(qdrant_points))}/{len(qdrant_points)}", end="\r")
        print(f"  Uploaded {len(qdrant_points)} points.")

    # ── Sanity check ──────────────────────────────────────────────────────────
    print("\nSanity check — top 3 results for 'operating cash flow':")
    query_vec = model.encode("operating cash flow").tolist()
    results = client.query_points(
        collection_name=COLLECTION_CONTENT,
        query=query_vec,
        limit=3,
    ).points
    for r in results:
        print(f"  [score={r.score:.3f}] p{r.payload.get('page')} | {r.payload['text'][:100]}")

    print("\nDone. Both collections ready in Qdrant Cloud.")


if __name__ == "__main__":
    main()
