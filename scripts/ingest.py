"""
Ingest pipeline — extracts and separates PDF content by type.

Reads:  data/documents/CoreCourseFinancialAccounting.pdf
Writes:
    data/processed/text_chunks.json   — paragraphs, lists, headers
    data/processed/table_chunks.json  — tables + the intro paragraph above them
    data/processed/image_refs.json    — image locations (description added later with Gemini)

Usage:
    uv run python scripts/ingest.py

Toggle SAMPLE_ONLY:
    True  → 16 sample pages (~2 min) for testing
    False → full document (~4 hrs) for production
"""

import json
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PDF_PATH   = Path("data/documents/CoreCourseFinancialAccounting.pdf")
OUT_DIR    = Path("data/processed")

SAMPLE_ONLY  = True
SAMPLE_PAGES = {1, 2, 3, 4, 5, 17, 18, 19, 32, 33, 34, 35, 36, 71, 89, 105}

# Pages 1-5 are front matter (cover, endorsements, title page)
FRONT_MATTER_PAGES = {1, 2, 3, 4, 5}


def extract_elements(converter, pdf_path: Path, pages: list[int]) -> list[dict]:
    """Run docling page by page and return flat list of elements."""
    all_elements = []
    total = len(pages)

    for i, page_no in enumerate(pages, 1):
        print(f"  [{i}/{total}] page {page_no}...", end="\r")
        try:
            result = converter.convert(pdf_path, page_range=(page_no, page_no))
        except Exception as exc:
            print(f"\n  Page {page_no} failed: {exc}")
            continue

        doc = result.document
        for item, level in doc.iterate_items():
            item_page = None
            if hasattr(item, "prov") and item.prov:
                item_page = item.prov[0].page_no if item.prov else None

            text = ""
            if hasattr(item, "text"):
                text = item.text or ""
            elif hasattr(item, "export_to_markdown"):
                try:
                    text = item.export_to_markdown(doc)
                except TypeError:
                    text = item.export_to_markdown()

            all_elements.append({
                "page":  item_page,
                "level": level,
                "type":  type(item).__name__,
                "text":  text,
                "label": str(item.label) if hasattr(item, "label") else "",
            })

    print()
    return all_elements


def separate(elements: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Split raw elements into text_chunks, table_chunks, image_refs.

    Rules:
    - level=3 TextItems are table cell fragments — skip (already inside TableItem markdown)
    - PictureItem → image_refs (no text content yet, saved for Gemini later)
    - TableItem   → table_chunks, carries the last non-empty paragraph as intro context
    - Everything else → text_chunks
    """
    text_chunks:  list[dict] = []
    table_chunks: list[dict] = []
    image_refs:   list[dict] = []

    current_section = ""
    last_paragraph  = ""   # the most recent body text, used as table intro context

    for item in elements:
        page      = item.get("page")
        item_type = item["type"]
        text      = item.get("text", "").strip()
        level     = item.get("level", 1)
        is_front  = page in FRONT_MATTER_PAGES

        # ── Skip duplicate table cell fragments ──────────────────────────────
        if item_type == "TextItem" and level == 3:
            continue

        # ── Track current section for context metadata ────────────────────────
        if item_type == "SectionHeaderItem":
            current_section = text

        # ── Images → image_refs (description filled in later) ────────────────
        if item_type == "PictureItem":
            image_refs.append({
                "page":            page,
                "section":         current_section,
                "description":     "",        # to be filled by Gemini later
                "is_front_matter": is_front,
            })
            continue

        if not text:
            continue

        # ── Tables → table_chunks (with intro context) ────────────────────────
        if item_type == "TableItem":
            table_chunks.append({
                "page":            page,
                "section":         current_section,
                "intro_text":      last_paragraph,   # paragraph that introduces the table
                "table_markdown":  text,
                "is_front_matter": is_front,
            })
            continue

        # ── Everything else → text_chunks ─────────────────────────────────────
        chunk = {
            "page":            page,
            "type":            item_type,
            "level":           level,
            "section":         current_section,
            "text":            text,
            "is_front_matter": is_front,
        }
        text_chunks.append(chunk)

        # Keep track of last meaningful paragraph for table context
        if item_type in ("TextItem", "ListItem") and len(text) > 40:
            last_paragraph = text

    return text_chunks, table_chunks, image_refs


def main():
    if not PDF_PATH.exists():
        print(f"ERROR: PDF not found at {PDF_PATH}")
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Set up docling converter ──────────────────────────────────────────────
    print("Loading Docling...")
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.datamodel.base_models import InputFormat

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr             = False
    pipeline_options.do_table_structure = True   # needed to get proper TableItem output

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    # ── Choose pages ──────────────────────────────────────────────────────────
    if SAMPLE_ONLY:
        pages = sorted(SAMPLE_PAGES)
        print(f"Mode: SAMPLE — {len(pages)} pages\n")
    else:
        pages = list(range(1, 1000))
        print("Mode: FULL DOCUMENT — this will take ~4 hours\n")

    # ── Extract ───────────────────────────────────────────────────────────────
    elements = extract_elements(converter, PDF_PATH, pages)
    print(f"Total raw elements extracted: {len(elements)}")

    # ── Separate ──────────────────────────────────────────────────────────────
    text_chunks, table_chunks, image_refs = separate(elements)

    print(f"\nSplit results:")
    print(f"  text_chunks  : {len(text_chunks)}")
    print(f"  table_chunks : {len(table_chunks)}")
    print(f"  image_refs   : {len(image_refs)}")

    # ── Save ──────────────────────────────────────────────────────────────────
    out_text   = OUT_DIR / "text_chunks.json"
    out_tables = OUT_DIR / "table_chunks.json"
    out_images = OUT_DIR / "image_refs.json"

    with open(out_text,   "w", encoding="utf-8") as f:
        json.dump(text_chunks,  f, indent=2, ensure_ascii=False)
    with open(out_tables, "w", encoding="utf-8") as f:
        json.dump(table_chunks, f, indent=2, ensure_ascii=False)
    with open(out_images, "w", encoding="utf-8") as f:
        json.dump(image_refs,   f, indent=2, ensure_ascii=False)

    print(f"\nSaved to {OUT_DIR}/")
    print(f"  {out_text.name}   ({len(text_chunks)} chunks)")
    print(f"  {out_tables.name}  ({len(table_chunks)} chunks)")
    print(f"  {out_images.name}   ({len(image_refs)} refs — descriptions added later with Gemini)")
    print("\nNext: run build_index.py to embed and upload to Qdrant")


if __name__ == "__main__":
    main()
