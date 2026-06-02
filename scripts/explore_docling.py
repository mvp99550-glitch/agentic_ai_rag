"""
Exploration script — run BEFORE building the prefilter.

Purpose: See exactly what Docling gives us for different page types
so we can write accurate filters.

Runs on a small page sample (not all 997 pages) and saves output to
data/exports/docling_exploration.json for inspection.

Usage:
    uv run python scripts/explore_docling.py

Output:
    data/exports/docling_exploration.json  — full element dump
    data/exports/docling_exploration.txt   — human-readable summary
"""

import json
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PDF_PATH = Path("data/documents/CoreCourseFinancialAccounting.pdf")
OUT_JSON = Path("data/exports/docling_exploration.json")
OUT_TXT  = Path("data/exports/docling_exploration.txt")

# Set to True to process only SAMPLE_PAGES; False to process the full document
SAMPLE_ONLY = True

# Pages to sample — chosen to cover all content types we identified
# TOC=1-5, plain text=17, table=19, exercises=32, diagram=35
SAMPLE_PAGES = {1, 2, 3, 4, 5, 17, 18, 19, 32, 33, 34, 35, 36, 71, 89, 105}


def main():
    if not PDF_PATH.exists():
        print(f"ERROR: PDF not found at {PDF_PATH}")
        print("Move it with: move CoreCourseFinancialAccounting.pdf data/documents/")
        sys.exit(1)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    print("Loading Docling (first run downloads ~500MB models, be patient)...")
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.datamodel.base_models import InputFormat

    # Process one page at a time so only one page image is in RAM at once.
    # A wide page_range accumulates all rendered bitmaps and causes std::bad_alloc
    # on machines with < 2 GB free (layout model itself uses ~500 MB).
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False
    pipeline_options.do_table_structure = False

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    pages_to_process = sorted(SAMPLE_PAGES) if SAMPLE_ONLY else list(range(1, 10000))
    mode = f"SAMPLE ({len(pages_to_process)} pages)" if SAMPLE_ONLY else "FULL DOCUMENT"
    print(f"Mode: {mode} — processing one page at a time to stay within RAM\n")

    # ── Collect elements page by page ─────────────────────────────────────────
    elements: list[dict] = []
    lines_txt: list[str] = []

    for i, page_no in enumerate(pages_to_process, 1):
        print(f"  [{i}/{len(pages_to_process)}] page {page_no}...", end="\r")
        try:
            result = converter.convert(PDF_PATH, page_range=(page_no, page_no))
        except Exception as exc:
            print(f"\n  Page {page_no} failed: {exc}")
            continue

        doc = result.document

        for item, level in doc.iterate_items():
            item_type = type(item).__name__

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

            extra = {}
            if hasattr(item, "label"):
                extra["label"] = str(item.label)

            elements.append({
                "page": item_page,
                "level": level,
                "type": item_type,
                "text": text[:500],
                "text_len": len(text),
                **extra,
            })

            short = text[:120].replace("\n", " ") if text else "(no text)"
            lines_txt.append(
                f"[p{item_page}] [{item_type}] level={level}  |  {short}"
            )

    print()  # newline after progress line

    # ── Save outputs ──────────────────────────────────────────────────────────
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(elements, f, indent=2, ensure_ascii=False)

    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines_txt))

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"Total elements found: {len(elements)}\n")

    type_counts: dict[str, int] = {}
    for e in elements:
        type_counts[e["type"]] = type_counts.get(e["type"], 0) + 1
    print("Element types:")
    for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  {t:<30} {c}")

    short_texts = [e for e in elements if 0 < e["text_len"] < 30]
    print(f"\nShort text elements (< 30 chars) — likely headers/sidebars: {len(short_texts)}")
    for e in short_texts[:30]:
        print(f"  p{e['page']} | {e['type']:<25} | '{e['text']}'")

    print(f"\nOutputs saved:")
    print(f"  {OUT_JSON}  <- full JSON dump")
    print(f"  {OUT_TXT}   <- human-readable lines")
    print("\nNext: review these files to tune prefilter.py")


if __name__ == "__main__":
    main()
