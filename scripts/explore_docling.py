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
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PDF_PATH = Path("data/documents/CoreCourseFinancialAccounting.pdf")
OUT_JSON = Path("data/exports/docling_exploration.json")
OUT_TXT = Path("data/exports/docling_exploration.txt")

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
    from docling.pipeline.simple_pipeline import SimplePipeline

    # Use SimplePipeline to avoid rendering page images for the AI layout model.
    # StandardPipeline (default) renders every page to a bitmap before running
    # docling_layout_heron, which causes std::bad_alloc with < 2 GB free RAM.
    # SimplePipeline uses only heuristic/rule-based extraction — no image rendering,
    # no transformer models loaded. Trade-off: heading/table detection is less precise.
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False
    pipeline_options.do_table_structure = False

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_cls=SimplePipeline,
                pipeline_options=pipeline_options,
            )
        }
    )

    if SAMPLE_ONLY:
        page_range = (min(SAMPLE_PAGES), max(SAMPLE_PAGES))
        print(f"Mode: SAMPLE — pages {sorted(SAMPLE_PAGES)}, range {page_range}")
    else:
        page_range = (1, 9999)
        print("Mode: FULL DOCUMENT")
    print("This takes 2-5 minutes for the sample...\n")

    result = converter.convert(PDF_PATH, page_range=page_range)
    doc = result.document

    # ── Collect all elements ──────────────────────────────────────────────────
    elements = []
    lines_txt = []

    for item, level in doc.iterate_items():
        item_type = type(item).__name__

        # Get page number safely
        page_no = None
        if hasattr(item, "prov") and item.prov:
            page_no = item.prov[0].page_no if item.prov else None

        # In sample mode, skip pages outside our target set
        if SAMPLE_ONLY and page_no is not None and page_no not in SAMPLE_PAGES:
            continue

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

        record = {
            "page": page_no,
            "level": level,
            "type": item_type,
            "text": text[:500],          # cap at 500 chars for readability
            "text_len": len(text),
            **extra,
        }
        elements.append(record)

        # Human-readable line
        short = text[:120].replace("\n", " ") if text else "(no text)"
        lines_txt.append(
            f"[p{page_no}] [{item_type}] level={level}  |  {short}"
        )

    # ── Save outputs ─────────────────────────────────────────────────────────
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(elements, f, indent=2, ensure_ascii=False)

    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines_txt))

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"Total elements found across sample pages: {len(elements)}\n")

    # Count by type
    type_counts: dict[str, int] = {}
    for e in elements:
        type_counts[e["type"]] = type_counts.get(e["type"], 0) + 1
    print("Element types:")
    for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  {t:<30} {c}")

    # Show very short text elements (likely sidebars/headers)
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