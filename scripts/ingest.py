"""
Ingest pipeline — extracts and separates PDF content by type.

Reads:  data/documents/CoreCourseFinancialAccounting.pdf
Writes:
    data/processed/text_chunks.json      — paragraphs, lists, headers
    data/processed/table_chunks.json     — tables + 8-line context before
    data/processed/image_refs.json       — image metadata + path to saved .png
    data/processed/images/               — actual image .png files

Each item carries:
    is_exercise: True  — content from exercise/Q&A sections (filtered at query time)
    is_exercise: False — theory/concept content (used for retrieval)

Pipeline order (do NOT skip steps):
    1. ingest.py          → extracts text, tables, images
    2. describe_images.py → converts images to text using Llama 4 Scout
    3. build_index.py     → chunks + embeds + uploads to Qdrant

Toggle SAMPLE_ONLY:
    True  → first FIRST_N_PAGES pages for testing
    False → full document (~4 hrs) for production
"""

import hashlib
import json
import sys
from collections import deque
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PDF_PATH   = Path("data/documents/CoreCourseFinancialAccounting.pdf")
OUT_DIR    = Path("data/processed")
IMAGES_DIR = OUT_DIR / "images"

SAMPLE_ONLY = True
START_PAGE  = 801    # first page of this batch (inclusive)
END_PAGE    = 1050   # last page of this batch (inclusive, covers end of book)

FRONT_MATTER_PAGES  = {1, 2, 3, 4, 5}
TABLE_CONTEXT_LINES = 8
MIN_IMAGE_SIZE      = 200   # px — skip logos/icons

# Section names that indicate exercise/Q&A content — tagged is_exercise=True
# so they can be routed to a separate Qdrant collection and excluded from
# theory retrieval queries.
EXERCISE_SECTION_NAMES = frozenset({
    "questions",
    "exercises",
    "exercise",
    "answers",
    "solutions",
    "problems",
    "further reading",
    "bibliography",
    "summary questions",
    "problems and solutions",
    "practice problems",
    "review questions",
    "discussion questions",
})


# ── Helpers ───────────────────────────────────────────────────────────────────

def compute_file_hash(filepath: Path) -> str:
    with open(filepath, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def verify_outputs(out_text, out_tables, out_images, text_chunks, table_chunks, image_refs):
    for path, data, name in [
        (out_text,   text_chunks,  "text_chunks"),
        (out_tables, table_chunks, "table_chunks"),
        (out_images, image_refs,   "image_refs"),
    ]:
        assert path.exists(),           f"[FAIL] Missing: {path}"
        assert path.stat().st_size > 0, f"[FAIL] Empty file: {path}"
        with open(path, encoding="utf-8") as f:
            disk = json.load(f)
        assert len(disk) == len(data), (
            f"[FAIL] {name} count mismatch — disk:{len(disk)} memory:{len(data)}"
        )
    print("[OK] Data integrity verified.")


# ── Extraction ────────────────────────────────────────────────────────────────

def extract_elements(converter, pdf_path: Path, pages: list[int]) -> list[dict]:
    all_elements = []
    total = len(pages)

    for i, page_no in enumerate(pages, 1):
        print(f"  [{i}/{total}] page {page_no}...", end="\r")
        try:
            result = converter.convert(pdf_path, page_range=(page_no, page_no))
        except Exception as exc:
            print(f"\n  Page {page_no} failed: {exc}")
            continue

        doc     = result.document
        pic_idx = 0

        for item, level in doc.iterate_items():
            item_type = type(item).__name__
            item_page = None
            if hasattr(item, "prov") and item.prov:
                item_page = item.prov[0].page_no if item.prov else None

            if item_type == "PictureItem":
                image_path = ""
                try:
                    image = item.get_image(doc)
                    if image and image.width > MIN_IMAGE_SIZE and image.height > MIN_IMAGE_SIZE:
                        filename   = f"page{item_page}_img{pic_idx}.png"
                        saved_path = IMAGES_DIR / filename
                        image.save(saved_path)
                        image_path = str(saved_path)
                        print(f"\n  Saved image: {filename} ({image.width}x{image.height}px)")
                except Exception as exc:
                    print(f"\n  Image save failed p{item_page}: {exc}")
                pic_idx += 1

                all_elements.append({
                    "page":       item_page,
                    "level":      level,
                    "type":       item_type,
                    "text":       "",
                    "label":      "",
                    "image_path": image_path,
                })
                continue

            text = ""
            if hasattr(item, "text"):
                text = item.text or ""
            elif hasattr(item, "export_to_markdown"):
                try:
                    text = item.export_to_markdown(doc)
                except TypeError:
                    text = item.export_to_markdown()

            all_elements.append({
                "page":       item_page,
                "level":      level,
                "type":       item_type,
                "text":       text,
                "label":      str(item.label) if hasattr(item, "label") else "",
                "image_path": "",
            })

    print()
    return all_elements


# ── Separation ────────────────────────────────────────────────────────────────

def separate(elements: list[dict], file_hash: str):
    text_chunks:  list[dict] = []
    table_chunks: list[dict] = []
    image_refs:   list[dict] = []

    current_section    = ""
    is_exercise_section = False
    context_buffer: deque[str] = deque(maxlen=TABLE_CONTEXT_LINES)

    for item in elements:
        page      = item.get("page")
        item_type = item["type"]
        text      = item.get("text", "").strip()
        level     = item.get("level", 1)
        is_front  = page in FRONT_MATTER_PAGES

        # Skip table cell fragments
        if item_type == "TextItem" and level == 3:
            continue

        # Track section header + detect exercise sections
        if item_type == "SectionHeaderItem":
            current_section     = text
            is_exercise_section = current_section.lower().strip() in EXERCISE_SECTION_NAMES

        # Images → image_refs
        if item_type == "PictureItem":
            image_refs.append({
                "page":            page,
                "section":         current_section,
                "image_path":      item.get("image_path", ""),
                "description":     "",
                "content_type":    "image_description",
                "is_front_matter": is_front,
                "is_exercise":     is_exercise_section,
                "file_hash":       file_hash,
            })
            continue

        if not text:
            continue

        # Tables → table_chunks with context
        if item_type == "TableItem":
            context_before = "\n".join(context_buffer)
            table_chunks.append({
                "page":            page,
                "section":         current_section,
                "context_before":  context_before,
                "intro_text":      context_before,
                "table_markdown":  text,
                "content_type":    "table",
                "is_front_matter": is_front,
                "is_exercise":     is_exercise_section,
                "file_hash":       file_hash,
            })
            continue

        # All other items → text_chunks
        text_chunks.append({
            "page":            page,
            "type":            item_type,
            "level":           level,
            "section":         current_section,
            "text":            text,
            "content_type":    "text",
            "is_front_matter": is_front,
            "is_exercise":     is_exercise_section,
            "file_hash":       file_hash,
        })

        if item_type in ("TextItem", "ListItem", "SectionHeaderItem") and len(text) > 20:
            context_buffer.append(text)

    return text_chunks, table_chunks, image_refs


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if not PDF_PATH.exists():
        print(f"ERROR: PDF not found at {PDF_PATH}")
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    print("Computing PDF file hash...")
    file_hash = compute_file_hash(PDF_PATH)
    print(f"  SHA256: {file_hash[:16]}...")

    print("\nLoading Docling...")
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.datamodel.base_models import InputFormat

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr                  = False
    pipeline_options.do_table_structure      = True
    pipeline_options.generate_picture_images = True
    pipeline_options.images_scale            = 0.5   # 1.0 caused std::bad_alloc cascade from p180

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    if SAMPLE_ONLY:
        pages = list(range(START_PAGE, END_PAGE + 1))
        print(f"Mode: BATCH — pages {START_PAGE}–{END_PAGE}\n")
    else:
        pages = list(range(1, 10000))
        print("Mode: FULL DOCUMENT — this will take ~4 hours\n")

    elements = extract_elements(converter, PDF_PATH, pages)
    print(f"Total raw elements: {len(elements)}")

    text_chunks, table_chunks, image_refs = separate(elements, file_hash)

    # Audit exercise vs theory split
    exercise_text   = sum(1 for c in text_chunks  if c["is_exercise"])
    exercise_tables = sum(1 for c in table_chunks if c["is_exercise"])
    print(f"\nSplit results:")
    print(f"  text_chunks  : {len(text_chunks)}  (exercise: {exercise_text})")
    print(f"  table_chunks : {len(table_chunks)}  (exercise: {exercise_tables})")
    print(f"  image_refs   : {len(image_refs)}")

    saved_images = [r for r in image_refs if r["image_path"]]
    print(f"  images saved : {len(saved_images)} (in {IMAGES_DIR})")

    out_text   = OUT_DIR / "text_chunks.json"
    out_tables = OUT_DIR / "table_chunks.json"
    out_images = OUT_DIR / "image_refs.json"

    # Append to existing files so batches accumulate without overwriting prior pages.
    # Deduplication in build_index.py (stable IDs + upsert) handles any overlaps.
    for path, new_data in [
        (out_text,   text_chunks),
        (out_tables, table_chunks),
        (out_images, image_refs),
    ]:
        existing: list = []
        if path.exists() and path.stat().st_size > 0:
            with open(path, encoding="utf-8") as f:
                existing = json.load(f)
        merged = existing + new_data
        with open(path, "w", encoding="utf-8") as f:
            json.dump(merged, f, indent=2, ensure_ascii=False)
        print(f"  {path.name}: {len(existing)} existing + {len(new_data)} new = {len(merged)} total")

    print("\nVerifying data integrity...")
    # Pass merged totals so verify_outputs compares disk count against cumulative total
    with open(out_text,   encoding="utf-8") as f: merged_text   = json.load(f)
    with open(out_tables, encoding="utf-8") as f: merged_tables = json.load(f)
    with open(out_images, encoding="utf-8") as f: merged_images = json.load(f)
    verify_outputs(out_text, out_tables, out_images, merged_text, merged_tables, merged_images)

    print(f"\nSaved to {OUT_DIR}/")
    print(f"  text_chunks.json   ({len(text_chunks)} chunks)")
    print(f"  table_chunks.json  ({len(table_chunks)} chunks, {TABLE_CONTEXT_LINES}-line context)")
    print(f"  image_refs.json    ({len(image_refs)} refs, {len(saved_images)} with .png files)")
    print(f"\nNext: run describe_images.py, then build_index.py")


if __name__ == "__main__":
    main()
