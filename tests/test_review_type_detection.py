"""Review type: urgent vs non-urgent across OCR layouts."""

from pathlib import Path
from unittest.mock import patch

from PIL import Image

from src.extract.checkbox_detector import (
    _resolve_review_type,
    _review_type_from_label_order,
    detect_texas_form_fields,
)
from src.extract.document_extractor import DocumentExtractor
from src.extract.form_parser import FormParser
from src.extract.ocr_engine import OCRLine
from src.extract.sanitize import normalize_ocr_text
from src.ingest.loader import load_image

SECTION_II_NON_URGENT = """
Non-Urgent
Clinical Reason for Urgency:
Urgent
Review Type:
Initial Request
"""

SECTION_II_URGENT = """
Urgent
Review Type:
Non-Urgent
Initial Request
"""

DANIEL_FORM = "0a01f77b-85f9-48c9-89bf-4b095bebb438_TX_page_1"
BOBBY_FORM = "0a9d92f7-b19a-40b5-8c48-1546b7e453ab_TX_page_2"


def test_label_order_non_urgent():
    assert _review_type_from_label_order(SECTION_II_NON_URGENT) == "non_urgent"


def test_label_order_urgent():
    assert _review_type_from_label_order(SECTION_II_URGENT) == "urgent"


def test_text_fallback_non_urgent_when_no_bboxes():
    full_text = (
        "SECTION II - GENERAL INFORMATION\n"
        f"{SECTION_II_NON_URGENT.strip()}\n"
        "SECTION III - PATIENT INFORMATION\n"
    )
    image = Image.new("RGB", (1200, 1600), color="white")
    lines = [OCRLine(text=ln) for ln in full_text.strip().splitlines()]
    fields = detect_texas_form_fields(image, lines, full_text=full_text)
    assert fields["review_type"] == "non_urgent"


def test_opencv_urgent_not_overridden_by_layout_text():
    image = Image.new("RGB", (1200, 1600), color="white")
    lines = [OCRLine(text="Review Type", bbox=(100, 200, 200, 220))]
    with patch(
        "src.extract.checkbox_detector._pick_exclusive",
        return_value=r"Urgent",
    ):
        assert (
            _resolve_review_type(
                image, lines, __import__("numpy").zeros((100, 100)), SECTION_II_URGENT
            )
            == "urgent"
        )


def _parse_review_type(filename: str) -> str | None:
    path = Path(f"data/raw/{filename}.png")
    if not path.exists():
        return None
    extractor = DocumentExtractor(ocr_mode="hybrid", fast_engine="rapidocr")
    extracted = extractor.extract(path)
    image = load_image(path)
    text = normalize_ocr_text(extracted.full_text)
    lines = [
        OCRLine(text=normalize_ocr_text(ln.text), bbox=ln.bbox, confidence=ln.confidence)
        for ln in extracted.lines
    ]
    doc = FormParser(extractor=extractor).parse_content(
        filename, "test.png", text, lines, image, extracted.method
    )
    return doc.section_ii_general.review_type


def test_daniel_jarvis_non_urgent():
    assert _parse_review_type(DANIEL_FORM) == "non_urgent"


def test_bobby_juarez_urgent():
    assert _parse_review_type(BOBBY_FORM) == "urgent"
