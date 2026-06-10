"""PyMuPDF-based text extraction for text-native PDFs."""

from __future__ import annotations

from pathlib import Path

import fitz

from src.extract.ocr_engine import OCRLine, OCRResult


def extract_pdf_text(path: Path, page_index: int = 0) -> OCRResult:
    """Extract embedded text from a PDF page using PyMuPDF."""
    doc = fitz.open(path)
    try:
        page = doc[page_index]
        text = page.get_text("text") or ""
    finally:
        doc.close()

    lines = [OCRLine(text=line.strip()) for line in text.splitlines() if line.strip()]
    return OCRResult(lines=lines, full_text=text)


def pdf_page_to_image(path: Path, page_index: int = 0, dpi: int = 200):
    """Render a PDF page to a PIL image for OCR on scanned documents."""
    from PIL import Image

    doc = fitz.open(path)
    try:
        page = doc[page_index]
        pix = page.get_pixmap(dpi=dpi)
        return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    finally:
        doc.close()


def is_text_sufficient(text: str, min_chars: int) -> bool:
    """Heuristic: enough embedded text to skip OCR."""
    cleaned = text.strip()
    if len(cleaned) < min_chars:
        return False
    markers = ("section", "patient", "provider", "prior authorization", "medicaid")
    lower = cleaned.lower()
    return any(marker in lower for marker in markers)
