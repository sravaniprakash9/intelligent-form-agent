"""Load form images and PDFs."""

from pathlib import Path

import fitz
from PIL import Image


SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".pdf"}


def list_form_files(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        return []
    return sorted(
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def load_image(path: Path) -> Image.Image:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return pdf_to_image(path)
    return Image.open(path).convert("RGB")


def pdf_to_image(path: Path, page_index: int = 0, dpi: int = 200) -> Image.Image:
    doc = fitz.open(path)
    try:
        page = doc[page_index]
        pix = page.get_pixmap(dpi=dpi)
        return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    finally:
        doc.close()
