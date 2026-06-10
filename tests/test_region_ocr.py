"""Tests for hybrid crop OCR reader."""

from unittest.mock import MagicMock

from PIL import Image

from src.extract.ocr_engine import OCRLine, OCRResult
from src.extract.region_ocr import RegionOCRReader


def _engine(result: OCRResult) -> MagicMock:
    engine = MagicMock()
    engine.run_crop.return_value = result
    engine.method = result.method
    return engine


def test_fast_crop_used_when_confidence_high():
    image = Image.new("RGB", (200, 80))
    fast = _engine(OCRResult(
        lines=[OCRLine(text="62106", confidence=0.92)],
        full_text="62106",
        method="rapidocr",
    ))
    accurate = _engine(OCRResult(lines=[], full_text="", method="surya"))

    reader = RegionOCRReader(fast, accurate, min_confidence=0.65)
    result = reader.read_digits(image, (10, 10, 100, 50), min_len=4, max_len=12)

    assert result == "62106"
    accurate.run_crop.assert_not_called()


def test_surya_fallback_when_fast_confidence_low():
    image = Image.new("RGB", (200, 80))
    fast = _engine(OCRResult(
        lines=[OCRLine(text="62106", confidence=0.2)],
        full_text="62106",
        method="rapidocr",
    ))
    accurate = _engine(OCRResult(
        lines=[OCRLine(text="62106", confidence=0.95)],
        full_text="62106",
        method="surya",
    ))

    reader = RegionOCRReader(fast, accurate, min_confidence=0.65)
    result = reader.read_digits(image, (10, 10, 100, 50), min_len=4, max_len=12)

    assert result == "62106"
    accurate.run_crop.assert_called_once()
