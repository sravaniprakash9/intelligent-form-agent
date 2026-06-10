"""OCR engine registry and dispatch tests."""

import pytest
from PIL import Image

from src.extract.ocr_engine import OCREngine
from src.extract.ocr_engines import available_ocr_engines, normalize_ocr_engine


def test_normalize_ocr_engine():
    assert normalize_ocr_engine("Surya") == "surya"
    assert normalize_ocr_engine("rapidocr") == "rapidocr"


def test_unknown_engine_raises():
    with pytest.raises(ValueError, match="Unknown OCR engine"):
        normalize_ocr_engine("not-a-real-engine")


def test_available_engines_include_surya_and_tesseract():
    engines = available_ocr_engines()
    assert "surya" in engines
    assert "tesseract" in engines


def test_tesseract_runs_on_blank_image():
    img = Image.new("RGB", (200, 80), "white")
    result = OCREngine(engine="tesseract").run(img)
    assert result.method == "tesseract"
    assert isinstance(result.full_text, str)


@pytest.mark.skipif(
    "rapidocr" not in available_ocr_engines(),
    reason="rapidocr-onnxruntime not installed",
)
def test_rapidocr_runs_on_blank_image():
    img = Image.new("RGB", (200, 80), "white")
    result = OCREngine(engine="rapidocr").run(img)
    assert result.method == "rapidocr"
    assert isinstance(result.full_text, str)
