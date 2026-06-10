"""OCR engine registry and UI metadata."""

from __future__ import annotations

from typing import Literal

OcrEngineName = Literal["surya", "rapidocr", "tesseract", "paddleocr", "easyocr"]
OcrMode = Literal["full", "hybrid"]

OCR_ENGINES: tuple[OcrEngineName, ...] = (
    "surya",
    "rapidocr",
    "tesseract",
    "paddleocr",
    "easyocr",
)
FAST_OCR_ENGINES: tuple[OcrEngineName, ...] = ("rapidocr", "paddleocr", "easyocr", "tesseract")
OCR_MODES: tuple[OcrMode, ...] = ("hybrid", "full")

OCR_MODE_INFO: dict[str, dict[str, str]] = {
    "hybrid": {
        "label": "Hybrid (recommended)",
        "speed": "Fast layout (~20–90 sec/form) + Surya only on failed fields",
        "accuracy": "Best speed/accuracy tradeoff for Texas PA forms",
        "hint": "RapidOCR/EasyOCR/PaddleOCR on crops; Surya fallback for low-confidence fields.",
    },
    "full": {
        "label": "Full page",
        "speed": "Depends on engine (Surya ~3–10 min/form)",
        "accuracy": "Single engine across entire document",
        "hint": "Use when you need one consistent OCR pass.",
    },
}

OCR_ENGINE_INFO: dict[str, dict[str, str]] = {
    "surya": {
        "label": "Surya",
        "speed": "Slow (~3–10 min/form)",
        "accuracy": "Highest — best for checkboxes, handwriting, dense forms",
        "hint": "Accurate mode / crop fallback only in hybrid.",
    },
    "rapidocr": {
        "label": "RapidOCR",
        "speed": "Fast (~20–60 sec/form)",
        "accuracy": "Good printed text; hybrid default fast engine",
        "hint": "ONNX Paddle-based engine. Best default for hybrid mode.",
    },
    "paddleocr": {
        "label": "PaddleOCR",
        "speed": "Fast (~30–90 sec/form)",
        "accuracy": "Strong on printed fields; optional pip install",
        "hint": "pip install paddlepaddle paddleocr",
    },
    "easyocr": {
        "label": "EasyOCR",
        "speed": "Medium (~45–120 sec/form)",
        "accuracy": "Good handwriting; heavier model download",
        "hint": "pip install easyocr",
    },
    "tesseract": {
        "label": "Tesseract",
        "speed": "~45–120 sec/form",
        "accuracy": "Weakest on dense layouts; local fallback",
        "hint": "Requires brew install tesseract.",
    },
}


def normalize_ocr_engine(name: str) -> OcrEngineName:
    key = name.strip().lower()
    if key not in OCR_ENGINES:
        raise ValueError(f"Unknown OCR engine '{name}'. Choose: {', '.join(OCR_ENGINES)}")
    return key  # type: ignore[return-value]


def ocr_engine_label(name: str) -> str:
    return OCR_ENGINE_INFO.get(name, {}).get("label", name)


def ocr_mode_label(name: str) -> str:
    return OCR_MODE_INFO.get(name, {}).get("label", name)


def _engine_installed(name: str) -> bool:
    if name == "rapidocr":
        try:
            import rapidocr_onnxruntime  # noqa: F401

            return True
        except ImportError:
            return False
    if name == "paddleocr":
        try:
            import paddleocr  # noqa: F401

            return True
        except ImportError:
            return False
    if name == "easyocr":
        try:
            import easyocr  # noqa: F401

            return True
        except ImportError:
            return False
    return True


def available_fast_ocr_engines() -> list[str]:
    """Fast engines usable for hybrid layout + crop OCR."""
    return [e for e in FAST_OCR_ENGINES if _engine_installed(e)]


def available_ocr_engines() -> list[str]:
    """Engines selectable for full-page mode."""
    return [e for e in OCR_ENGINES if _engine_installed(e)]
