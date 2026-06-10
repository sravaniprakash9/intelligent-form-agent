"""Crop-level OCR: fast engine first, Surya fallback for low-confidence reads."""

from __future__ import annotations

import re
from dataclasses import dataclass

from PIL import Image

from src.extract.ocr_engine import OCREngine, OCRResult


@dataclass
class CropReadResult:
    text: str
    confidence: float
    method: str


def _best_confidence(result: OCRResult) -> float:
    if not result.lines:
        return 0.0
    return max(line.confidence for line in result.lines)


def _merge_text(result: OCRResult) -> str:
    return " ".join(line.text.strip() for line in result.lines if line.text.strip())


class RegionOCRReader:
    """Read a field crop with a fast OCR engine, then Surya if needed."""

    def __init__(
        self,
        fast_engine: OCREngine,
        accurate_engine: OCREngine | None = None,
        *,
        min_confidence: float = 0.65,
        enable_accurate_fallback: bool = True,
    ) -> None:
        self.fast_engine = fast_engine
        self.accurate_engine = accurate_engine
        self.min_confidence = min_confidence
        self.enable_accurate_fallback = enable_accurate_fallback

    def read_crop(
        self,
        image: Image.Image,
        box: tuple[int, int, int, int],
        *,
        accept: re.Pattern[str] | None = None,
    ) -> CropReadResult | None:
        fast = self.fast_engine.run_crop(image, box)
        text = _merge_text(fast)
        conf = _best_confidence(fast)
        if accept and text:
            match = accept.search(text)
            text = match.group(0) if match else ""
        if text and conf >= self.min_confidence:
            return CropReadResult(text=text, confidence=conf, method=fast.method)

        if self.enable_accurate_fallback and self.accurate_engine is not None:
            accurate = self.accurate_engine.run_crop(image, box)
            acc_text = _merge_text(accurate)
            acc_conf = _best_confidence(accurate)
            if accept and acc_text:
                match = accept.search(acc_text)
                acc_text = match.group(0) if match else ""
            if acc_text and (not text or acc_conf >= conf):
                return CropReadResult(
                    text=acc_text,
                    confidence=acc_conf,
                    method=f"{accurate.method}-crop",
                )

        if text:
            return CropReadResult(text=text, confidence=conf, method=fast.method)
        return None

    def read_digits(
        self,
        image: Image.Image,
        box: tuple[int, int, int, int],
        *,
        min_len: int = 1,
        max_len: int = 12,
    ) -> str | None:
        pattern = re.compile(rf"\d{{{min_len},{max_len}}}")
        result = self.read_crop(image, box, accept=pattern)
        return result.text if result else None

    def read_text(self, image: Image.Image, box: tuple[int, int, int, int]) -> str | None:
        result = self.read_crop(image, box)
        return result.text.strip() if result and result.text.strip() else None
