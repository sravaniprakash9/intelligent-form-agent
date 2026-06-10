"""Document extraction orchestrator.

Priority:
1. PyMuPDF embedded text (PDFs with selectable text)
2. Hybrid: fast OCR full page + Surya crop fallback (via FormParser field refiner)
3. Full-page OCR (Surya / RapidOCR / PaddleOCR / EasyOCR / Tesseract)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from src.config.settings import settings
from src.extract.ocr_engine import OCREngine, OCRLine, OCRResult
from src.extract.ocr_engines import available_fast_ocr_engines, normalize_ocr_engine
from src.extract.pdf_extractor import extract_pdf_text, is_text_sufficient, pdf_page_to_image
from src.extract.region_ocr import RegionOCRReader
from src.ingest.loader import load_image
from src.ingest.preprocess import preprocess_for_engine

logger = logging.getLogger(__name__)


@dataclass
class ExtractedDocument:
    full_text: str
    lines: list[OCRLine]
    method: str
    image: Image.Image | None = None


class DocumentExtractor:
    def __init__(
        self,
        ocr: OCREngine | None = None,
        *,
        fast_preprocess: bool = False,
        ocr_engine: str = "surya",
        ocr_mode: str = "full",
        fast_engine: str | None = None,
    ) -> None:
        self.ocr_mode = ocr_mode.strip().lower()
        self.fast_preprocess = fast_preprocess

        if self.ocr_mode == "hybrid":
            fast = fast_engine or settings.hybrid_fast_engine
            installed = available_fast_ocr_engines()
            if fast not in installed:
                fast = installed[0] if installed else "tesseract"
            self.fast_engine_name = normalize_ocr_engine(fast)
            self.ocr = ocr or OCREngine(engine=self.fast_engine_name)
            self._accurate_ocr: OCREngine | None = None
            self._crop_reader: RegionOCRReader | None = None
        else:
            self.fast_engine_name = None
            self.ocr = ocr or OCREngine(engine=ocr_engine)
            self._accurate_ocr = None
            self._crop_reader = None

    @property
    def is_hybrid(self) -> bool:
        return self.ocr_mode == "hybrid"

    def _accurate_engine(self) -> OCREngine:
        if self._accurate_ocr is None:
            self._accurate_ocr = OCREngine(engine="surya")
        return self._accurate_ocr

    def crop_reader(self) -> RegionOCRReader:
        if not self.is_hybrid:
            raise RuntimeError("crop_reader is only available in hybrid OCR mode")
        if self._crop_reader is None:
            self._crop_reader = RegionOCRReader(
                fast_engine=self.ocr,
                accurate_engine=self._accurate_engine(),
                min_confidence=settings.hybrid_crop_confidence_threshold,
                enable_accurate_fallback=settings.hybrid_enable_surya_fallback,
            )
        return self._crop_reader

    def warmup(self) -> None:
        """Load OCR models for the active mode."""
        self.ocr.warmup()
        if self.is_hybrid and settings.hybrid_warmup_surya:
            try:
                self._accurate_engine().warmup()
            except Exception as exc:
                logger.warning("Surya warmup skipped (crop fallback may be slower): %s", exc)

    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        engine = getattr(self.ocr, "engine", "surya")
        use_fast = self.fast_preprocess and engine == "surya"
        return preprocess_for_engine(image, engine, fast=use_fast)

    def extract(self, path: Path) -> ExtractedDocument:
        suffix = path.suffix.lower()

        if suffix == ".pdf":
            return self._extract_pdf(path)
        return self._extract_image(path)

    def _extract_pdf(self, path: Path) -> ExtractedDocument:
        pdf_result = extract_pdf_text(path)
        if is_text_sufficient(pdf_result.full_text, settings.pdf_min_text_chars):
            logger.info("Using PyMuPDF text extraction for %s", path.name)
            image = pdf_page_to_image(path)
            return ExtractedDocument(
                full_text=pdf_result.full_text,
                lines=pdf_result.lines,
                method="pymupdf",
                image=image,
            )

        logger.info("PDF appears scanned; using OCR for %s", path.name)
        image = self._preprocess_image(pdf_page_to_image(path))
        ocr_result = self._run_ocr_with_fallback(image)
        return ExtractedDocument(
            full_text=ocr_result.full_text,
            lines=ocr_result.lines,
            method=ocr_result.method,
            image=image,
        )

    def _extract_image(self, path: Path) -> ExtractedDocument:
        image = self._preprocess_image(load_image(path))
        ocr_result = self._run_ocr_with_fallback(image)
        return ExtractedDocument(
            full_text=ocr_result.full_text,
            lines=ocr_result.lines,
            method=ocr_result.method,
            image=image,
        )

    def _run_ocr_with_fallback(self, image: Image.Image) -> OCRResult:
        if self.is_hybrid:
            result = self.ocr.run(image)
            result.method = f"hybrid:{result.method}"
            return result

        engine = getattr(self.ocr, "engine", "surya")
        if engine != "surya":
            return self.ocr.run(image)
        try:
            return self.ocr.run(image)
        except Exception as exc:
            if not settings.enable_tesseract_fallback:
                raise RuntimeError(
                    f"Surya OCR failed and Tesseract fallback is disabled: {exc}"
                ) from exc
            logger.warning("Surya OCR failed (%s); using optional Tesseract fallback", exc)
            return self.ocr.run_tesseract(image)
