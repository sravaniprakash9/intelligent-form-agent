"""OCR engine — Surya, RapidOCR, PaddleOCR, EasyOCR, or Tesseract."""

from __future__ import annotations

from dataclasses import dataclass

from PIL import Image

from src.extract.hf_utils import can_use_surya
from src.extract.ocr_engines import normalize_ocr_engine

# Balance speed vs accuracy — higher than 2000 helps RapidOCR on dense forms.
_RAPID_OCR_MAX_DIM = 2800
_TESSERACT_MAX_DIM = 2400


def _resize_for_fast_ocr(image: Image.Image, max_dim: int = _RAPID_OCR_MAX_DIM) -> tuple[Image.Image, float]:
    """Downscale large scans; returns (image, scale) where scale = resized/original."""
    w, h = image.size
    longest = max(w, h)
    if longest <= max_dim:
        return image, 1.0
    scale = max_dim / longest
    new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
    return image.resize(new_size, Image.Resampling.LANCZOS), scale


def _scale_bbox(
    bbox: tuple[float, float, float, float] | None, scale: float
) -> tuple[float, float, float, float] | None:
    if bbox is None or scale == 1.0:
        return bbox
    inv = 1.0 / scale
    x0, y0, x1, y1 = bbox
    return (x0 * inv, y0 * inv, x1 * inv, y1 * inv)


@dataclass
class OCRLine:
    text: str
    bbox: tuple[float, float, float, float] | None = None
    confidence: float = 1.0


@dataclass
class OCRResult:
    lines: list[OCRLine]
    full_text: str
    method: str = "surya"


class OCREngine:
    """Runs document OCR using the selected engine."""

    def __init__(self, languages: list[str] | None = None, engine: str = "surya") -> None:
        self.languages = languages or ["en"]
        self.engine = normalize_ocr_engine(engine)
        self._surya_models = None
        self._surya_load_failed = False
        self._rapidocr = None
        self._paddleocr = None
        self._easyocr = None

    def warmup(self) -> None:
        """Eagerly load models for the selected engine (call once before batch OCR)."""
        if self.engine == "surya":
            self._load_surya()
        elif self.engine == "rapidocr":
            self._load_rapidocr()
        elif self.engine == "paddleocr":
            self._load_paddleocr()
        elif self.engine == "easyocr":
            self._load_easyocr()

    def run(self, image: Image.Image) -> OCRResult:
        if self.engine == "surya":
            return self._run_surya(image)
        if self.engine == "rapidocr":
            return self.run_rapidocr(image)
        if self.engine == "paddleocr":
            return self.run_paddleocr(image)
        if self.engine == "easyocr":
            return self.run_easyocr(image)
        return self.run_tesseract(image)

    def run_crop(self, image: Image.Image, box: tuple[int, int, int, int]) -> OCRResult:
        """OCR a single field region (used by hybrid crop readers)."""
        x0, y0, x1, y1 = (int(v) for v in box)
        if x1 <= x0 or y1 <= y0:
            return OCRResult(lines=[], full_text="", method=self.engine)
        crop = image.crop((x0, y0, x1, y1))
        if crop.width < 8 or crop.height < 8:
            return OCRResult(lines=[], full_text="", method=self.engine)
        return self.run(crop)

    def _load_surya(self) -> None:
        if self._surya_models is not None:
            return
        if self._surya_load_failed:
            raise RuntimeError("Surya models previously failed to load")

        if not can_use_surya():
            self._surya_load_failed = True
            raise RuntimeError(
                "Surya models not cached and huggingface.co is unreachable. "
                "Fix DNS/network and run: python -m src.cli download-models"
            )

        try:
            from surya.model.detection.segformer import load_model as load_det_model
            from surya.model.detection.segformer import load_processor as load_det_processor
            from surya.model.recognition.model import load_model as load_rec_model
            from surya.model.recognition.processor import load_processor as load_rec_processor

            self._surya_models = (
                load_det_model(),
                load_det_processor(),
                load_rec_model(),
                load_rec_processor(),
            )
        except Exception:
            self._surya_load_failed = True
            raise

    def _load_rapidocr(self) -> None:
        if self._rapidocr is not None:
            return
        try:
            from rapidocr_onnxruntime import RapidOCR

            self._rapidocr = RapidOCR()
        except ImportError as exc:
            raise RuntimeError(
                "RapidOCR not installed. Run: pip install rapidocr-onnxruntime"
            ) from exc

    def _load_paddleocr(self) -> None:
        if self._paddleocr is not None:
            return
        try:
            from paddleocr import PaddleOCR

            self._paddleocr = PaddleOCR(use_angle_cls=False, lang="en", show_log=False)
        except ImportError as exc:
            raise RuntimeError(
                "PaddleOCR not installed. Run: pip install paddlepaddle paddleocr"
            ) from exc

    def _load_easyocr(self) -> None:
        if self._easyocr is not None:
            return
        try:
            import easyocr

            self._easyocr = easyocr.Reader(["en"], gpu=False, verbose=False)
        except ImportError as exc:
            raise RuntimeError("EasyOCR not installed. Run: pip install easyocr") from exc

    def _run_surya(self, image: Image.Image) -> OCRResult:
        from surya.ocr import run_ocr

        self._load_surya()
        det_model, det_processor, rec_model, rec_processor = self._surya_models

        predictions = run_ocr(
            [image.convert("RGB")],
            [self.languages],
            det_model,
            det_processor,
            rec_model,
            rec_processor,
        )

        lines: list[OCRLine] = []
        for page in predictions:
            for line in page.text_lines:
                bbox = tuple(line.bbox) if line.bbox else None
                confidence = float(line.confidence or 1.0)
                lines.append(OCRLine(text=line.text, bbox=bbox, confidence=confidence))

        full_text = "\n".join(line.text for line in lines)
        return OCRResult(lines=lines, full_text=full_text, method="surya")

    def run_rapidocr(self, image: Image.Image) -> OCRResult:
        """Fast ONNX OCR — good printed-text speed/accuracy tradeoff."""
        import numpy as np

        self._load_rapidocr()
        ocr_image, scale = _resize_for_fast_ocr(image, max_dim=_RAPID_OCR_MAX_DIM)
        arr = np.array(ocr_image.convert("RGB"))
        result, _ = self._rapidocr(arr)

        lines: list[OCRLine] = []
        if result:
            for box, text, score in result:
                text = (text or "").strip()
                if not text:
                    continue
                xs = [float(p[0]) for p in box]
                ys = [float(p[1]) for p in box]
                bbox = _scale_bbox((min(xs), min(ys), max(xs), max(ys)), scale)
                lines.append(
                    OCRLine(text=text, bbox=bbox, confidence=float(score or 0.0))
                )

        lines.sort(key=lambda ln: (ln.bbox[1] if ln.bbox else 0, ln.bbox[0] if ln.bbox else 0))
        full_text = "\n".join(line.text for line in lines)
        return OCRResult(lines=lines, full_text=full_text, method="rapidocr")

    def run_paddleocr(self, image: Image.Image) -> OCRResult:
        """PaddleOCR — optional fast engine (Paddle-based)."""
        import numpy as np

        self._load_paddleocr()
        ocr_image, scale = _resize_for_fast_ocr(image, max_dim=_RAPID_OCR_MAX_DIM)
        arr = np.array(ocr_image.convert("RGB"))
        raw = self._paddleocr.ocr(arr, cls=False)

        lines: list[OCRLine] = []
        if raw:
            for block in raw:
                if not block:
                    continue
                for box, (text, score) in block:
                    text = (text or "").strip()
                    if not text:
                        continue
                    xs = [float(p[0]) for p in box]
                    ys = [float(p[1]) for p in box]
                    bbox = _scale_bbox((min(xs), min(ys), max(xs), max(ys)), scale)
                    lines.append(
                        OCRLine(text=text, bbox=bbox, confidence=float(score or 0.0))
                    )

        lines.sort(key=lambda ln: (ln.bbox[1] if ln.bbox else 0, ln.bbox[0] if ln.bbox else 0))
        full_text = "\n".join(line.text for line in lines)
        return OCRResult(lines=lines, full_text=full_text, method="paddleocr")

    def run_easyocr(self, image: Image.Image) -> OCRResult:
        """EasyOCR — optional fast engine with decent handwriting support."""
        import numpy as np

        self._load_easyocr()
        ocr_image, scale = _resize_for_fast_ocr(image, max_dim=_RAPID_OCR_MAX_DIM)
        arr = np.array(ocr_image.convert("RGB"))
        raw = self._easyocr.readtext(arr)

        lines: list[OCRLine] = []
        for box, text, score in raw:
            text = (text or "").strip()
            if not text:
                continue
            xs = [float(p[0]) for p in box]
            ys = [float(p[1]) for p in box]
            bbox = _scale_bbox((min(xs), min(ys), max(xs), max(ys)), scale)
            lines.append(OCRLine(text=text, bbox=bbox, confidence=float(score or 0.0)))

        lines.sort(key=lambda ln: (ln.bbox[1] if ln.bbox else 0, ln.bbox[0] if ln.bbox else 0))
        full_text = "\n".join(line.text for line in lines)
        return OCRResult(lines=lines, full_text=full_text, method="easyocr")

    def run_tesseract(self, image: Image.Image) -> OCRResult:
        """Fast local OCR — lower accuracy on dense checkbox forms."""
        import pytesseract

        ocr_image, scale = _resize_for_fast_ocr(image, max_dim=_TESSERACT_MAX_DIM)
        # PSM 3 = automatic page layout (better on multi-column medical forms than PSM 6)
        data = pytesseract.image_to_data(
            ocr_image,
            config="--psm 3 --oem 3",
            output_type=pytesseract.Output.DICT,
        )
        line_map: dict[tuple[int, int, int], dict] = {}

        for i, text in enumerate(data["text"]):
            word = text.strip()
            if not word:
                continue
            key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
            conf = float(data["conf"][i]) if data["conf"][i] != "-1" else 0.0
            if conf < 25:
                continue
            x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]

            if key not in line_map:
                line_map[key] = {"words": [], "x0": x, "y0": y, "x1": x + w, "y1": y + h, "conf": []}
            entry = line_map[key]
            entry["words"].append(word)
            entry["x0"] = min(entry["x0"], x)
            entry["y0"] = min(entry["y0"], y)
            entry["x1"] = max(entry["x1"], x + w)
            entry["y1"] = max(entry["y1"], y + h)
            entry["conf"].append(conf)

        lines: list[OCRLine] = []
        for entry in line_map.values():
            text = " ".join(entry["words"])
            avg_conf = sum(entry["conf"]) / len(entry["conf"]) if entry["conf"] else 0.0
            bbox = _scale_bbox(
                (entry["x0"], entry["y0"], entry["x1"], entry["y1"]), scale
            )
            lines.append(OCRLine(text=text, bbox=bbox, confidence=avg_conf / 100.0))

        lines.sort(key=lambda ln: (ln.bbox[1] if ln.bbox else 0, ln.bbox[0] if ln.bbox else 0))
        full_text = "\n".join(line.text for line in lines)
        return OCRResult(lines=lines, full_text=full_text, method="tesseract")
