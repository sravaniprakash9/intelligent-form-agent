"""Image preprocessing before OCR."""

import cv2
import numpy as np
from PIL import Image


def preprocess(image: Image.Image, *, fast: bool = False) -> Image.Image:
    """Light preprocessing: grayscale optional path, denoise, contrast.

    fast=True skips denoise/CLAHE for quicker Surya batch OCR only.
    """
    if fast:
        return image.convert("RGB")
    arr = np.array(image)
    if len(arr.shape) == 3:
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    else:
        gray = arr

    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    return Image.fromarray(cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB))


def preprocess_light_enhance(image: Image.Image) -> Image.Image:
    """CLAHE + mild sharpen for RapidOCR/Tesseract (faster than full denoise)."""
    arr = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    sharpened = cv2.filter2D(
        enhanced,
        -1,
        np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32),
    )
    return Image.fromarray(cv2.cvtColor(sharpened, cv2.COLOR_GRAY2RGB))


def preprocess_for_engine(image: Image.Image, engine: str, *, fast: bool = False) -> Image.Image:
    """Pick preprocessing tuned to the OCR engine."""
    engine = engine.lower()
    if engine == "surya":
        return preprocess(image, fast=fast)
    return preprocess_light_enhance(image)
