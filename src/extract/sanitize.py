"""Clean OCR placeholder values from extracted fields."""

from __future__ import annotations

import re

_JUNK_VALUES = frozenset({"section", "n/a", "none", "na", "unknown"})


def clean_text_field(value: str | None) -> str | None:
    """Drop label placeholders and OCR checkbox artifacts from a text field."""
    if not value:
        return None
    stripped = value.strip()
    if not stripped:
        return None
    if stripped.startswith("|"):
        return None
    if re.search(r"\(if different\)", stripped, re.IGNORECASE):
        return None
    if stripped.lower() in _JUNK_VALUES:
        return None
    return stripped


_OCR_TEXT_FIXES: list[tuple[str, str]] = [
    (r"\bSectlon\b", "Section"),
    (r"\bMedica1d\b", "Medicaid"),
    (r"\blssuer\b", "Issuer"),
    (r"\bNon[-\s]?Urgent\b", "Non-Urgent"),
    (r"\bOut[-\s]?Patient\b", "Outpatient"),
    (r"\bIn[-\s]?Patient\b", "Inpatient"),
    (r"\bMemberorMedicaid\b", "Member or Medicaid"),
    (r"\bGENERALINFORMATION\b", "GENERAL INFORMATION"),
    (r"\bPATIENTINFORMATION\b", "PATIENT INFORMATION"),
    (r"\bPROVIDERINFORMATION\b", "PROVIDER INFORMATION"),
    (r"\bQccupational\b", "Occupational"),
]


def _insert_section_spaces(text: str) -> str:
    text = re.sub(r"SECTION([IVX]{1,4})(?=[A-Z\-])", r"SECTION \1 ", text, flags=re.IGNORECASE)
    text = re.sub(r"SECTION\s+([IVX]{1,4})([A-Z])", r"SECTION \1 \2", text, flags=re.IGNORECASE)
    return text


def _split_camel_names(text: str) -> str:
    """ElizabethFoley -> Elizabeth Foley (RapidOCR drops spaces in names)."""
    return re.sub(r"([a-z])([A-Z])", r"\1 \2", text)


def _split_mashed_codes_dates(text: str) -> str:
    text = re.sub(r"(\d{1,2}/\d{1,2}/\d{4})(\d{1,2}/\d{1,2}/\d{4})", r"\1 \2", text)
    text = re.sub(r"(?<![/\d])(\d{4,5})(\d{1,2}/\d{1,2}/\d{4})", r"\1 \2", text)
    return text


def uses_fast_ocr_normalization(extraction_method: str | None) -> bool:
    method = (extraction_method or "").lower()
    return method in ("rapidocr", "tesseract", "paddleocr", "easyocr") or method.startswith(
        ("hybrid:", "hybrid+")
    )


def normalize_ocr_text(text: str) -> str:
    """Fix common RapidOCR/Tesseract mangling before template parsing."""
    text = _insert_section_spaces(text)
    for pattern, replacement in _OCR_TEXT_FIXES:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    text = _split_mashed_codes_dates(text)
    text = _split_camel_names(text)
    return text


def clean_prev_auth_number(value: str | None) -> str | None:
    cleaned = clean_text_field(value)
    if not cleaned:
        return None
    if cleaned.lower() == "section":
        return None
    if not re.search(r"\d", cleaned):
        return None
    if cleaned.strip(".") == "":
        return None
    return cleaned
