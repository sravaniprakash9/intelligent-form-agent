"""Surya crop fallback for critical fields still missing after fast hybrid parse."""

from __future__ import annotations

import re

from PIL import Image

from src.extract.member_id_parser import (
    _GROUP_LABEL,
    _MEMBER_LABEL,
    _column_crop_box,
    _find_line as _find_line_pattern,
    _section_iii_lines,
)
from src.extract.ocr_engine import OCRLine
from src.extract.region_ocr import RegionOCRReader
from src.extract.schema import FormDocument
from src.extract.therapy_parser import _find_line
from src.extract.therapy_parser import _section_v_lines

_NPI = re.compile(r"\b(\d{10})\b")
_DATE = re.compile(r"\b(\d{1,2}/\d{1,2}/\d{4})\b")
_DIGITS = re.compile(r"\d{4,12}")
_SESSIONS = re.compile(r"\d{1,3}")


def _therapy_session_boxes(image: Image.Image, lines: list[OCRLine]) -> list[tuple[int, int, int, int]]:
    section_lines = _section_v_lines(lines)
    boxes: list[tuple[int, int, int, int]] = []
    sessions_label = _find_line(section_lines, r"Number of Sessions")
    pt = _find_line(section_lines, r"Physical Therapy")
    width = image.width

    if sessions_label and sessions_label.bbox:
        sx0, sy0, sx1, sy1 = [int(v) for v in sessions_label.bbox]
        weeks_line = _find_line(section_lines, r"\d+\s*weeks?")
        x_end = int(weeks_line.bbox[0] - 8) if weeks_line and weeks_line.bbox else sx1 + 240
        x_end = min(max(x_end, sx1 + 80), width - 5)
        boxes.append((sx0, sy0 - 6, x_end, sy1 + 8))

    if pt and pt.bbox:
        px0, py0, px1, py1 = [int(v) for v in pt.bbox]
        boxes.append((px1 + 4, py0 - 4, int(width * 0.48), py1 + 40))

    return boxes


def _section_iv_lines(lines: list[OCRLine]) -> list[OCRLine]:
    start: int | None = None
    end = len(lines)
    for i, line in enumerate(lines):
        lower = line.text.lower()
        if start is None and ("section iv" in lower or "provider information" in lower):
            start = i
        elif start is not None and ("section v" in lower or "services requested" in lower):
            end = i
            break
    if start is None:
        return lines
    return lines[start:end]


def _label_crop_below(
    lines: list[OCRLine], pattern: str, image: Image.Image, *, width_scale: float = 1.2
) -> tuple[int, int, int, int] | None:
    label = _find_line(lines, pattern)
    if not label or not label.bbox:
        return None
    x0, _y0, x1, y1 = label.bbox
    col_w = max(x1 - x0, image.width * 0.08) * width_scale
    left = max(0, int(x0 - 8))
    right = min(image.width, int(left + col_w))
    top = max(0, int(y1 - 2))
    bottom = min(image.height, int(y1 + 50))
    if right <= left or bottom <= top:
        return None
    return (left, top, right, bottom)


def refine_low_confidence_fields(
    doc: FormDocument,
    image: Image.Image,
    lines: list[OCRLine],
    reader: RegionOCRReader,
) -> tuple[FormDocument, list[str]]:
    """Re-OCR missing critical fields on crops (Surya only when fast read fails)."""
    refined: list[str] = []
    section_iii = _section_iii_lines(lines)
    section_iv = _section_iv_lines(lines)

    patient = doc.section_iii_patient
    if not patient.member_id:
        member_label = _find_line_pattern(section_iii, _MEMBER_LABEL)
        if member_label:
            box = _column_crop_box(member_label, image)
            if box:
                value = reader.read_digits(image, box, min_len=4, max_len=12)
                if value:
                    patient.member_id = value
                    refined.append("member_id")

    if not patient.group_number:
        group_label = _find_line_pattern(section_iii, _GROUP_LABEL)
        if group_label:
            box = _column_crop_box(group_label, image)
            if box:
                value = reader.read_digits(image, box, min_len=4, max_len=12)
                if value and value != patient.member_id:
                    patient.group_number = value
                    refined.append("group_number")

    if not patient.dob:
        box = _label_crop_below(section_iii, r"DOB", image, width_scale=0.5)
        if box:
            result = reader.read_crop(image, box, accept=_DATE)
            if result:
                from datetime import datetime

                parsed = None
                for fmt in ("%m/%d/%Y", "%m/%d/%y"):
                    try:
                        parsed = datetime.strptime(result.text.strip(), fmt).date()
                        break
                    except ValueError:
                        continue
                if parsed:
                    patient.dob = parsed
                    refined.append("dob")

    general = doc.section_ii_general
    if not general.prev_auth_number or general.prev_auth_number.upper() == "SECTION":
        box = _label_crop_below(lines, r"Prev\.?\s*Auth", image, width_scale=0.8)
        if box:
            text = reader.read_text(image, box)
            if text and "section" not in text.lower():
                general.prev_auth_number = text.split()[0][:32]
                refined.append("prev_auth_number")

    requesting = doc.section_iv_providers.requesting
    if not requesting.npi:
        box = _label_crop_below(section_iv, r"NPI\s*#", image, width_scale=0.7)
        if box:
            result = reader.read_crop(image, box, accept=_NPI)
            if result:
                requesting.npi = result.text
                refined.append("requesting_npi")

    service = doc.section_iv_providers.service
    if not service.npi:
        npi_lines = [ln for ln in section_iv if re.search(r"NPI\s*#", ln.text, re.I)]
        if len(npi_lines) >= 2 and npi_lines[1].bbox:
            x0, y0, x1, y1 = npi_lines[1].bbox
            box = (int(x0), int(y0), int(x1 + 120), int(y1 + 40))
            result = reader.read_crop(image, box, accept=_NPI)
            if result:
                service.npi = result.text
                refined.append("service_npi")

    therapies = doc.section_v_services.therapies
    needs_sessions = therapies and all(t.sessions is None for t in therapies)
    if needs_sessions:
        for box in _therapy_session_boxes(image, lines):
            value = reader.read_digits(image, box, min_len=1, max_len=3)
            if value:
                sessions = int(value)
                for therapy in therapies:
                    therapy.sessions = sessions
                refined.append("therapy_sessions")
                break

    if refined:
        suffix = f"+surya-crops({','.join(refined)})"
        base = (doc.extraction_method or "hybrid").split("+", 1)[0]
        doc.extraction_method = f"{base}{suffix}"

    return doc, refined
