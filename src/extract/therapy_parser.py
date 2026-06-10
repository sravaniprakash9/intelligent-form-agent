"""Parse therapy session count and duration from Section V."""

from __future__ import annotations

import re

import cv2
import numpy as np
from PIL import Image

from src.extract.ocr_engine import OCRLine

try:
    from src.extract.region_ocr import RegionOCRReader
except ImportError:  # pragma: no cover
    RegionOCRReader = None  # type: ignore[misc, assignment]

_CHECKBOX_ARTIFACT = re.compile(r"^\[\s*\d*\s*.*therapy", re.IGNORECASE)
_WEEKS = re.compile(r"\d+\s*weeks?", re.IGNORECASE)


def _find_line(lines: list[OCRLine], pattern: str) -> OCRLine | None:
    regex = re.compile(pattern, re.IGNORECASE)
    for line in lines:
        if regex.search(line.text):
            return line
    return None


def _section_v_lines(lines: list[OCRLine]) -> list[OCRLine]:
    start: int | None = None
    end = len(lines)
    for i, line in enumerate(lines):
        lower = line.text.lower()
        if start is None and ("section v" in lower or "services requested" in lower):
            start = i
        elif start is not None and ("section vi" in lower or "clinical documentation" in lower):
            end = i
            break
    if start is None:
        return lines
    return lines[start:end]


def _ocr_digits_in_crop(
    image: Image.Image,
    box: tuple[int, int, int, int],
    crop_reader: "RegionOCRReader | None" = None,
) -> list[int]:
    """OCR digits from a crop; hybrid mode uses fast OCR + Surya fallback."""
    if crop_reader is not None:
        text = crop_reader.read_digits(image, box, min_len=1, max_len=3)
        return [int(text)] if text else []
    try:
        import pytesseract
    except ImportError:
        return []
    x0, y0, x1, y1 = box
    if x1 <= x0 or y1 <= y0:
        return []
    crop = image.crop((x0, y0, x1, y1))
    gray = np.array(crop.convert("L"))
    scaled = cv2.resize(gray, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    found: list[int] = []
    for img in (scaled, cv2.bitwise_not(cv2.threshold(scaled, 180, 255, cv2.THRESH_BINARY)[1])):
        for psm in (8, 7):
            text = pytesseract.image_to_string(
                img, config=f"--psm {psm} -c tessedit_char_whitelist=0123456789"
            ).strip()
            for token in re.findall(r"\d{1,3}", text):
                found.append(int(token))
    return found


def _pick_session_count(candidates: list[int], duration_weeks: int | None = None) -> int | None:
    """Prefer the session count over duration digits (e.g. 2 from '2 weeks')."""
    if not candidates:
        return None
    filtered = [c for c in candidates if c != duration_weeks or len(candidates) == 1]
    pool = filtered or candidates
    # Handwritten session counts are usually single values on the sessions row
    for val in pool:
        if 1 <= val <= 200:
            return val
    return None


def _sessions_near_anchor(
    lines: list[OCRLine],
    anchor: OCRLine,
    image_width: int,
) -> int | None:
    if anchor.bbox is None:
        return None
    ay0, ay1 = anchor.bbox[1], anchor.bbox[3]
    y_mid = (ay0 + ay1) / 2
    for line in lines:
        if line.bbox is None:
            continue
        ly = (line.bbox[1] + line.bbox[3]) / 2
        if abs(ly - y_mid) > 35:
            continue
        stripped = line.text.strip()
        if _CHECKBOX_ARTIFACT.search(stripped) or _WEEKS.search(stripped):
            continue
        if re.fullmatch(r"\d{1,3}", stripped):
            return int(stripped)
    # Digit in the sessions column band (mid-page) on the therapy row
    for line in lines:
        if line.bbox is None:
            continue
        ly = (line.bbox[1] + line.bbox[3]) / 2
        if abs(ly - y_mid) > 25:
            continue
        lx = line.bbox[0]
        if lx < image_width * 0.28 or lx > image_width * 0.58:
            continue
        stripped = line.text.strip()
        if re.fullmatch(r"\d{1,3}", stripped):
            return int(stripped)
    return None


def _center_x(line: OCRLine) -> float | None:
    if not line.bbox:
        return None
    return (line.bbox[0] + line.bbox[2]) / 2


def infer_therapy_from_session_column(lines: list[OCRLine]) -> str | None:
    """
    Sessions/duration are filled only on the checked therapy column.
    Map the session digit x-position between PT (left) and Speech (right) labels.
    """
    section_lines = _section_v_lines(lines)
    sessions_label = _find_line(section_lines, r"Number of Sessions")
    pt_line = _find_line(section_lines, r"Physical Therapy")
    st_line = _find_line(section_lines, r"Speech Therapy")
    if not sessions_label or not pt_line or not st_line:
        return None

    label_y = None
    if sessions_label.bbox:
        label_y = (sessions_label.bbox[1] + sessions_label.bbox[3]) / 2

    session_line: OCRLine | None = None
    for line in section_lines:
        if not line.bbox or not re.fullmatch(r"\d{1,3}", line.text.strip()):
            continue
        if label_y is not None:
            ly = (line.bbox[1] + line.bbox[3]) / 2
            if abs(ly - label_y) > 40:
                continue
        session_line = line
        break

    if not session_line:
        return None

    sx = _center_x(session_line)
    pt_x = _center_x(pt_line)
    st_x = _center_x(st_line)
    if sx is None or pt_x is None or st_x is None or st_x <= pt_x:
        return None

    span = st_x - pt_x
    if sx < pt_x + span / 3:
        return "physical_therapy"
    if sx < pt_x + 2 * span / 3:
        return "occupational_therapy"
    return "speech_therapy"


def parse_therapy_duration(
    text: str,
    lines: list[OCRLine] | None = None,
) -> str | None:
    """Duration from therapy row (e.g. '1 week'); avoid mistaking session count for duration."""
    section_v = ""
    start_m = re.search(r"services\s*requested|section\s*v\b", text, re.IGNORECASE)
    if start_m:
        start = start_m.start()
        end_m = re.search(r"clinical\s*documentation|section\s*vi\b", text[start:], re.IGNORECASE)
        section_v = text[start : start + end_m.start() if end_m else len(text)]
    else:
        lower = text.lower()
        start = lower.find("section v")
        end = lower.find("section vi")
        if start != -1:
            section_v = text[start : end if end != -1 else len(text)]

    section_lines = _section_v_lines(lines or [])
    sessions_label = _find_line(section_lines, r"Number of Sessions")

    def _weeks_text(raw: str) -> str | None:
        m = re.search(r"(\d+)\s*weeks?", raw, re.IGNORECASE)
        if m:
            n = int(m.group(1))
            return f"{n} week" if n == 1 else f"{n} weeks"
        m = re.search(r"(\d+)week\b", raw, re.IGNORECASE)
        if m:
            n = int(m.group(1))
            return f"{n} week" if n == 1 else f"{n} weeks"
        return None

    if sessions_label and sessions_label.bbox:
        ly = (sessions_label.bbox[1] + sessions_label.bbox[3]) / 2
        for line in section_lines:
            if not line.bbox:
                continue
            cy = (line.bbox[1] + line.bbox[3]) / 2
            if abs(cy - ly) > 45:
                continue
            weeks = _weeks_text(line.text)
            if weeks:
                return weeks

    for line in section_lines:
        weeks = _weeks_text(line.text)
        if weeks:
            return weeks

    for line in section_v.splitlines():
        weeks = _weeks_text(line)
        if weeks:
            return weeks

    duration = re.search(
        r"Duration[:\s]+(.+?)(?:\n|Frequency|$)", section_v, re.IGNORECASE | re.DOTALL
    )
    if duration:
        val = duration.group(1).strip()
        if val and not re.fullmatch(r"\d{1,2}", val):
            weeks = _weeks_text(val)
            if weeks:
                return weeks
            if val not in {"", "Other:", "Frequency:"}:
                return val
    return None


def _duration_weeks_from_lines(lines: list[OCRLine]) -> int | None:
    for line in lines:
        m = re.search(r"(\d+)\s*weeks?", line.text, re.IGNORECASE)
        if m:
            return int(m.group(1))
    return None


def _sessions_from_image(
    image: Image.Image,
    lines: list[OCRLine],
    crop_reader: "RegionOCRReader | None" = None,
) -> int | None:
    sessions_label = _find_line(lines, r"Number of Sessions")
    pt = _find_line(lines, r"Physical Therapy")
    duration_weeks = _duration_weeks_from_lines(lines)
    width = image.width

    if sessions_label and sessions_label.bbox:
        sx0, sy0, sx1, sy1 = [int(v) for v in sessions_label.bbox]
        # Full sessions row — handwritten value sits between label and duration column
        weeks_line = _find_line(lines, r"\d+\s*weeks?")
        x_end = int(weeks_line.bbox[0] - 8) if weeks_line and weeks_line.bbox else sx1 + 240
        x_end = min(max(x_end, sx1 + 80), width - 5)
        row_box = (sx0, sy0 - 6, x_end, sy1 + 8)
        picked = _pick_session_count(
            _ocr_digits_in_crop(image, row_box, crop_reader), duration_weeks
        )
        if picked is not None:
            return picked

    if pt and pt.bbox:
        px0, py0, px1, py1 = [int(v) for v in pt.bbox]
        row_box = (px1 + 4, py0 - 4, int(width * 0.48), py1 + 40)
        picked = _pick_session_count(
            _ocr_digits_in_crop(image, row_box, crop_reader), duration_weeks
        )
        if picked is not None:
            return picked

    return None


def parse_therapy_sessions(
    text: str,
    lines: list[OCRLine],
    image: Image.Image | None = None,
    *,
    skip_image_crop: bool = False,
    crop_reader: "RegionOCRReader | None" = None,
) -> int | None:
    """Extract session count; never treat checkbox artifacts like '[ 2 Physical Therapy' as sessions."""
    section_v = ""
    start_m = re.search(r"services\s*requested|section\s*v\b", text, re.IGNORECASE)
    if start_m:
        start = start_m.start()
        end_m = re.search(r"clinical\s*documentation|section\s*vi\b", text[start:], re.IGNORECASE)
        section_v = text[start:start + end_m.start() if end_m else len(text)]
    else:
        lower = text.lower()
        start = lower.find("section v")
        end = lower.find("section vi")
        if start != -1:
            section_v = text[start:end if end != -1 else len(text)]

    sessions_label = re.search(r"Number of Sessions", section_v, re.IGNORECASE)
    if sessions_label:
        tail = section_v[sessions_label.end() : sessions_label.end() + 120]
        for line in tail.splitlines()[:5]:
            stripped = line.strip()
            if not stripped or re.search(r"weeks?", stripped, re.I):
                continue
            if re.fullmatch(r"\d{1,3}", stripped):
                return int(stripped)

    section_lines = _section_v_lines(lines)
    for i, line in enumerate(section_lines):
        if "number of sessions" not in line.text.lower():
            continue
        row_match = re.search(r"number of sessions[:\s]*(\d+)", line.text, re.IGNORECASE)
        if row_match:
            return int(row_match.group(1))
        for nxt in section_lines[i + 1 : i + 6]:
            stripped = nxt.text.strip()
            if re.search(r"weeks?", stripped, re.I):
                continue
            if re.fullmatch(r"\d{1,3}", stripped):
                return int(stripped)

    pt = _find_line(section_lines, r"Physical Therapy")
    sessions_label = _find_line(section_lines, r"Number of Sessions")
    image_width = image.width if image else 1200
    for anchor in (sessions_label, pt):
        if anchor:
            val = _sessions_near_anchor(section_lines, anchor, image_width)
            if val is not None:
                return val

    if image is not None and not skip_image_crop:
        return _sessions_from_image(image, lines, crop_reader)

    return None
