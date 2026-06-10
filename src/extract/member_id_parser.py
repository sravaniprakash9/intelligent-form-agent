"""Parse Member/Medicaid ID and Group # from Section III (layout-aware)."""

from __future__ import annotations

import re

from PIL import Image

from src.extract.ocr_engine import OCRLine

try:
    from src.extract.region_ocr import RegionOCRReader
except ImportError:  # pragma: no cover
    RegionOCRReader = None  # type: ignore[misc, assignment]

_MEMBER_LABEL = re.compile(r"Member\s*or\s*Medicaid\s*ID", re.IGNORECASE)
_GROUP_LABEL = re.compile(r"Group\s*#", re.IGNORECASE)
_DIGIT_ID = re.compile(r"^\d{4,12}$")
_INLINE_MEMBER = re.compile(
    r"Member\s*or\s*Medicaid\s*ID\s*#\s*:?\s*(\d{4,12})",
    re.IGNORECASE,
)
_INLINE_GROUP = re.compile(r"Group\s*#\s*:?\s*(\d{4,12})", re.IGNORECASE)


def _find_line(lines: list[OCRLine], pattern: re.Pattern[str]) -> OCRLine | None:
    for line in lines:
        if pattern.search(line.text):
            return line
    return None


def _section_iii_lines(lines: list[OCRLine]) -> list[OCRLine]:
    start: int | None = None
    end = len(lines)
    for i, line in enumerate(lines):
        lower = line.text.lower()
        if start is None and "section iii" in lower:
            start = i
        elif start is not None and ("section iv" in lower or "provider information" in lower):
            end = i
            break
    if start is None:
        return lines
    return lines[start:end]


def _center_x(line: OCRLine) -> float | None:
    if not line.bbox:
        return None
    return (line.bbox[0] + line.bbox[2]) / 2


def _center_y(line: OCRLine) -> float | None:
    if not line.bbox:
        return None
    return (line.bbox[1] + line.bbox[3]) / 2


def _ocr_digits_in_crop(
    image: Image.Image,
    box: tuple[int, int, int, int],
    crop_reader: "RegionOCRReader | None" = None,
) -> str | None:
    if crop_reader is not None:
        return crop_reader.read_digits(image, box, min_len=4, max_len=12)
    try:
        import cv2
        import numpy as np
        import pytesseract
    except ImportError:
        return None
    x0, y0, x1, y1 = box
    if x1 <= x0 or y1 <= y0:
        return None
    crop = image.crop((x0, y0, x1, y1))
    gray = np.array(crop.convert("L"))
    scaled = cv2.resize(gray, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    for img in (scaled, cv2.bitwise_not(cv2.threshold(scaled, 180, 255, cv2.THRESH_BINARY)[1])):
        for psm in (7, 8):
            text = pytesseract.image_to_string(
                img, config=f"--psm {psm} -c tessedit_char_whitelist=0123456789"
            ).strip()
            match = re.search(r"\d{4,12}", text)
            if match:
                return match.group(0)
    return None


def _column_crop_box(label: OCRLine, image: Image.Image) -> tuple[int, int, int, int] | None:
    if not label.bbox:
        return None
    x0, _y0, x1, y1 = label.bbox
    col_w = max(x1 - x0, image.width * 0.12)
    left = max(0, int(x0 - 8))
    right = min(image.width, int(left + col_w + 24))
    top = max(0, int(y1 - 2))
    bottom = min(image.height, int(y1 + 55))
    if right <= left or bottom <= top:
        return None
    return (left, top, right, bottom)


def _parse_inline(section_iii: str) -> tuple[str | None, str | None]:
    member = None
    group = None
    for line in section_iii.splitlines():
        if member is None:
            m = _INLINE_MEMBER.search(line)
            if m:
                member = m.group(1)
        if group is None:
            g = _INLINE_GROUP.search(line)
            if g:
                group = g.group(1)
    return member, group


def _parse_from_spatial(
    section_lines: list[OCRLine],
    image: Image.Image | None,
    crop_reader: "RegionOCRReader | None" = None,
) -> tuple[str | None, str | None]:
    member_label = _find_line(section_lines, _MEMBER_LABEL)
    group_label = _find_line(section_lines, _GROUP_LABEL)
    if not member_label or not group_label:
        return None, None

    member_cx = _center_x(member_label)
    group_cx = _center_x(group_label)
    label_y = _center_y(member_label) or _center_y(group_label)
    if member_cx is None or group_cx is None or label_y is None:
        return None, None

    member_id: str | None = None
    group_number: str | None = None
    for line in section_lines:
        digits = line.text.strip()
        if not _DIGIT_ID.match(digits) or not line.bbox:
            continue
        cy = _center_y(line)
        if cy is None or cy < label_y - 8 or cy > label_y + 90:
            continue
        digit_cx = _center_x(line)
        if digit_cx is None:
            continue
        if abs(digit_cx - member_cx) <= abs(digit_cx - group_cx):
            member_id = digits
        else:
            group_number = digits

    if image is not None:
        if member_id is None and member_label.bbox:
            box = _column_crop_box(member_label, image)
            if box:
                member_id = _ocr_digits_in_crop(image, box, crop_reader)
        if group_number is None and group_label.bbox:
            box = _column_crop_box(group_label, image)
            if box:
                group_number = _ocr_digits_in_crop(image, box, crop_reader)

    return member_id, group_number


def _parse_lone_digit_line(section_iii: str) -> tuple[str | None, str | None]:
    """When OCR drops Member/Group labels, a single ID line may remain in the patient block."""
    digit_lines: list[str] = []
    for line in section_iii.splitlines():
        stripped = line.strip()
        if _DIGIT_ID.match(stripped):
            digit_lines.append(stripped)
    if len(digit_lines) == 1:
        return digit_lines[0], None
    return None, None


def _parse_from_text_order(section_iii: str) -> tuple[str | None, str | None]:
    """Fallback when OCR lines lack bboxes — avoid Group regex stealing member ID."""
    lines = section_iii.splitlines()
    member_idx: int | None = None
    group_idx: int | None = None
    for i, line in enumerate(lines):
        if member_idx is None and _MEMBER_LABEL.search(line):
            member_idx = i
        if group_idx is None and _GROUP_LABEL.search(line):
            group_idx = i
    if member_idx is None or group_idx is None:
        return None, None

    start = max(member_idx, group_idx) + 1
    digit_lines: list[str] = []
    for line in lines[start:]:
        stripped = line.strip()
        if _DIGIT_ID.match(stripped):
            digit_lines.append(stripped)
        elif digit_lines:
            break

    if len(digit_lines) >= 2:
        return digit_lines[0], digit_lines[1]
    return None, None


def parse_member_and_group_ids(
    section_iii: str,
    lines: list[OCRLine],
    image: Image.Image | None = None,
    crop_reader: "RegionOCRReader | None" = None,
) -> tuple[str | None, str | None]:
    """Return (member_id, group_number) using inline, spatial, then text-order fallbacks."""
    member, group = _parse_inline(section_iii)
    if member or group:
        return member, group

    section_lines = _section_iii_lines(lines)
    member, group = _parse_from_spatial(section_lines, image, crop_reader)
    if member or group:
        return member, group

    member, group = _parse_from_text_order(section_iii)
    if member or group:
        return member, group

    return _parse_lone_digit_line(section_iii)
