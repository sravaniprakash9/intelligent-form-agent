"""Checkbox detection using OpenCV heuristics."""

from __future__ import annotations

import re

import cv2
import numpy as np
from PIL import Image

from src.extract.ocr_engine import OCRLine

# Checked boxes contain a tick — more ink than empty box borders (~0.15–0.25)
CHECKED_THRESHOLD = 0.28
MIN_INK = 0.25


def _gray_array(image: Image.Image) -> np.ndarray:
    return np.array(image.convert("L"))


def _fill_in_region(arr: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> float:
    if x1 <= x0 or y1 <= y0:
        return 0.0
    y0 = max(0, y0)
    x0 = max(0, x0)
    y1 = min(arr.shape[0], y1)
    x1 = min(arr.shape[1], x1)
    region = arr[y0:y1, x0:x1]
    if region.size == 0:
        return 0.0
    _, binary = cv2.threshold(region, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return float(np.count_nonzero(binary)) / binary.size


def _fill_left_of_line(arr: np.ndarray, line: OCRLine, box_size: int | None = None) -> float:
    if line.bbox is None:
        return 0.0
    x0, y0, x1, y1 = [int(v) for v in line.bbox]
    size = box_size or max(18, int((y1 - y0) * 0.95))
    cx1 = max(0, x0 - 4)
    cx0 = max(0, cx1 - size)
    cy0 = max(0, y0 - 2)
    cy1 = min(arr.shape[0], y1 + 2)
    return _fill_in_region(arr, cx0, cy0, cx1, cy1)


def _fill_at(arr: np.ndarray, x: int, y: int, size: int = 22) -> float:
    return _fill_in_region(arr, x, y, x + size, y + size)


def _find_line(lines: list[OCRLine], pattern: str) -> OCRLine | None:
    regex = re.compile(pattern, re.IGNORECASE)
    for line in lines:
        if regex.search(line.text) and line.bbox is not None:
            return line
    return None


def _pick_exclusive(scores: dict[str, float], min_threshold: float = MIN_INK) -> str | None:
    if not scores:
        return None
    best_key = max(scores, key=scores.get)
    best_val = scores[best_key]
    if best_val < min_threshold:
        return None
    others = [v for k, v in scores.items() if k != best_key]
    second = max(others) if others else 0.0
    if best_val >= second + 0.04:
        return best_key
    return None


def _row_max_fill(arr: np.ndarray, y: int, x_start: int, x_end: int, step: int = 25) -> float:
    return max((_fill_at(arr, x, y) for x in range(x_start, x_end, step)), default=0.0)


def _detect_left_right_row(
    arr: np.ndarray,
    lines: list[OCRLine],
    anchor_pattern: str,
    left_key: str,
    right_key: str,
    y_offset: int = 18,
    left_band: tuple[float, float] = (0.10, 0.26),
    right_band: tuple[float, float] = (0.28, 0.42),
) -> str | None:
    """When OCR misses one label (e.g. Non-Urgent), compare max fill in left vs right bands."""
    anchor = _find_line(lines, anchor_pattern)
    if anchor is None or anchor.bbox is None:
        return None
    y = int(anchor.bbox[1]) + y_offset
    width = arr.shape[1]
    left_max = _row_max_fill(arr, y, int(width * left_band[0]), int(width * left_band[1]))
    right_max = _row_max_fill(arr, y, int(width * right_band[0]), int(width * right_band[1]))
    if left_max >= MIN_INK and left_max >= right_max + 0.05:
        return left_key
    if right_max >= MIN_INK and right_max >= left_max + 0.05:
        return right_key
    return None


def detect_checked_labels(
    image: Image.Image,
    lines: list[OCRLine],
    label_patterns: list[str],
) -> dict[str, bool]:
    """Return which label patterns appear checked (legacy bool API)."""
    scores = detect_checkbox_scores(image, lines, label_patterns)
    return {k: v >= CHECKED_THRESHOLD for k, v in scores.items()}


def detect_checkbox_scores(
    image: Image.Image,
    lines: list[OCRLine],
    label_patterns: list[str],
) -> dict[str, float]:
    """Score checkbox fill ratio to the left of each matching label line."""
    arr = _gray_array(image)
    scores: dict[str, float] = {}

    for pattern in label_patterns:
        regex = re.compile(pattern, re.IGNORECASE)
        best = 0.0
        for line in lines:
            if line.bbox is None:
                continue
            text = line.text.strip()
            # OCR often reads empty box borders as "| Label" — not a real check
            if text.startswith("|"):
                continue
            if not regex.search(line.text):
                continue
            fill = _fill_left_of_line(arr, line)
            best = max(best, fill)
        scores[pattern] = best

    return scores


def _section_ii_text(text: str) -> str:
    start_m = re.search(r"general\s*information", text, re.IGNORECASE)
    if not start_m:
        lower = text.lower()
        start = lower.find("section ii")
        end = lower.find("section iii")
        if start != -1:
            return text[start:end if end != -1 else len(text)]
        return ""
    start = start_m.start()
    end_m = re.search(r"patient\s*information", text[start:], re.IGNORECASE)
    end = start + end_m.start() if end_m else len(text)
    return text[start:end]


def _scores_near_anchor(
    arr: np.ndarray,
    lines: list[OCRLine],
    anchor_pattern: str,
    label_patterns: list[str],
    y_tolerance: float = 45,
) -> dict[str, float]:
    anchor = _find_line(lines, anchor_pattern)
    if anchor is None or anchor.bbox is None:
        return {}
    y_mid = (anchor.bbox[1] + anchor.bbox[3]) / 2
    scores: dict[str, float] = {}
    for pattern in label_patterns:
        regex = re.compile(pattern, re.IGNORECASE)
        best = 0.0
        for line in lines:
            if line.bbox is None or not regex.search(line.text):
                continue
            ly = (line.bbox[1] + line.bbox[3]) / 2
            if abs(ly - y_mid) > y_tolerance:
                continue
            best = max(best, _fill_left_of_line(arr, line))
        scores[pattern] = best
    return scores


def _section_v_text(text: str) -> str:
    start_m = re.search(r"services\s*requested|section\s*v\b", text, re.IGNORECASE)
    if not start_m:
        lower = text.lower()
        start = lower.find("section v")
        end = lower.find("section vi")
        if start != -1:
            return text[start:end if end != -1 else len(text)]
        return ""
    start = start_m.start()
    end_m = re.search(r"clinical\s*documentation|section\s*vi\b", text[start:], re.IGNORECASE)
    end = start + end_m.start() if end_m else len(text)
    return text[start:end]


def _text_review_type_from_layout(section_ii: str) -> str | None:
    """RapidOCR puts a spurious 'Urgent' under Clinical Reason; Non-Urgent is checked."""
    if re.search(
        r"Clinical Reason[^\n]*\n\s*Urgent\s*\n\s*Review Type",
        section_ii,
        re.IGNORECASE | re.DOTALL,
    ):
        return "non_urgent"
    if re.search(r"Non-Urgent\s*\n\s*Clinical Reason", section_ii, re.IGNORECASE | re.DOTALL):
        return "non_urgent"
    return None


def _text_setting_fallback(section_v: str) -> str | None:
    """RapidOCR: 'Inpatient ] Outpatient[' means outpatient box is checked."""
    for line in section_v.splitlines():
        if not re.search(r"outpatient", line, re.I):
            continue
        if re.search(r"Inpatient\s*\]\s*Outpatient\s*\[", line, re.I):
            return "outpatient"
        if re.search(r"Inpatient\s*\[\s*\]\s*Outpatient", line, re.I):
            return "outpatient"
        if re.search(r"Outpatient\s*\[\s*\d", line, re.I):
            return "outpatient"
        if re.search(r"Outpatient\s*\[(?!\s*\])", line, re.I):
            return "outpatient"
        if re.search(r"Inpatient\s*\[\s*\d", line, re.I):
            return "inpatient"
    return None


def _text_therapies_fallback(section_v: str) -> str | None:
    """Infer checked therapy when OpenCV fails but sessions/duration are filled."""
    lower = section_v.lower()
    has_session_block = bool(
        re.search(r"number of sessions", lower)
        and re.search(r"\b\d{1,3}\b", section_v)
    )
    has_duration = bool(re.search(r"\d+\s*weeks?", section_v, re.I))
    if not (has_session_block or has_duration):
        return None

    if re.search(r"physical therapy\s+occupational therapy", section_v, re.I):
        return "physical_therapy"
    if "occupational therapy" in lower and re.search(
        r"occupational therapy\s*(\[|$)", section_v, re.I
    ):
        return "occupational_therapy"
    if "physical therapy" in lower and "occupational therapy" not in lower:
        return "physical_therapy"
    if "speech therapy" in lower and re.search(r"speech therapy\s*\[", section_v, re.I):
        return "speech_therapy"
    return None


def _text_checkbox_fallback(text: str) -> dict[str, str | None]:
    """OCR artifacts for request type (review type uses OpenCV-first resolution)."""
    section_ii = _section_ii_text(text)
    result: dict[str, str | None] = {}
    if re.search(r"\|\s*Extension", section_ii, re.IGNORECASE):
        result["request_type"] = "initial"
    return result


def _review_type_from_text_fallback(section_ii: str) -> str | None:
    """Text-only fallbacks when OpenCV cannot score checkbox ink."""
    layout_review = _text_review_type_from_layout(section_ii)
    if layout_review:
        return layout_review
    if re.search(r"\|\s*Urgent", section_ii, re.IGNORECASE):
        return "non_urgent"
    return None


def _urgent_label_line(lines: list[OCRLine]) -> OCRLine | None:
    for line in lines:
        if line.bbox and re.match(r"^Urgent\s*$", line.text.strip(), re.IGNORECASE):
            return line
    return None


def _review_type_from_label_order(section_ii: str) -> str | None:
    """
    RapidOCR column order before 'Review Type':
    - Non-urgent checked: 'Non-Urgent' appears, often before spurious 'Urgent' under Clinical Reason.
    - Urgent checked: only 'Urgent' appears (no 'Non-Urgent' before Review Type).
    """
    parts = re.split(r"Review\s*Type", section_ii, maxsplit=1, flags=re.IGNORECASE)
    pre_review = parts[0] if parts else section_ii
    nu_m = re.search(r"Non-?Urgent", pre_review, re.IGNORECASE)
    u_m = re.search(r"\bUrgent\b", pre_review, re.IGNORECASE)
    if u_m and not nu_m:
        return "urgent"
    if nu_m and u_m and nu_m.start() < u_m.start():
        return "non_urgent"
    if nu_m and not u_m:
        return "non_urgent"
    return None


def _review_type_from_checkbox_bands(
    arr: np.ndarray,
    lines: list[OCRLine],
) -> str | None:
    """Compare ink left of Non-Urgent vs Urgent labels on the same row."""
    nu_line = _find_line(lines, r"Non-?Urgent")
    u_line = _urgent_label_line(lines)
    if not nu_line or not nu_line.bbox or not u_line or not u_line.bbox:
        return None
    y = int((nu_line.bbox[1] + nu_line.bbox[3]) / 2)
    nu_x = int(nu_line.bbox[0])
    u_x = int(u_line.bbox[0])
    left = _row_max_fill(arr, y, max(0, nu_x - 90), nu_x - 4)
    right = _row_max_fill(arr, y, max(0, u_x - 90), u_x - 4)
    if left >= 0.40 and left >= right + 0.12:
        return "non_urgent"
    if right >= 0.40 and right >= left + 0.12:
        return "urgent"
    return None


def _resolve_review_type(
    image: Image.Image,
    lines: list[OCRLine],
    arr: np.ndarray,
    full_text: str,
) -> str | None:
    """Resolve urgent vs non-urgent using OpenCV, label order, then text fallbacks."""
    section_ii = _section_ii_text(full_text)

    review_scores = _scores_near_anchor(
        arr, lines, r"Review Type", [r"Non-?Urgent", r"Urgent"], y_tolerance=45
    )
    if not review_scores:
        review_scores = detect_checkbox_scores(image, lines, [r"Non-?Urgent", r"Urgent"])
    review_pick = _pick_exclusive(
        {k: v for k, v in review_scores.items() if v > 0},
        min_threshold=CHECKED_THRESHOLD,
    )
    if review_pick == r"Non-?Urgent":
        return "non_urgent"
    if review_pick == r"Urgent":
        return "urgent"

    label_order = _review_type_from_label_order(section_ii)
    if label_order:
        return label_order

    band_pick = _review_type_from_checkbox_bands(arr, lines)
    if band_pick:
        return band_pick

    return _review_type_from_text_fallback(section_ii)


def detect_texas_form_fields(image: Image.Image, lines: list[OCRLine], full_text: str = "") -> dict[str, str | None]:
    """
    Texas Prior Auth form checkbox groups with layout-aware fallbacks.
    Returns keys: review_type, request_type, gender, setting, therapies (comma-sep).
    """
    arr = _gray_array(image)
    text_hints = _text_checkbox_fallback(full_text)
    result: dict[str, str | None] = {
        "review_type": _resolve_review_type(image, lines, arr, full_text),
        "request_type": text_hints.get("request_type"),
        "gender": None,
        "setting": None,
        "therapies": None,
    }

    # ── Section II: Request Type ──
    request_scores = detect_checkbox_scores(
        image, lines, [r"Initial Request", r"Extension/Renewal/Amendment"]
    )
    request_pick = _pick_exclusive(
        {k: v for k, v in request_scores.items() if v > 0},
        min_threshold=CHECKED_THRESHOLD,
    )
    if not result["request_type"]:
        if request_pick and "Initial" in request_pick:
            result["request_type"] = "initial"
        elif request_pick:
            result["request_type"] = "extension_renewal_amendment"
        else:
            row_pick = _detect_left_right_row(
                arr, lines, r"Request Type", "initial", "extension_renewal_amendment",
                y_offset=8, left_band=(0.10, 0.28), right_band=(0.30, 0.42),
            )
            result["request_type"] = row_pick

    # ── Section III: Gender (Unknown label often missing from OCR) ──
    gender_specs = [
        (r"\bMale\b", "male"),
        (r"\bFemale\b", "female"),
        (r"\bOther\b", "other"),
        (r"Unknown", "unknown"),
    ]
    gender_scores: dict[str, float] = {}
    for pattern, key in gender_specs:
        gender_scores[key] = detect_checkbox_scores(image, lines, [pattern]).get(pattern, 0.0)

    # Unknown label often missing — scan right side of gender row
    female_line = _find_line(lines, r"\bFemale\b")
    male_line = _find_line(lines, r"\bMale\b")
    if female_line and female_line.bbox is not None:
        y_base = int(female_line.bbox[1])
        for y_try in (y_base - 10, y_base, y_base + 15, y_base + 30):
            for fx in range(int(female_line.bbox[2]) + 20, int(female_line.bbox[2]) + 200, 30):
                gender_scores["unknown"] = max(
                    gender_scores.get("unknown", 0.0), _fill_at(arr, fx, y_try)
                )
    if male_line and male_line.bbox is not None:
        y_row = int(male_line.bbox[1]) + 12
        width = arr.shape[1]
        gender_scores["unknown"] = max(
            gender_scores.get("unknown", 0.0),
            _row_max_fill(arr, y_row, int(width * 0.82), int(width * 0.95), step=20),
        )

    # Other label on a second row — only count if clearly higher than unknown
    other_line = _find_line(lines, r"^Other$")
    if other_line and other_line.bbox is not None:
        gender_scores["other"] = max(
            gender_scores.get("other", 0.0), _fill_left_of_line(arr, other_line)
        )

    gender_pick = _pick_exclusive(gender_scores)
    result["gender"] = gender_pick

    # ── Section V: Setting ──
    setting_scores: dict[str, float] = {}
    for pattern, key in [
        (r"Inpatient", "inpatient"),
        (r"Outpatient", "outpatient"),
        (r"Provider Office", "provider_office"),
        (r"Home", "home"),
    ]:
        setting_scores[key] = detect_checkbox_scores(image, lines, [pattern]).get(pattern, 0.0)

    # Setting row often OCR'd as one long line — probe left of each label on shared row
    # Setting row: "Inpatient [ ] Outpatient [ ] ..." in Section V (not phone numbers in Section I)
    setting_line = None
    for line in lines:
        if line.bbox is None:
            continue
        if re.search(r"Outpatient", line.text, re.I) and re.search(
            r"Inpatient|Observation|Provider Office", line.text, re.I
        ):
            setting_line = line
            break
    if setting_line and setting_line.bbox is not None:
        sy = int(setting_line.bbox[1])
        sw = arr.shape[1]
        setting_scores["inpatient"] = max(
            setting_scores.get("inpatient", 0.0),
            _row_max_fill(arr, sy, int(sw * 0.12), int(sw * 0.18)),
        )
        setting_scores["outpatient"] = max(
            setting_scores.get("outpatient", 0.0),
            _row_max_fill(arr, sy, int(sw * 0.20), int(sw * 0.28)),
        )

    io_scores = {k: v for k, v in setting_scores.items() if k in ("inpatient", "outpatient")}
    setting_pick = _pick_exclusive(io_scores)
    if not setting_pick and io_scores:
        # When OCR marks both boxes similarly (e.g. "[ 7 ]"), pick the higher score.
        best = max(io_scores, key=io_scores.get)
        best_val = io_scores[best]
        second = max((v for k, v in io_scores.items() if k != best), default=0.0)
        if best_val >= MIN_INK and best_val > second:
            setting_pick = best
    if not setting_pick:
        setting_pick = _text_setting_fallback(_section_v_text(full_text))
    result["setting"] = setting_pick

    # ── Section V: Therapies ──
    therapy_map = [
        (r"Physical Therapy", "physical_therapy"),
        (r"Occupational Therapy", "occupational_therapy"),
        (r"Speech Therapy", "speech_therapy"),
        (r"Cardiac Rehab", "cardiac_rehab"),
        (r"Mental Health", "mental_health_substance_abuse"),
    ]
    therapy_scores = detect_checkbox_scores(image, lines, [p for p, _ in therapy_map])

    # Therapy row — scan checkbox bands (PT left, OT center, ST right)
    pt_line = _find_line(lines, r"Physical Therapy")
    if pt_line and pt_line.bbox is not None:
        py = int(pt_line.bbox[1])
        therapy_scores[r"Physical Therapy"] = max(
            therapy_scores.get(r"Physical Therapy", 0.0),
            _row_max_fill(arr, py, 260, 400, step=20),
        )
        therapy_scores[r"Occupational Therapy"] = max(
            therapy_scores.get(r"Occupational Therapy", 0.0),
            _row_max_fill(arr, py, 480, 720, step=20),
        )
        st_line = _find_line(lines, r"Speech Therapy")
        if st_line and st_line.bbox is not None:
            sy = int(st_line.bbox[1])
            therapy_scores[r"Speech Therapy"] = max(
                therapy_scores.get(r"Speech Therapy", 0.0),
                _row_max_fill(arr, sy, 980, 1280, step=20),
            )

    checked_therapies = [
        key for pattern, key in therapy_map if therapy_scores.get(pattern, 0) >= CHECKED_THRESHOLD
    ]
    if checked_therapies:
        result["therapies"] = ",".join(checked_therapies)
    elif not result["therapies"]:
        result["therapies"] = _text_therapies_fallback(_section_v_text(full_text))

    return result
