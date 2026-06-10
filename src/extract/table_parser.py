"""Parse Section V procedure table from OCR lines."""

from __future__ import annotations

import re
from datetime import datetime

from src.extract.ocr_engine import OCRLine
from src.extract.schema import ProcedureRow

DATE_PATTERN = re.compile(r"\b(\d{1,2}/\d{1,2}/\d{4})\b")
CODE_PATTERN = re.compile(r"\b([A-Z]?\d{5}[A-Z]?|[A-Z]{1,2}\d{4,5})\b")
ICD_PATTERN = re.compile(r"\b([A-TV-Z][0-9][0-9A-Z](?:\.[0-9A-Z]{1,4})?[A-Z]?)\b")
YEAR_PATTERN = re.compile(r"^\d{4}$")
_SECTION_V_START = re.compile(r"section\s*v|services\s*requested", re.IGNORECASE)
_CPT_CODE = re.compile(r"^\d{5}$")

_SKIP_PREFIXES = (
    "inpatient",
    "outpatient",
    "observation",
    "provider office",
    "home",
    "day surgery",
    "other:",
    "mental health",
    "cardiac rehab",
    "speech therapy",
    "physical therapy",
    "occupational therapy",
    "number of sessions",
    "home health",
    "dme ",
)
_HEADER_FRAGMENTS = (
    "planned service",
    "diagnosis description",
    "start date",
    "end date",
    "icd version",
    "procedure",
    "supporting diagnoses",
)


def _parse_date(value: str):
    for fmt in ("%m/%d/%Y", "%m/%d/%y"):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    return None


def _normalize_icd(token: str) -> str | None:
    m = ICD_PATTERN.search(token)
    if m:
        return m.group(1)
    mangled = re.search(r"\.(\d{2,3}\.\d{1,4})", token)
    if mangled:
        body = mangled.group(1)
        category = body.split(".")[0]
        if len(category) == 3:
            body = f"{category[1:]}.{body.split('.', 1)[1]}"
        return f"Z{body}"
    return None


def _is_diagnosis_fragment(text: str) -> bool:
    t = text.strip()
    if not t:
        return False
    if re.match(r"^Other\w+", t, re.IGNORECASE):
        return True
    if re.match(r"^Other\s+", t, re.IGNORECASE):
        return True
    return False


def _normalize_service_name(text: str) -> str:
    t = text.strip()
    t = re.sub(r"([a-z])([A-Z])", r"\1 \2", t)
    t = re.sub(r"(\w)(with|follow-?up)\b", r"\1 \2", t, flags=re.IGNORECASE)
    return t.strip()


def _is_procedure_text(text: str) -> bool:
    if _is_diagnosis_fragment(text):
        return False
    if DATE_PATTERN.search(text) or _CPT_CODE.fullmatch(text.strip().lstrip(".")):
        return False
    if _normalize_icd(text):
        return False
    if len(text.strip()) < 4:
        return False
    return bool(re.search(r"[A-Za-z]{3,}", text))


def _procedure_zone_lines(lines: list[OCRLine]) -> list[str]:
    text_lines = [ln.text.strip() for ln in lines if ln.text.strip()]
    start_idx = 0
    for i, line in enumerate(text_lines):
        if _SECTION_V_START.search(line):
            start_idx = i + 1
            break

    zone: list[str] = []
    for line in text_lines[start_idx:]:
        lower = line.lower()
        if lower.startswith("section vi") or "clinical documentation" in lower:
            break
        if any(h in lower for h in _HEADER_FRAGMENTS):
            continue
        if lower.strip() in {"code", "code code"}:
            continue
        if lower.startswith(_SKIP_PREFIXES):
            break
        zone.append(line)
    return zone


def _cpt_codes_in_line(line: str) -> list[str]:
    codes: list[str] = []
    for raw in CODE_PATTERN.findall(line):
        token = raw.lstrip(".")
        if YEAR_PATTERN.match(token) or not _CPT_CODE.fullmatch(token):
            continue
        codes.append(token)
    return codes


def _row_from_cpt_anchor(zone: list[str], code_idx: int, code: str) -> ProcedureRow | None:
    window = zone[max(0, code_idx - 4) : min(len(zone), code_idx + 5)]
    dates: list[str] = []
    icd_code: str | None = None
    diagnosis: str | None = None
    planned_service: str | None = None

    for part in window:
        dates.extend(DATE_PATTERN.findall(part))
        if icd_code is None:
            icd_code = _normalize_icd(part)
        if _is_diagnosis_fragment(part):
            diagnosis = part.strip()

    for j, part in enumerate(window):
        if part.strip() == code or code in part.replace(" ", ""):
            proc_parts: list[str] = []
            for k in range(j - 1, -1, -1):
                candidate = window[k].strip()
                if _is_procedure_text(candidate):
                    proc_parts.insert(0, candidate)
                elif proc_parts:
                    break
            if proc_parts:
                planned_service = _normalize_service_name(" ".join(proc_parts))
            if not planned_service:
                for k in range(j + 1, len(window)):
                    candidate = window[k].strip()
                    if _is_procedure_text(candidate):
                        planned_service = _normalize_service_name(candidate)
                        break
            break

    if not planned_service:
        for part in window:
            if _is_procedure_text(part):
                planned_service = _normalize_service_name(part)
                break

    if not dates:
        return None

    return ProcedureRow(
        planned_service=planned_service,
        code=code,
        start_date=_parse_date(dates[0]),
        end_date=_parse_date(dates[1]) if len(dates) > 1 else None,
        diagnosis_description=_normalize_service_name(diagnosis) if diagnosis else None,
        icd_code=icd_code,
    )


def _row_from_date_flush(pending: list[str], date_line: str) -> ProcedureRow | None:
    row_lines = pending + [date_line]
    joined = " | ".join(row_lines)
    dates = DATE_PATTERN.findall(joined)
    if not dates:
        return None
    icd_code = None
    diagnosis = None
    for part in reversed(row_lines):
        if icd_code is None:
            icd_code = _normalize_icd(part)
        if diagnosis is None and _is_diagnosis_fragment(part):
            diagnosis = part.strip()
    proc_codes = [
        c
        for c in CODE_PATTERN.findall(joined)
        if not YEAR_PATTERN.match(c.lstrip(".")) and _CPT_CODE.fullmatch(c.lstrip("."))
    ]
    planned_service = None
    for part in row_lines:
        if _is_procedure_text(part):
            planned_service = _normalize_service_name(part)
            break
    return ProcedureRow(
        planned_service=planned_service,
        code=proc_codes[0] if proc_codes else None,
        start_date=_parse_date(dates[0]),
        end_date=_parse_date(dates[1]) if len(dates) > 1 else None,
        diagnosis_description=_normalize_service_name(diagnosis) if diagnosis else None,
        icd_code=icd_code,
    )


def parse_procedure_table(lines: list[OCRLine]) -> list[ProcedureRow]:
    """Extract procedure rows from Section V (CPT-anchored for scrambled OCR)."""
    zone = _procedure_zone_lines(lines)
    rows: list[ProcedureRow] = []
    seen: set[tuple[str | None, str | None, str | None]] = set()

    for i, line in enumerate(zone):
        for code in _cpt_codes_in_line(line):
            row = _row_from_cpt_anchor(zone, i, code)
            if not row:
                continue
            key = (row.code, str(row.start_date), str(row.end_date))
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)

    covered_dates = {str(r.start_date) for r in rows if r.start_date}
    pending: list[str] = []
    for line in zone:
        if _cpt_codes_in_line(line) and DATE_PATTERN.search(line):
            pending = []
            continue
        if DATE_PATTERN.search(line):
            row = _row_from_date_flush(pending, line)
            pending = []
            if not row or not row.start_date:
                continue
            if str(row.start_date) in covered_dates:
                continue
            key = (row.code, str(row.start_date), str(row.end_date))
            if key in seen:
                continue
            seen.add(key)
            covered_dates.add(str(row.start_date))
            rows.append(row)
            continue
        if not _cpt_codes_in_line(line):
            pending.append(line)

    rows.sort(
        key=lambda r: (r.start_date or datetime.min.date(), r.code or ""),
    )
    return rows
