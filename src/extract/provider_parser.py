"""Parse Section IV requesting vs service providers (layout-aware)."""

from __future__ import annotations

import re

from src.extract.ocr_engine import OCRLine
from src.extract.schema import ProviderInfo

_REQUESTING_HEADER = re.compile(
    r"Requesting\s+Provider\s+or\s+Facility", re.IGNORECASE
)
_SERVICE_HEADER = re.compile(r"Service\s+Provider\s+or\s+Facility", re.IGNORECASE)
_NPI = re.compile(r"\b(\d{9,10})\b")
_PHONE = re.compile(r"\(\d{2,3}\)-\d{3}-\d{4}")
_PERSON_NAME = re.compile(r"^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+$")
_PERSON_NAME_MASHED = re.compile(r"^[A-Z][a-z]+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*$")
_SKIP_NAMES = frozenset(
    {
        "male", "female", "other", "unknown", "phone", "name", "section",
        "observation", "inpatient", "outpatient", "start date", "end date",
        "provider office", "physical therapy", "occupational therapy",
        "speech therapy", "cardiac rehab", "mental health", "day surgery",
        "home health", "planned service",
    }
)


def _center_x(line: OCRLine) -> float | None:
    if not line.bbox:
        return None
    return (line.bbox[0] + line.bbox[2]) / 2


def _center_y(line: OCRLine) -> float | None:
    if not line.bbox:
        return None
    return (line.bbox[1] + line.bbox[3]) / 2


def _normalize_name(text: str) -> str | None:
    stripped = text.strip()
    if _PERSON_NAME.match(stripped):
        return stripped
    if _PERSON_NAME_MASHED.match(stripped):
        return re.sub(r"([a-z])([A-Z])", r"\1 \2", stripped)
    return None


def _section_iv_lines(lines: list[OCRLine]) -> list[OCRLine]:
    start: int | None = None
    end = len(lines)
    for i, line in enumerate(lines):
        lower = line.text.lower()
        if start is None and "section iv" in lower:
            start = i
        elif start is not None and (
            "section v" in lower or "services requested" in lower
        ):
            end = i
            break
    if start is None:
        return [ln for ln in lines if ln.bbox]
    return [ln for ln in lines[start:end] if ln.bbox]


def _column_midpoint(section_lines: list[OCRLine]) -> float | None:
    req_cx: float | None = None
    svc_cx: float | None = None
    for line in section_lines:
        cx = _center_x(line)
        if cx is None:
            continue
        if req_cx is None and _REQUESTING_HEADER.search(line.text):
            req_cx = cx
        elif svc_cx is None and _SERVICE_HEADER.search(line.text):
            svc_cx = cx
        if req_cx is not None and svc_cx is not None:
            break
    if req_cx is not None and svc_cx is not None:
        return (req_cx + svc_cx) / 2
    return None


def _column_of(line: OCRLine, midpoint: float) -> str:
    cx = _center_x(line)
    if cx is None:
        return "requesting"
    return "requesting" if cx < midpoint else "service"


def _anchor_y(section_lines: list[OCRLine]) -> float:
    for line in section_lines:
        if _REQUESTING_HEADER.search(line.text) and line.bbox:
            return line.bbox[1]
    for line in section_lines:
        if line.bbox:
            return line.bbox[1]
    return 0.0


def _section_iv_y_range(section_lines: list[OCRLine]) -> tuple[float, float]:
    """Provider block y-bounds (exclude Section I/III phone labels above header)."""
    min_y = _anchor_y(section_lines)
    max_y = min_y + 420
    for line in section_lines:
        if line.bbox and re.search(
            r"services\s*requested|section\s*v\b", line.text, re.I
        ):
            max_y = min(max_y, line.bbox[1])
    return min_y, max_y


def _in_provider_block(line: OCRLine, y_min: float, y_max: float) -> bool:
    cy = _center_y(line)
    return cy is not None and y_min <= cy <= y_max


def _name_near_label(
    label: OCRLine | None,
    section_lines: list[OCRLine],
    midpoint: float,
    column: str,
) -> str | None:
    if label is None or not label.bbox:
        return None
    lx = _center_x(label)
    ly = _center_y(label)
    if lx is None or ly is None:
        return None
    best: tuple[float, str] | None = None
    for line in section_lines:
        if _column_of(line, midpoint) != column:
            continue
        name = _normalize_name(line.text)
        if not name or name.lower() in _SKIP_NAMES:
            continue
        if not line.bbox:
            continue
        cx = _center_x(line)
        cy = _center_y(line)
        if cx is None or cy is None:
            continue
        if cy < ly - 5:
            continue
        if abs(cy - ly) > 80:
            continue
        dist = (cx - lx) ** 2 + (cy - ly) ** 2
        if best is None or dist < best[0]:
            best = (dist, name)
    return best[1] if best else None


def _primary_provider_name(
    section_lines: list[OCRLine],
    midpoint: float,
    column: str,
    y_min: float,
) -> str | None:
    """Provider name on the first row below section headers."""
    names: list[tuple[float, str]] = []
    for line in section_lines:
        if _column_of(line, midpoint) != column:
            continue
        name = _normalize_name(line.text)
        if not name or name.lower() in _SKIP_NAMES:
            continue
        cy = _center_y(line)
        if cy is None:
            continue
        row_offset = cy - y_min
        if 25 <= row_offset <= 120:
            names.append((cy, name))
    if not names:
        return None
    names.sort(key=lambda item: item[0])
    return names[0][1]


def _phone_near_label(
    label: OCRLine | None,
    section_lines: list[OCRLine],
    midpoint: float,
    column: str,
) -> str | None:
    if label is None or not label.bbox:
        return None
    lx = _center_x(label)
    ly = _center_y(label)
    if lx is None or ly is None:
        return None
    best: tuple[float, str] | None = None
    for line in section_lines:
        if _column_of(line, midpoint) != column:
            continue
        match = _PHONE.search(line.text)
        if not match or not line.bbox:
            continue
        cx = _center_x(line)
        cy = _center_y(line)
        if cx is None or cy is None:
            continue
        if cy < ly - 8:
            continue
        if abs(cy - ly) > 55:
            continue
        if cx < lx - 30:
            continue
        dist = (cx - lx) ** 2 + (cy - ly) ** 2
        if best is None or dist < best[0]:
            best = (dist, match.group(0))
    return best[1] if best else None


def _phones_for_column(
    section_lines: list[OCRLine],
    midpoint: float,
    column: str,
    y_min: float,
    y_max: float,
) -> list[str]:
    found: list[tuple[float, str]] = []
    for line in section_lines:
        if _column_of(line, midpoint) != column:
            continue
        match = _PHONE.search(line.text)
        if not match or not line.bbox:
            continue
        if not _in_provider_block(line, y_min, y_max):
            continue
        cy = _center_y(line)
        if cy is None:
            continue
        found.append((cy, match.group(0)))
    found.sort(key=lambda item: item[0])
    return list(dict.fromkeys(num for _, num in found))


def _names_for_column(
    section_lines: list[OCRLine],
    midpoint: float,
    column: str,
    y_min: float,
    y_max: float,
) -> list[str]:
    names: list[tuple[float, str]] = []
    for line in section_lines:
        if _column_of(line, midpoint) != column:
            continue
        name = _normalize_name(line.text)
        if not name or name.lower() in _SKIP_NAMES:
            continue
        if not _in_provider_block(line, y_min, y_max):
            continue
        cy = _center_y(line)
        if cy is None:
            continue
        names.append((cy, name))
    names.sort(key=lambda item: item[0])
    return list(dict.fromkeys(n for _, n in names))


def _npis_for_column(
    section_lines: list[OCRLine],
    midpoint: float,
    column: str,
    y_min: float,
    y_max: float,
) -> list[str]:
    npis: list[tuple[float, str]] = []
    for line in section_lines:
        if _column_of(line, midpoint) != column:
            continue
        match = _NPI.search(line.text)
        if not match or not line.bbox:
            continue
        if not _in_provider_block(line, y_min, y_max):
            continue
        cy = _center_y(line)
        if cy is None:
            continue
        npis.append((cy, match.group(1)))
    npis.sort(key=lambda item: item[0])
    return list(dict.fromkeys(n for _, n in npis))


def _parse_spatial(section_lines: list[OCRLine]) -> tuple[ProviderInfo, ProviderInfo] | None:
    midpoint = _column_midpoint(section_lines)
    if midpoint is None:
        return None

    y_min, y_max = _section_iv_y_range(section_lines)
    req_npis = _npis_for_column(section_lines, midpoint, "requesting", y_min, y_max)
    svc_npis = _npis_for_column(section_lines, midpoint, "service", y_min, y_max)

    def _labels_in_column(col: str, pattern: str) -> list[OCRLine]:
        hits = [
            ln
            for ln in section_lines
            if ln.bbox
            and _column_of(ln, midpoint) == col
            and re.fullmatch(pattern, ln.text.strip(), re.I)
            and _in_provider_block(ln, y_min, y_max)
        ]
        hits.sort(key=lambda ln: _center_y(ln) or 0.0)
        return hits

    req_phone_labels = _labels_in_column("requesting", r"Phone:?")
    req_fax_labels = _labels_in_column("requesting", r"Fax:?")
    svc_phone_labels = _labels_in_column("service", r"Phone:?")
    req_phone_label = req_phone_labels[0] if req_phone_labels else None
    req_fax_label = req_fax_labels[0] if req_fax_labels else None
    svc_phone_label = svc_phone_labels[0] if svc_phone_labels else None

    svc_fax_labels = _labels_in_column("service", r"Fax:?")
    contact_label = next(
        (
            ln
            for ln in section_lines
            if ln.bbox
            and _column_of(ln, midpoint) == "requesting"
            and re.search(r"Contact\s+Name", ln.text, re.I)
            and _in_provider_block(ln, y_min, y_max)
        ),
        None,
    )
    pcp_label = next(
        (
            ln
            for ln in section_lines
            if ln.bbox
            and _column_of(ln, midpoint) == "service"
            and re.search(r"Primary\s+Care\s+Provider", ln.text, re.I)
            and _in_provider_block(ln, y_min, y_max)
        ),
        None,
    )

    requesting = ProviderInfo(
        name=_primary_provider_name(section_lines, midpoint, "requesting", y_min),
        npi=req_npis[0] if req_npis else None,
        phone=_phone_near_label(req_phone_label, section_lines, midpoint, "requesting"),
        fax=_phone_near_label(req_fax_label, section_lines, midpoint, "requesting"),
        contact_name=_name_near_label(
            contact_label, section_lines, midpoint, "requesting"
        ),
        contact_phone=_phone_near_label(
            req_phone_labels[1] if len(req_phone_labels) >= 2 else None,
            section_lines,
            midpoint,
            "requesting",
        ),
    )

    service = ProviderInfo(
        name=_primary_provider_name(section_lines, midpoint, "service", y_min),
        npi=svc_npis[0] if svc_npis else None,
        phone=_phone_near_label(svc_phone_label, section_lines, midpoint, "service"),
        fax=_phone_near_label(
            svc_fax_labels[0] if svc_fax_labels else None,
            section_lines,
            midpoint,
            "service",
        ),
        primary_care_provider_name=_name_near_label(
            pcp_label, section_lines, midpoint, "service"
        ),
        primary_care_provider_phone=_phone_near_label(
            svc_phone_labels[1] if len(svc_phone_labels) >= 2 else None,
            section_lines,
            midpoint,
            "service",
        ),
    )
    return requesting, service


def _parse_text_fallback(section_iv: str) -> tuple[ProviderInfo, ProviderInfo]:
    """Fallback when OCR lines lack bboxes — NPI order swap handles RapidOCR column flip."""
    names: list[str] = []
    for line in section_iv.splitlines():
        name = _normalize_name(line.strip())
        if name and name.lower() not in _SKIP_NAMES:
            names.append(name)

    npis = list(dict.fromkeys(_NPI.findall(section_iv)))
    phones = _PHONE.findall(section_iv)

    if len(npis) >= 2:
        req_npi, svc_npi = npis[0], npis[1]
    elif len(npis) == 1:
        req_npi, svc_npi = npis[0], None
    else:
        req_npi, svc_npi = None, None

    req_name = names[0] if names else None
    svc_name = names[1] if len(names) >= 2 else None
    if len(names) >= 2:
        n0_pos = section_iv.find(names[0])
        n1_pos = section_iv.find(names[1], n0_pos + 1)
        if n0_pos != -1 and n1_pos != -1:
            between = section_iv[n0_pos:n1_pos]
            if re.search(r"\bName\s*:", between, re.I):
                svc_name, req_name = names[0], names[1]
    extra = names[2:] if len(names) > 2 else []

    requesting = ProviderInfo(
        name=req_name,
        npi=req_npi,
        phone=phones[2] if len(phones) >= 3 else (phones[0] if phones else None),
        fax=phones[1] if len(phones) >= 2 else None,
        contact_name=extra[1] if len(extra) >= 2 else (extra[0] if extra else None),
        contact_phone=phones[3] if len(phones) >= 4 else None,
    )
    service = ProviderInfo(
        name=svc_name,
        npi=svc_npi,
        phone=phones[0] if phones else None,
        primary_care_provider_name=extra[0] if extra else None,
        primary_care_provider_phone=phones[4] if len(phones) >= 5 else None,
    )
    return requesting, service


def parse_section_iv_providers(
    section_iv: str,
    lines: list[OCRLine],
) -> tuple[ProviderInfo, ProviderInfo]:
    """Return (requesting, service) providers using spatial columns then text fallback."""
    section_lines = _section_iv_lines(lines)
    spatial = _parse_spatial(section_lines)
    if spatial and (spatial[0].name or spatial[1].name):
        return spatial
    return _parse_text_fallback(section_iv)
