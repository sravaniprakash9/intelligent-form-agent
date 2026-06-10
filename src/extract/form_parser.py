"""Map extracted document text to canonical form JSON."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from PIL import Image

from src.extract.checkbox_detector import detect_texas_form_fields
from src.extract.document_extractor import DocumentExtractor
from src.extract.ocr_engine import OCRLine
from src.extract.schema import (
    ClinicalSection,
    FormDocument,
    GeneralSection,
    PatientSection,
    ProviderInfo,
    ProviderSection,
    ServicesSection,
    SubmissionSection,
    TherapyInfo,
)
from src.extract.sanitize import (
    clean_prev_auth_number,
    clean_text_field,
    normalize_ocr_text,
    uses_fast_ocr_normalization,
)
from src.extract.field_refiner import refine_low_confidence_fields
from src.extract.member_id_parser import parse_member_and_group_ids
from src.extract.provider_parser import parse_section_iv_providers
from src.extract.table_parser import parse_procedure_table
from src.extract.therapy_parser import (
    infer_therapy_from_session_column,
    parse_therapy_duration,
    parse_therapy_sessions,
)
from src.ingest.loader import load_image


def _parse_date(value: str | None):
    if not value:
        return None
    for fmt in ("%m/%d/%Y", "%m/%d/%y"):
        try:
            return datetime.strptime(value.strip(), fmt).date()
        except ValueError:
            continue
    return None


def _search(pattern: str, text: str, flags: int = re.IGNORECASE) -> str | None:
    match = re.search(pattern, text, flags)
    return match.group(1).strip() if match else None


_SECTION_KEYWORDS: dict[tuple[str, str | None], tuple[str, str]] = {
    ("Section I", "Section II"): (r"submission", r"general\s*information"),
    ("Section II", "Section III"): (r"general\s*information", r"patient\s*information"),
    ("Section III", "Section IV"): (r"patient\s*information", r"provider\s*information"),
    ("Section IV", "Section V"): (
        r"provider\s*information",
        r"services\s*requested|section\s*v\b",
    ),
    ("Section V", "Section VI"): (
        r"services\s*requested|section\s*v\b",
        r"clinical\s*documentation|section\s*vi\b",
    ),
}


def _slice_between_patterns(text: str, start_pat: str, end_pat: str) -> str:
    start_m = re.search(start_pat, text, re.IGNORECASE)
    if not start_m:
        return ""
    start = start_m.end()
    end_m = re.search(end_pat, text[start:], re.IGNORECASE)
    end = start + end_m.start() if end_m else len(text)
    return text[start:end]


def _section_slice(text: str, start_marker: str, end_marker: str | None) -> str:
    key = (start_marker, end_marker)
    if key in _SECTION_KEYWORDS:
        start_pat, end_pat = _SECTION_KEYWORDS[key]
        chunk = _slice_between_patterns(text, start_pat, end_pat)
        if chunk:
            return chunk

    lower = text.lower()
    start = lower.find(start_marker.lower())
    if start == -1:
        return ""
    start = start + len(start_marker)
    if end_marker:
        end = lower.find(end_marker.lower(), start)
        if end != -1:
            return text[start:end]
    return text[start:]


_PERSON_NAME = re.compile(r"^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+$")
_PERSON_NAME_MASHED = re.compile(r"^[A-Z][a-z]+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*$")
_SKIP_NAME_LINES = frozenset(
    {
        "male", "female", "other", "unknown", "phone", "name", "section",
        "observation", "inpatient", "outpatient", "home health", "day surgery",
    }
)


def _normalize_name_candidate(candidate: str) -> str | None:
    if _PERSON_NAME.match(candidate):
        return candidate
    if _PERSON_NAME_MASHED.match(candidate):
        return re.sub(r"([a-z])([A-Z])", r"\1 \2", candidate)
    return None


def _find_person_names(text: str, limit: int = 2) -> list[str]:
    """Find standalone 'First Last' lines (common when labels are on separate OCR lines)."""
    names: list[str] = []
    for line in text.splitlines():
        candidate = line.strip()
        normalized = _normalize_name_candidate(candidate)
        if not normalized:
            continue
        if normalized.lower() in _SKIP_NAME_LINES:
            continue
        names.append(normalized)
        if len(names) >= limit:
            break
    return names


_DATE_IN_TEXT = re.compile(r"\b(\d{1,2}/\d{1,2}/\d{4})\b")
_PHONE_IN_TEXT = re.compile(r"\(\d{2,3}\)-\d{3}-\d{4}")
_HEALTH_PLAN = re.compile(
    r"healthcare|molina|aetna|cigna|humana|anthem|bcbs|united|wellcare|centene",
    re.IGNORECASE,
)


def _try_parse_dob_date(raw: str):
    """Parse a date string as DOB; fix common OCR errors (0/20/1992 → 03/20/1992)."""
    parsed = _parse_date(raw)
    if parsed:
        return parsed
    parts = raw.split("/")
    if len(parts) != 3:
        return None
    month, day, year = parts
    # OCR often drops leading digit of month: 03 → 0
    if month == "0":
        return _parse_date(f"03/{day}/{year}")
    if month == "1" and int(day) > 12:
        return _parse_date(f"01/{day}/{year}")
    return None


def _parse_dob_in_section(text: str):
    """Parse DOB from Section III; label and value are often on separate OCR lines."""
    # Same-line: DOB: 03/20/1992
    match = re.search(r"DOB[:\s]*(\d{1,2}/\d{1,2}/\d{4})", text, re.IGNORECASE)
    if match:
        result = _try_parse_dob_date(match.group(1))
        if result:
            return result

    # Prefer date near patient name (OCR layout: name then DOB on next lines)
    names = _find_person_names(text, limit=1)
    if names:
        anchor = text.find(names[0])
        if anchor != -1:
            window = text[anchor : anchor + 120]
            for m in _DATE_IN_TEXT.finditer(window):
                result = _try_parse_dob_date(m.group(1))
                if result:
                    return result

    # Any plausible birth-year date in section (exclude recent service-like dates)
    for m in _DATE_IN_TEXT.finditer(text):
        result = _try_parse_dob_date(m.group(1))
        if result and result.year < 2010:
            return result
    return None


def _parse_issuer_name(section_i: str) -> str | None:
    for line in section_i.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        lower = stripped.lower()
        if lower.startswith(("phone", "date", "fax", "issuer", "lssuer", "section")):
            continue
        if _HEALTH_PLAN.search(stripped):
            return stripped.split("|")[0].strip()
    return _search(r"(?:Issuer|lssuer)\s*Name[:\s]+(.+?)(?:\n|Phone)", section_i)


def _parse_submission_date(section_i: str):
    for match in _DATE_IN_TEXT.finditer(section_i):
        parsed = _parse_date(match.group(1))
        if parsed and parsed.year >= 2015:
            return parsed
    return None


class FormParser:
    def __init__(self, extractor: DocumentExtractor | None = None) -> None:
        self.extractor = extractor or DocumentExtractor()

    def parse_file(self, form_id: str, source_file: Path) -> FormDocument:
        extracted = self.extractor.extract(source_file)
        image = extracted.image or load_image(source_file)
        return self.parse_content(
            form_id=form_id,
            source_file=source_file.name,
            text=extracted.full_text,
            lines=extracted.lines,
            image=image,
            extraction_method=extracted.method,
        )

    def parse_content(
        self,
        form_id: str,
        source_file: str,
        text: str,
        lines: list[OCRLine],
        image: Image.Image,
        extraction_method: str | None = None,
    ) -> FormDocument:
        if uses_fast_ocr_normalization(extraction_method):
            text = normalize_ocr_text(text)
            lines = [
                OCRLine(text=normalize_ocr_text(ln.text), bbox=ln.bbox, confidence=ln.confidence)
                for ln in lines
            ]

        section_i = _section_slice(text, "Section I", "Section II")
        submission = SubmissionSection(
            issuer_name=_parse_issuer_name(section_i),
            submission_date=_parse_submission_date(section_i),
        )

        tx_fields = detect_texas_form_fields(image, lines, full_text=text)
        review_type = tx_fields.get("review_type")
        request_type = tx_fields.get("request_type")

        general = GeneralSection(
            review_type=review_type,
            request_type=request_type,
            clinical_reason_for_urgency=clean_text_field(
                _search(r"Clinical Reason for Urgency[:\s]+(.+?)(?:\n|Prev)", text)
            ),
            prev_auth_number=clean_prev_auth_number(
                _search(r"Prev\.?\s*Auth\.?\s*#[:\s]+(\S+)", text)
            ),
        )

        crop_reader = (
            self.extractor.crop_reader()
            if getattr(self.extractor, "is_hybrid", False)
            else None
        )

        section_iii = _section_slice(text, "Section III", "Section IV")
        member_id, group_number = parse_member_and_group_ids(
            section_iii, lines, image, crop_reader=crop_reader
        )
        patient_names = _find_person_names(section_iii, limit=1)
        patient = PatientSection(
            name=patient_names[0] if patient_names else _search(
                r"Name[:\s]+([A-Za-z ,.'-]+?)(?:\n|Phone)", section_iii
            ),
            phone=_search(r"(\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})", section_iii),
            dob=_parse_dob_in_section(section_iii),
            member_id=member_id,
            subscriber_name=clean_text_field(
                _search(r"Subscriber Name[:\s]+(.+?)(?:\n|Member)", section_iii)
            ),
            group_number=group_number,
        )
        if tx_fields.get("gender"):
            patient.gender = tx_fields["gender"]  # type: ignore[assignment]

        section_iv = _section_slice(text, "Section IV", "Section V")
        requesting, service = parse_section_iv_providers(section_iv, lines)

        therapies: list[TherapyInfo] = []
        checked_therapy_types = [
            t.strip()
            for t in (tx_fields.get("therapies") or "").split(",")
            if t.strip()
        ]
        inferred_therapy = infer_therapy_from_session_column(lines)
        if inferred_therapy:
            checked_therapy_types = [inferred_therapy]
        therapy_sessions = parse_therapy_sessions(text, lines, image, crop_reader=crop_reader)
        therapy_duration = parse_therapy_duration(text, lines)
        for key in checked_therapy_types:
            therapies.append(
                TherapyInfo(
                    type=key,
                    sessions=therapy_sessions,
                    duration=therapy_duration,
                )
            )

        services = ServicesSection(
            setting=tx_fields.get("setting"),
            therapies=therapies,
            procedures=parse_procedure_table(lines),
        )

        clinical = ClinicalSection(
            address=_search(
                r"Section VI.*?(\d+[^\n]+(?:Texas|TX)\s*\d{5})",
                text,
                re.DOTALL,
            )
            or _search(r"(\d+[^\n]+(?:Texas|TX)\s*\d{5})", text),
        )

        confidence = min(1.0, 0.5 + 0.05 * sum(1 for v in [
            submission.submission_date,
            submission.issuer_name,
            patient.name,
            patient.dob,
            patient.gender,
            patient.group_number,
            general.review_type,
            general.request_type,
            requesting.npi,
            services.setting,
            services.procedures,
            services.therapies,
        ] if v))

        doc = FormDocument(
            form_id=form_id,
            source_file=source_file,
            section_i_submission=submission,
            section_ii_general=general,
            section_iii_patient=patient,
            section_iv_providers=ProviderSection(requesting=requesting, service=service),
            section_v_services=services,
            section_vi_clinical=clinical,
            extraction_confidence=confidence,
            extraction_method=extraction_method,
            raw_text=text,
        )

        if crop_reader is not None:
            doc, _refined = refine_low_confidence_fields(doc, image, lines, crop_reader)
            confidence = min(1.0, 0.5 + 0.05 * sum(1 for v in [
                submission.submission_date,
                submission.issuer_name,
                patient.name,
                patient.dob,
                patient.gender,
                patient.member_id,
                patient.group_number,
                general.review_type,
                general.request_type,
                requesting.npi,
                services.setting,
                services.procedures,
                services.therapies,
            ] if v))
            doc.extraction_confidence = confidence

        return doc
