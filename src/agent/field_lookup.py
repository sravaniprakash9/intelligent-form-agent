"""Direct field lookup from structured form JSON — bypasses LLM for known fields."""

from __future__ import annotations

import re

from src.extract.schema import FormDocument


def _fmt(value: object) -> str:
    if value is None:
        return "not specified"
    return str(value).replace("_", " ")


def lookup_field(doc: FormDocument, question: str) -> str | None:
    """Return a direct answer when the question maps to a structured field."""
    q = question.lower()
    p = doc.section_iii_patient
    g = doc.section_ii_general
    prov = doc.section_iv_providers
    svc = doc.section_v_services

    if re.search(r"\b(group\s*#?|group number)\b", q):
        return f"Group number: {_fmt(p.group_number)} (Section III)"
    # Setting before gender — "is he inpatient or outpatient" must not match gender.
    if re.search(r"\binpatient\b|\boutpatient\b|service setting|\bsetting\b", q):
        setting = svc.setting
        if setting:
            return f"Service setting: {_fmt(setting)} (Section V — Services Requested)"
        return "Service setting: not specified (Section V)"
    if re.search(r"\b(gender|sex)\b", q) or re.search(
        r"\b(is he|is she|what is (his|her) gender)\b", q
    ):
        return f"Gender: {_fmt(p.gender)} (Section III)"
    if "phone" in q and "patient" in q:
        return f"Patient phone: {_fmt(p.phone)} (Section III)"
    if "member" in q or "medicaid" in q:
        return f"Member ID: {_fmt(p.member_id)} (Section III — Patient Information)"
    if re.search(r"patient.*name|who is the patient|patient's name", q):
        return f"Patient name: {_fmt(p.name)} (Section III)"
    if re.search(r"\bdob\b|date of birth|birth date|born\b", q):
        return f"Date of birth: {_fmt(p.dob)} (Section III)"

    if re.search(r"\bnon[- ]?urgent\b", q):
        return f"Review type: {_fmt(g.review_type)} (Section II)"
    if re.search(r"\burgent\b", q) and "non" not in q:
        return f"Review type: {_fmt(g.review_type)} (Section II)"
    if re.search(r"request type|initial request|extension|renewal|amendment", q):
        return f"Request type: {_fmt(g.request_type)} (Section II)"
    if re.search(r"prev\.?\s*auth|authorization number", q):
        return f"Previous authorization #: {_fmt(g.prev_auth_number)} (Section II)"

    if re.search(r"\bissuer\b|insurance company|health plan|molina", q):
        return f"Issuer: {_fmt(doc.section_i_submission.issuer_name)} (Section I)"
    if re.search(r"submission date|form date|submitted on|date submitted", q):
        return f"Submission date: {_fmt(doc.section_i_submission.submission_date)} (Section I)"

    if re.search(r"requesting.*npi|npi.*requesting", q):
        return f"Requesting provider NPI: {_fmt(prov.requesting.npi)} (Section IV)"
    if re.search(r"service.*npi|npi.*service", q):
        return f"Service provider NPI: {_fmt(prov.service.npi)} (Section IV)"
    if re.search(r"requesting provider|requesting physician|requesting doctor", q):
        return f"Requesting provider: {_fmt(prov.requesting.name)} (Section IV)"
    if re.search(r"service provider|servicing provider|facility provider", q):
        return f"Service provider: {_fmt(prov.service.name)} (Section IV)"
    if re.search(r"primary care|pcp\b", q):
        return (
            f"Primary care provider: {_fmt(prov.service.primary_care_provider_name)} "
            f"(Section IV)"
        )
    if re.search(r"contact name", q):
        return f"Contact name: {_fmt(prov.requesting.contact_name)} (Section IV)"

    if re.search(
        r"what kind of therapy|which therapy|type of therapy|what therapy|"
        r"therapy did|went through.*therapy|therapy.*went through|"
        r"therapy.*patient|patient.*therapy|kind of therapy",
        q,
    ):
        if svc.therapies:
            parts = [
                f"{_fmt(t.type)} ({_fmt(t.sessions)} sessions, {_fmt(t.duration)})"
                for t in svc.therapies
            ]
            return f"Therapy: {', '.join(parts)} (Section V)"
        return "Therapy: none specified (Section V)"
    if re.search(r"\b(sessions|how many|number of)\b", q) and re.search(
        r"therapy|physical|occupational|speech", q
    ):
        if svc.therapies:
            t = svc.therapies[0]
            return f"Therapy sessions: {_fmt(t.sessions)} ({t.type}, Section V)"
        return "Therapy sessions: not specified (Section V)"
    if re.search(r"\b(duration|how long)\b", q) and re.search(
        r"therapy|physical|occupational|speech", q
    ):
        if svc.therapies:
            t = svc.therapies[0]
            return f"Therapy duration: {_fmt(t.duration)} ({t.type}, Section V)"
        return "Therapy duration: not specified (Section V)"

    if re.search(r"\bicd\b|diagnosis code", q):
        codes = [proc.icd_code for proc in svc.procedures if proc.icd_code]
        if codes:
            return f"ICD code(s): {', '.join(codes)} (Section V)"
        return "ICD code: not specified (Section V)"
    if re.search(r"\bcpt\b|procedure code|hcpcs|service code", q):
        codes = [proc.code for proc in svc.procedures if proc.code]
        if codes:
            return f"Procedure code(s): {', '.join(codes)} (Section V)"
        return "Procedure code: not specified (Section V)"
    if re.search(r"\bprocedures?\b", q) and not re.search(r"why|explain|describe", q):
        if svc.procedures:
            parts = [
                f"{proc.planned_service or 'unknown'} (code {proc.code}, ICD {proc.icd_code})"
                for proc in svc.procedures
            ]
            return "Procedures (Section V): " + "; ".join(parts)
        return "Procedures: none listed (Section V)"

    if re.search(r"clinical address|service address|facility address|where.*service", q):
        return f"Clinical address: {_fmt(doc.section_vi_clinical.address)} (Section VI)"
    if "phone" in q and re.search(r"requesting|provider", q):
        return f"Requesting provider phone: {_fmt(prov.requesting.phone)} (Section IV)"
    if "phone" in q and re.search(r"service|facility", q):
        return f"Service provider phone: {_fmt(prov.service.phone)} (Section IV)"
    if re.search(r"\bphone\b", q) and not re.search(r"patient|member", q):
        return (
            f"Patient phone: {_fmt(p.phone)} (Section III); "
            f"Requesting provider: {_fmt(prov.requesting.phone)} (Section IV); "
            f"Service provider: {_fmt(prov.service.phone)} (Section IV)"
        )

    return None
