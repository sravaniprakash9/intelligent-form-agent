"""Generate concise form summaries."""

from __future__ import annotations

import re
from datetime import date

from src.agent.llm_client import OllamaClient
from src.agent.prompt_utils import form_context_for_llm
from src.extract.schema import FormDocument, ProcedureRow, TherapyInfo

SUMMARY_SYSTEM = (
    "You summarize Texas prior authorization forms concisely. "
    "Use bullet points covering: patient, urgency, request type, "
    "providers, services/procedures, therapy details, and key dates. "
    "Prefer structured JSON fields over any OCR noise. Keep it under 200 words."
)

_LABEL_MAP = {
    "non_urgent": "Non-Urgent",
    "urgent": "Urgent",
    "initial": "Initial Request",
    "extension_renewal_amendment": "Extension / Renewal / Amendment",
    "physical_therapy": "Physical Therapy",
    "occupational_therapy": "Occupational Therapy",
    "speech_therapy": "Speech Therapy",
    "cardiac_rehab": "Cardiac Rehab",
    "mental_health_substance_abuse": "Mental Health / Substance Abuse",
    "inpatient": "Inpatient",
    "outpatient": "Outpatient",
    "provider_office": "Provider Office",
    "home": "Home",
    "unknown": "Unknown",
    "male": "Male",
    "female": "Female",
    "other": "Other",
}


def _humanize(value: object | None) -> str:
    if value is None:
        return "—"
    text = str(value).strip()
    if not text:
        return "—"
    return _LABEL_MAP.get(text, text.replace("_", " ").title())


def _fmt_date(value: date | None) -> str:
    if value is None:
        return "—"
    return value.strftime("%b %d, %Y")


def _date_range(start: date | None, end: date | None) -> str:
    if start and end:
        if start == end:
            return _fmt_date(start)
        return f"{_fmt_date(start)} → {_fmt_date(end)}"
    if start:
        return _fmt_date(start)
    if end:
        return _fmt_date(end)
    return "—"


def _procedure_title(proc: ProcedureRow) -> str:
    name = (proc.planned_service or "").strip()
    code = (proc.code or "").strip()
    if name and not re.fullmatch(r"[\d\s/\-]+", name) and code not in name:
        return name
    if code:
        return f"Procedure {code}"
    return name or "Unnamed procedure"


def _format_procedure(proc: ProcedureRow) -> str:
    title = _procedure_title(proc)
    parts = [_date_range(proc.start_date, proc.end_date)]
    if proc.code:
        parts.append(f"Code **{proc.code}**")
    if proc.icd_code:
        parts.append(f"ICD **{proc.icd_code}**")
    if proc.diagnosis_description and proc.diagnosis_description not in title:
        parts.append(proc.diagnosis_description)
    detail = " · ".join(parts)
    return f"- **{title}**  \n  {detail}"


def _format_therapy(t: TherapyInfo) -> str:
    parts = [_humanize(t.type)]
    if t.sessions is not None:
        parts.append(f"{t.sessions} session{'s' if t.sessions != 1 else ''}")
    if t.duration:
        parts.append(f"Duration: {t.duration}")
    if t.frequency:
        parts.append(f"Frequency: {t.frequency}")
    return "- " + " · ".join(parts)


class FormSummarizer:
    def __init__(self, llm: OllamaClient | None = None) -> None:
        self.llm = llm or OllamaClient()

    def summarize(self, doc: FormDocument) -> str:
        prompt = f"""Summarize this prior authorization form:

{form_context_for_llm(doc)}
"""
        return self.llm.generate(prompt, system=SUMMARY_SYSTEM)

    def summarize_structured(self, doc: FormDocument) -> str:
        """Formatted markdown summary without LLM."""
        p = doc.section_iii_patient
        g = doc.section_ii_general
        sub = doc.section_i_submission
        rp = doc.section_iv_providers.requesting
        sp = doc.section_iv_providers.service
        svc = doc.section_v_services

        sections: list[str] = []

        sections.append(
            "### Patient\n"
            f"- **Name:** {p.name or '—'}  \n"
            f"- **Member ID:** {p.member_id or '—'}  \n"
            f"- **DOB:** {_fmt_date(p.dob)}  \n"
            f"- **Gender:** {_humanize(p.gender)}  \n"
            f"- **Phone:** {p.phone or '—'}"
        )

        sections.append(
            "### Request\n"
            f"- **Review type:** {_humanize(g.review_type)}  \n"
            f"- **Request type:** {_humanize(g.request_type)}  \n"
            f"- **Submission date:** {_fmt_date(sub.submission_date)}  \n"
            f"- **Issuer:** {sub.issuer_name or '—'}"
        )

        req_lines = [
            f"- **Requesting:** {rp.name or '—'}"
            + (f" · NPI {rp.npi}" if rp.npi else ""),
            f"  - Phone: {rp.phone or '—'} · Fax: {rp.fax or '—'}",
        ]
        if rp.contact_name:
            req_lines.append(
                f"  - Contact: {rp.contact_name}"
                + (f" · {rp.contact_phone}" if rp.contact_phone else "")
            )
        svc_lines = [
            f"- **Service:** {sp.name or '—'}"
            + (f" · NPI {sp.npi}" if sp.npi else ""),
            f"  - Phone: {sp.phone or '—'} · Fax: {sp.fax or '—'}",
        ]
        if sp.primary_care_provider_name:
            svc_lines.append(
                f"  - Primary care: {sp.primary_care_provider_name}"
                + (f" · {sp.primary_care_provider_phone}" if sp.primary_care_provider_phone else "")
            )
        sections.append("### Providers\n" + "  \n".join(req_lines + svc_lines))

        service_lines = [f"- **Setting:** {_humanize(svc.setting)}"]
        if svc.procedures:
            service_lines.append("")
            service_lines.append("**Procedures**")
            service_lines.extend(_format_procedure(proc) for proc in svc.procedures)
        if svc.therapies:
            service_lines.append("")
            service_lines.append("**Therapy**")
            service_lines.extend(_format_therapy(t) for t in svc.therapies)
        sections.append("### Services\n" + "\n".join(service_lines))

        if doc.section_vi_clinical.address:
            sections.append(
                "### Clinical\n"
                f"- **Service address:** {doc.section_vi_clinical.address}"
            )

        return "\n\n---\n\n".join(sections)
