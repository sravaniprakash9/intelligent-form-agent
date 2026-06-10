"""Shared helpers for building LLM prompts from structured form data."""

from __future__ import annotations

import json

from src.extract.schema import FormDocument


def key_fields_summary(doc: FormDocument) -> dict:
    """Compact summary of authoritative structured fields for LLM context."""
    return {
        "issuer": doc.section_i_submission.issuer_name,
        "submission_date": doc.section_i_submission.submission_date,
        "review_type": doc.section_ii_general.review_type,
        "request_type": doc.section_ii_general.request_type,
        "patient": {
            "name": doc.section_iii_patient.name,
            "dob": doc.section_iii_patient.dob,
            "gender": doc.section_iii_patient.gender,
            "member_id": doc.section_iii_patient.member_id,
            "group_number": doc.section_iii_patient.group_number,
        },
        "requesting_provider": doc.section_iv_providers.requesting.name,
        "service_provider": doc.section_iv_providers.service.name,
        "setting": doc.section_v_services.setting,
        "therapies": [
            {"type": t.type, "sessions": t.sessions, "duration": t.duration}
            for t in doc.section_v_services.therapies
        ],
        "procedures": [
            {
                "service": p.planned_service,
                "code": p.code,
                "icd": p.icd_code,
                "dates": f"{p.start_date} to {p.end_date}",
            }
            for p in doc.section_v_services.procedures
        ],
        "clinical_address": doc.section_vi_clinical.address,
    }


def form_context_for_llm(doc: FormDocument) -> str:
    """Structured JSON for LLM prompts — excludes noisy raw OCR text."""
    summary = key_fields_summary(doc)
    structured = doc.model_dump_json(indent=2, exclude={"raw_text"})
    return (
        f"Key fields summary (authoritative — prefer over context chunks):\n"
        f"{json.dumps(summary, indent=2, default=str)}\n\n"
        f"Structured form JSON:\n{structured}"
    )
