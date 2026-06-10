"""Tests for structured summarizer (no LLM)."""

from datetime import date

from src.agent.summarizer import FormSummarizer
from src.extract.schema import (
    FormDocument,
    GeneralSection,
    PatientSection,
    ProcedureRow,
    ServicesSection,
    SubmissionSection,
)


def test_structured_summary():
    doc = FormDocument(
        form_id="test_001",
        source_file="test.png",
        section_i_submission=SubmissionSection(submission_date=date(2023, 6, 2)),
        section_ii_general=GeneralSection(review_type="non_urgent", request_type="initial"),
        section_iii_patient=PatientSection(name="Jane Doe", member_id="12345"),
        section_v_services=ServicesSection(
            setting="outpatient",
            procedures=[
                ProcedureRow(
                    planned_service="Test procedure",
                    code="12345",
                    start_date=date(2023, 6, 2),
                    icd_code="Z00.0",
                )
            ],
        ),
    )
    summary = FormSummarizer().summarize_structured(doc)
    assert "### Patient" in summary
    assert "Jane Doe" in summary
    assert "12345" in summary
    assert "### Services" in summary
    assert "Test procedure" in summary
    assert "Non-Urgent" in summary
    assert "Outpatient" in summary
