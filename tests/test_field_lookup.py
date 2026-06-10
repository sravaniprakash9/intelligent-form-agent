"""Tests for direct field lookup — no LLM required."""

from datetime import date

from src.agent.field_lookup import lookup_field
from src.extract.schema import (
    FormDocument,
    GeneralSection,
    PatientSection,
    ProcedureRow,
    ServicesSection,
    TherapyInfo,
)


def _sample_doc() -> FormDocument:
    return FormDocument(
        form_id="test_form",
        source_file="test.png",
        section_ii_general=GeneralSection(review_type="non_urgent", request_type="initial"),
        section_iii_patient=PatientSection(
            name="Daniel Jarvis",
            dob=date(1992, 3, 20),
            gender="unknown",
            group_number="62106",
        ),
        section_v_services=ServicesSection(
            setting="outpatient",
            therapies=[TherapyInfo(type="physical_therapy", sessions=4, duration="2 weeks")],
            procedures=[
                ProcedureRow(
                    planned_service="Open placement",
                    code="44300",
                    start_date=date(2022, 11, 20),
                    end_date=date(2022, 11, 26),
                    icd_code="Z47.1",
                ),
            ],
        ),
        raw_text="Inpatient [ 7 ] Outpatient [ 7 ]",
    )


def test_inpatient_outpatient_question_uses_structured_setting():
    answer = lookup_field(_sample_doc(), "is he inpatient or outpatient")
    assert answer is not None
    assert "outpatient" in answer.lower()
    assert "inpatient" not in answer.lower() or "outpatient" in answer.lower()


def test_gender_lookup():
    answer = lookup_field(_sample_doc(), "what is the patient gender")
    assert answer is not None
    assert "unknown" in answer.lower()


def test_review_type_lookup():
    answer = lookup_field(_sample_doc(), "is this urgent")
    assert answer is not None
    assert "non urgent" in answer.lower()


def test_therapy_sessions_lookup():
    answer = lookup_field(_sample_doc(), "how many physical therapy sessions")
    assert answer is not None
    assert "4" in answer


def test_therapy_kind_lookup():
    answer = lookup_field(_sample_doc(), "what kind of therapy did the patient go through")
    assert answer is not None
    assert "physical" in answer.lower()
    assert "4" in answer
