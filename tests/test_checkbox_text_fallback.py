"""Text fallbacks for RapidOCR checkbox artifacts."""

from src.extract.checkbox_detector import (
    _section_v_text,
    _text_review_type_from_layout,
    _text_setting_fallback,
    _text_therapies_fallback,
)

SECTION_V_SNIPPET = """
Inpatient ] Outpatient[
Physical Therapy Occupational Therapy
Number of Sessions:
2 weeks
4
Duration:
"""

SECTION_II_SNIPPET = """
Non-Urgent
Clinical Reason for Urgency:
Urgent
Review Type:
Initial Request
"""


def test_outpatient_bracket_pattern():
    assert _text_setting_fallback(SECTION_V_SNIPPET) == "outpatient"


def test_physical_therapy_with_sessions():
    section_v = _section_v_text("SECTION V\n" + SECTION_V_SNIPPET + "\nSECTION VI")
    assert _text_therapies_fallback(section_v) == "physical_therapy"


def test_non_urgent_from_clinical_reason_layout():
    assert _text_review_type_from_layout(SECTION_II_SNIPPET) == "non_urgent"
