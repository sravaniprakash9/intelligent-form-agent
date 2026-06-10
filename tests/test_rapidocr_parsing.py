"""Regression tests for RapidOCR / hybrid text normalization."""

from pathlib import Path

from src.extract.form_parser import FormParser
from src.extract.ocr_engine import OCRLine
from src.extract.sanitize import normalize_ocr_text, uses_fast_ocr_normalization
from src.ingest.loader import load_image

RAPIDOCR_RAW = """
Texas68230
HS
TEXASSTANDARDPRIORAUTHORIZATIONREQUESTFORMFORHEALTHCARESERVICES
SECTIONI-SUBMISSION
Molina Healthcare of Texas
11/20/2022
SECTIONII-GENERALINFORMATION
Non-Urgent
Clinical Reason for Urgency:
Urgent
Review Type:
Initial Request
SECTIONIIPATIENTINFORMATION
Daniel Jarvis
(378)-041-2101
03/20/1992
Unknown
62106
SECTIONIV-PROVIDERINFORMATION
ElizabethFoley
Leslie Johnson
2990955730
3659236757
SECTIONV-SERVICESREQUESTED
4430011/20/202211/26/2022
Aftercare following
Openplacement
Z47.1.
44015
Needlecatheter
Type 1
E10.69
11/29/202311/29/2023
Inpatient ] Outpatient[
Physical Therapy Occupational Therapy
Number of Sessions:
2 weeks
4
SECTIONVI-CLINICALDOCUMENTATION
411TurnerViaductApt.Texas77641
"""


def test_hybrid_method_triggers_normalization():
    assert uses_fast_ocr_normalization("hybrid:rapidocr+surya-crops(prev_auth_number)")


def test_normalize_splits_section_and_names():
    text = normalize_ocr_text(RAPIDOCR_RAW)
    assert "PATIENT INFORMATION" in text.upper() or "patient information" in text.lower()
    assert "Elizabeth Foley" in text


def test_parse_rapidocr_fixture_fields():
    image_path = Path("data/raw/0a01f77b-85f9-48c9-89bf-4b095bebb438_TX_page_1.png")
    if not image_path.exists():
        return
    image = load_image(image_path)
    parser = FormParser()
    doc = parser.parse_content(
        form_id="test",
        source_file="test.png",
        text=RAPIDOCR_RAW,
        lines=[OCRLine(text=ln) for ln in RAPIDOCR_RAW.strip().splitlines()],
        image=image,
        extraction_method="hybrid:rapidocr",
    )
    assert doc.section_iii_patient.name == "Daniel Jarvis"
    assert doc.section_iii_patient.member_id == "62106"
    assert doc.section_iii_patient.dob is not None
    assert doc.section_iv_providers.requesting.name == "Elizabeth Foley"
    assert doc.section_iv_providers.service.name == "Leslie Johnson"
    assert doc.section_iv_providers.requesting.npi == "2990955730"
    assert doc.section_ii_general.review_type == "non_urgent"
    assert doc.section_v_services.setting == "outpatient"
    assert len(doc.section_v_services.therapies) >= 1
    assert doc.section_v_services.therapies[0].type == "physical_therapy"
    assert doc.section_v_services.therapies[0].sessions == 4
    assert doc.section_v_services.therapies[0].duration == "2 weeks"
    codes = [p.code for p in doc.section_v_services.procedures if p.code]
    assert "44300" in codes or "44015" in codes
