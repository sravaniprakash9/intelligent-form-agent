"""Section V services/therapy on Bobby Juarez sample (page 2)."""

from pathlib import Path

from src.extract.document_extractor import DocumentExtractor
from src.extract.form_parser import FormParser
from src.extract.sanitize import normalize_ocr_text
from src.extract.therapy_parser import infer_therapy_from_session_column, parse_therapy_duration
from src.ingest.loader import load_image

BOBBY_FORM = "0a9d92f7-b19a-40b5-8c48-1546b7e453ab_TX_page_2"


def _extract_bobby():
    path = Path(f"data/raw/{BOBBY_FORM}.png")
    if not path.exists():
        return None
    extractor = DocumentExtractor(ocr_mode="hybrid", fast_engine="rapidocr")
    extracted = extractor.extract(path)
    image = load_image(path)
    text = normalize_ocr_text(extracted.full_text)
    return extracted, image, text


def test_bobby_procedure_rows():
    data = _extract_bobby()
    if data is None:
        return
    extracted, image, text = data
    parser = FormParser(extractor=DocumentExtractor(ocr_mode="hybrid", fast_engine="rapidocr"))
    doc = parser.parse_content(
        BOBBY_FORM, "test.png", text, extracted.lines, image, extracted.method
    )
    codes = {p.code: p for p in doc.section_v_services.procedures}
    assert "69631" in codes
    assert "93624" in codes
    assert "Othermotorcycle" not in (codes["69631"].planned_service or "").lower()
    assert "tympanoplasty" in (codes["69631"].planned_service or "").lower()
    assert codes["69631"].icd_code == "V22.19XA"
    assert "electrophysiologic" in (codes["93624"].planned_service or "").lower()


def test_bobby_occupational_therapy_and_duration():
    data = _extract_bobby()
    if data is None:
        return
    extracted, _image, text = data
    assert infer_therapy_from_session_column(extracted.lines) == "occupational_therapy"
    assert parse_therapy_duration(text, extracted.lines) == "1 week"

    parser = FormParser(extractor=DocumentExtractor(ocr_mode="hybrid", fast_engine="rapidocr"))
    doc = parser.parse_content(
        BOBBY_FORM, "test.png", text, extracted.lines, _image, extracted.method
    )
    assert len(doc.section_v_services.therapies) == 1
    therapy = doc.section_v_services.therapies[0]
    assert therapy.type == "occupational_therapy"
    assert therapy.sessions == 3
    assert therapy.duration == "1 week"
