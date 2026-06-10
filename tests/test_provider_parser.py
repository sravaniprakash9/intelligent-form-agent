"""Section IV provider column assignment."""

from pathlib import Path

from src.extract.document_extractor import DocumentExtractor
from src.extract.form_parser import _section_slice
from src.extract.provider_parser import parse_section_iv_providers
from src.extract.sanitize import normalize_ocr_text
from src.extract.ocr_engine import OCRLine

BOBBY_FORM = "0a9d92f7-b19a-40b5-8c48-1546b7e453ab_TX_page_2"
DANIEL_FORM = "0a01f77b-85f9-48c9-89bf-4b095bebb438_TX_page_1"


def _parse_form(filename: str):
    path = Path(f"data/raw/{filename}.png")
    if not path.exists():
        return None
    extracted = DocumentExtractor(ocr_mode="hybrid", fast_engine="rapidocr").extract(path)
    text = normalize_ocr_text(extracted.full_text)
    lines = [
        OCRLine(text=normalize_ocr_text(ln.text), bbox=ln.bbox, confidence=ln.confidence)
        for ln in extracted.lines
    ]
    section_iv = _section_slice(text, "Section IV", "Section V")
    return parse_section_iv_providers(section_iv, lines)


def test_bobby_juarez_provider_columns():
    result = _parse_form(BOBBY_FORM)
    if result is None:
        return
    requesting, service = result
    assert requesting.name == "Philip Mitchell"
    assert requesting.npi == "246387268"
    assert requesting.phone == "(527)-715-3405"
    assert requesting.fax == "(171)-587-8767"
    assert requesting.contact_name == "Amanda Mitchell"
    assert requesting.contact_phone == "(418)-275-1812"
    assert service.name == "Taylor Dickson"
    assert service.npi == "2266362534"
    assert service.phone == "(183)-461-0581"
    assert service.primary_care_provider_name == "Jamie English"
    assert service.primary_care_provider_phone == "(767)-907-6678"


def test_daniel_jarvis_provider_columns():
    result = _parse_form(DANIEL_FORM)
    if result is None:
        return
    requesting, service = result
    assert requesting.name == "Elizabeth Foley"
    assert requesting.npi == "2990955730"
    assert service.name == "Leslie Johnson"
    assert service.npi == "3659236757"
