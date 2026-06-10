"""Tests for Section V procedure table parsing."""

from src.extract.ocr_engine import OCRLine
from src.extract.table_parser import parse_procedure_table

RAW_SECTION_V = """
Section V — Services Requested
Planned Service or Procedure
Start Date   :
End Date
Code
Aftercare following
Open placement
.44300 11/20/2022 11/26/2022
.247.1...
Needle catheter
Type 1
E10.69
11/29/2023 11/29/2023 11/29/2023
Observation
Inpatient  [ 7 ] Outpatient  [ 7 ]
SECTION VI — CLINICAL DOCUMENTATION
"""


def _lines(text: str) -> list[OCRLine]:
    return [OCRLine(text=line) for line in text.strip().splitlines()]


def test_procedure_rows_exclude_year_as_code():
    rows = parse_procedure_table(_lines(RAW_SECTION_V))
    assert len(rows) == 2

    assert rows[0].code == "44300"
    assert rows[0].icd_code == "Z47.1"
    assert rows[0].planned_service == "Aftercare following Open placement"

    assert rows[1].code is None  # 44015 missing from OCR
    assert rows[1].icd_code == "E10.69"
    assert "2023" not in (rows[1].code or "")
