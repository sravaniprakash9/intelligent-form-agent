"""Tests for Member/Medicaid ID and Group # parsing."""

from src.extract.member_id_parser import parse_member_and_group_ids
from src.extract.ocr_engine import OCRLine

# RapidOCR bboxes from 0a01f77b form: 62106 sits under Member ID, Group # is blank.
_MEMBER_LABEL = OCRLine(
    text="MemberorMedicaid ID#：",
    bbox=(957.74, 1206.22, 1366.19, 1246.90),
)
_GROUP_LABEL = OCRLine(text="Group#:", bbox=(1626.75, 1206.22, 1771.12, 1248.67))
_MEMBER_VALUE = OCRLine(text="62106", bbox=(1153.16, 1266.35, 1272.88, 1308.80))

_SECTION_III_TEXT = """
Name:
Daniel Jarvis
Subscriber Name (if different):
Member or Medicaid ID #:
Group #:
62106
SECTION IV — PROVIDER INFORMATION
"""


def test_spatial_assigns_digit_to_member_column():
    lines = [
        OCRLine(text="SECTION III — PATIENT INFORMATION"),
        OCRLine(text="Subscriber Name (if different):", bbox=(297.53, 1206.22, 760.56, 1246.90)),
        _MEMBER_LABEL,
        _GROUP_LABEL,
        _MEMBER_VALUE,
        OCRLine(text="SECTION IV — PROVIDER INFORMATION"),
    ]
    member_id, group_number = parse_member_and_group_ids(_SECTION_III_TEXT, lines)
    assert member_id == "62106"
    assert group_number is None


def test_inline_member_and_group_on_same_line():
    text = "Member or Medicaid ID #: 12345678\nGroup #: 99999\n"
    member_id, group_number = parse_member_and_group_ids(text, [])
    assert member_id == "12345678"
    assert group_number == "99999"


def test_spatial_assigns_digit_to_group_column():
    group_value = OCRLine(text="85735", bbox=(1680.0, 1266.0, 1760.0, 1308.0))
    lines = [_MEMBER_LABEL, _GROUP_LABEL, group_value]
    text = "Member or Medicaid ID #:\nGroup #:\n85735\n"
    member_id, group_number = parse_member_and_group_ids(text, lines)
    assert member_id is None
    assert group_number == "85735"
