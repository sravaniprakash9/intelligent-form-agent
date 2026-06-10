"""Tests for therapy session extraction."""

from PIL import Image

from src.extract.ocr_engine import OCRLine
from src.extract.therapy_parser import parse_therapy_sessions

RAW = """
SECTION V — Services Requested
[ 2 Physical Therapy
2 weeks
Number of Sessions:
Duration:
SECTION VI
"""


def test_does_not_use_checkbox_artifact_as_sessions():
    lines = [OCRLine(text=line) for line in RAW.strip().splitlines()]
    assert parse_therapy_sessions(RAW, lines, image=None) is None


def test_reads_inline_sessions_label():
    text = "SECTION V\nNumber of Sessions: 4\nSECTION VI"
    lines = [OCRLine(text="Number of Sessions: 4", bbox=(100, 200, 300, 220))]
    assert parse_therapy_sessions(text, lines, image=None) == 4


def test_pick_session_count_ignores_duration_weeks():
    from src.extract.therapy_parser import _pick_session_count

    assert _pick_session_count([4, 2], duration_weeks=2) == 4


def test_reads_digit_on_therapy_row_bbox():
    text = "SECTION V\nPhysical Therapy\nNumber of Sessions:\nSECTION VI"
    lines = [
        OCRLine(text="Physical Therapy", bbox=(200, 400, 350, 420)),
        OCRLine(text="4", bbox=(450, 398, 470, 422)),
        OCRLine(text="Number of Sessions:", bbox=(100, 450, 280, 470)),
    ]
    assert parse_therapy_sessions(text, lines, image=None) == 4
