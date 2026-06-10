"""Tests for PDF text sufficiency heuristic."""

from src.extract.pdf_extractor import is_text_sufficient


def test_text_sufficient_with_form_markers():
    text = "Section III Patient Information\nName: Jane Doe\nMember or Medicaid ID"
    assert is_text_sufficient(text, min_chars=20) is True


def test_text_insufficient_when_too_short():
    assert is_text_sufficient("abc", min_chars=100) is False


def test_text_insufficient_without_markers():
    text = "x" * 200
    assert is_text_sufficient(text, min_chars=100) is False
