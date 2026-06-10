"""Tests for OCR field sanitization."""

from src.extract.sanitize import clean_prev_auth_number, clean_text_field, normalize_ocr_text


def test_clean_text_field_drops_pipe_artifacts():
    assert clean_text_field("| Urgent") is None


def test_clean_text_field_drops_subscriber_placeholder():
    assert clean_text_field("(if different):") is None


def test_clean_prev_auth_rejects_section_header():
    assert clean_prev_auth_number("SECTION") is None


def test_clean_prev_auth_keeps_real_number():
    assert clean_prev_auth_number("PA-12345") == "PA-12345"


def test_normalize_ocr_text_fixes_section_typo():
    assert "Section II" in normalize_ocr_text("Sectlon II General")
