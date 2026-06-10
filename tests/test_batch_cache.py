"""Tests for batch extraction cache matching."""

from src.pipeline.batch import _extraction_cache_matches


def test_hybrid_cache_matches_fast_engine_prefix():
    assert _extraction_cache_matches("hybrid:rapidocr+surya-crops(member_id)", "hybrid", "rapidocr")
    assert not _extraction_cache_matches("hybrid:rapidocr", "hybrid", "tesseract")


def test_full_cache_matches_engine():
    assert _extraction_cache_matches("surya", "full", "surya")
    assert not _extraction_cache_matches("rapidocr", "full", "surya")
