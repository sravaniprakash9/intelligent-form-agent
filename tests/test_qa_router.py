"""Tests for query router."""

from src.agent.router import classify_query


def test_classify_lookup():
    assert classify_query("What is the member ID?") == "lookup"


def test_classify_aggregate():
    assert classify_query("How many forms are urgent?", multi_form=True) == "aggregate"


def test_classify_semantic():
    assert classify_query("Explain the clinical reasoning") == "semantic"


def test_classify_setting_lookup():
    assert classify_query("is he inpatient or outpatient") == "lookup"
    assert classify_query("what is the service setting") == "lookup"
