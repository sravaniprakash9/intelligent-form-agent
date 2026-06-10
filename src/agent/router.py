"""Route queries to structured lookup, vector search, or aggregation."""

from __future__ import annotations

import re


AGGREGATE_PATTERNS = [
    r"\bhow many\b",
    r"\bcount\b",
    r"\bacross\b",
    r"\ball forms\b",
    r"\bcompare\b",
    r"\bmost common\b",
    r"\bdistribution\b",
    r"\bpercentage\b",
]

FIELD_PATTERNS = {
    "member_id": [r"member id", r"medicaid id", r"member or medicaid"],
    "patient_name": [r"patient name", r"patient's name", r"who is the patient"],
    "npi": [r"\bnpi\b", r"national provider"],
    "dob": [r"\bdob\b", r"date of birth", r"birth date"],
    "gender": [r"\bgender\b", r"\bsex\b", r"\bmale\b", r"\bfemale\b"],
    "group_number": [r"group\s*#", r"group number"],
    "review_type": [r"non[- ]?urgent", r"\burgent\b", r"review type"],
    "request_type": [r"request type", r"initial request", r"extension", r"renewal"],
    "requesting_provider": [r"requesting provider"],
    "service_provider": [r"service provider"],
    "submission_date": [r"submission date", r"form date"],
    "issuer": [r"\bissuer\b", r"insurance company", r"health plan"],
    "setting": [r"\binpatient\b", r"\boutpatient\b", r"service setting"],
    "therapy": [
        r"physical therapy",
        r"occupational therapy",
        r"speech therapy",
        r"\btherapy\b",
        r"what kind of therapy",
        r"number of.*session",
        r"therapy session",
    ],
    "procedure": [r"\bprocedure\b", r"\bcpt\b", r"procedure code", r"\bicd\b"],
    "clinical_address": [r"clinical address", r"facility address", r"service address"],
}


def classify_query(question: str, multi_form: bool = False) -> str:
    q = question.lower()

    if multi_form or any(re.search(p, q) for p in AGGREGATE_PATTERNS):
        return "aggregate"

    for field, patterns in FIELD_PATTERNS.items():
        if any(re.search(p, q) for p in patterns):
            return "lookup"

    return "semantic"
