"""Pydantic models for Texas Prior Authorization form extraction."""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field


class SubmissionSection(BaseModel):
    issuer_name: str | None = None
    submission_date: date | None = None


class GeneralSection(BaseModel):
    review_type: Literal["urgent", "non_urgent"] | None = None
    request_type: Literal["initial", "extension_renewal_amendment"] | None = None
    clinical_reason_for_urgency: str | None = None
    prev_auth_number: str | None = None


class PatientSection(BaseModel):
    name: str | None = None
    phone: str | None = None
    dob: date | None = None
    gender: Literal["male", "female", "other", "unknown"] | None = None
    subscriber_name: str | None = None
    member_id: str | None = None
    group_number: str | None = None


class ProviderInfo(BaseModel):
    name: str | None = None
    npi: str | None = None
    specialty: str | None = None
    phone: str | None = None
    fax: str | None = None
    contact_name: str | None = None
    contact_phone: str | None = None
    primary_care_provider_name: str | None = None
    primary_care_provider_phone: str | None = None


class ProviderSection(BaseModel):
    requesting: ProviderInfo = Field(default_factory=ProviderInfo)
    service: ProviderInfo = Field(default_factory=ProviderInfo)


class ProcedureRow(BaseModel):
    planned_service: str | None = None
    code: str | None = None
    start_date: date | None = None
    end_date: date | None = None
    diagnosis_description: str | None = None
    icd_code: str | None = None


class TherapyInfo(BaseModel):
    type: str
    sessions: int | None = None
    duration: str | None = None
    frequency: str | None = None


class ServicesSection(BaseModel):
    setting: str | None = None
    therapies: list[TherapyInfo] = Field(default_factory=list)
    procedures: list[ProcedureRow] = Field(default_factory=list)
    home_health_md_order_attached: bool | None = None
    dme_md_order_attached: bool | None = None


class ClinicalSection(BaseModel):
    address: str | None = None
    notes: str | None = None


class FormDocument(BaseModel):
    form_id: str
    source_file: str
    section_i_submission: SubmissionSection = Field(default_factory=SubmissionSection)
    section_ii_general: GeneralSection = Field(default_factory=GeneralSection)
    section_iii_patient: PatientSection = Field(default_factory=PatientSection)
    section_iv_providers: ProviderSection = Field(default_factory=ProviderSection)
    section_v_services: ServicesSection = Field(default_factory=ServicesSection)
    section_vi_clinical: ClinicalSection = Field(default_factory=ClinicalSection)
    extraction_confidence: float = 0.0
    extraction_method: str | None = None
    raw_text: str | None = None
