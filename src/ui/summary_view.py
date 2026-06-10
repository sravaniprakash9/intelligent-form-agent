"""Streamlit card layout for structured form summaries (white & blue theme)."""

from __future__ import annotations

import html

import streamlit as st

from src.agent.summarizer import (
    _date_range,
    _fmt_date,
    _humanize,
    _procedure_title,
)
from src.extract.schema import FormDocument, ProcedureRow, TherapyInfo

_SUMMARY_STYLES = """
<style>
.pa-header {
    background: linear-gradient(135deg, #1E40AF 0%, #2563EB 100%);
    color: #FFFFFF;
    padding: 0.85rem 1.25rem;
    border-radius: 8px 8px 0 0;
    font-size: 1.05rem;
    font-weight: 600;
    letter-spacing: 0.02em;
    margin-bottom: 0;
}
.pa-wrap {
    background: #F8FAFC;
    border: 1px solid #BFDBFE;
    border-radius: 8px;
    padding: 0.75rem;
    margin-bottom: 1rem;
}
.pa-section-title {
    background: #EFF6FF;
    color: #1E40AF;
    border-left: 4px solid #2563EB;
    padding: 0.45rem 0.75rem;
    font-weight: 600;
    font-size: 0.92rem;
    margin: 0 0 0.65rem 0;
    border-radius: 0 4px 4px 0;
}
.pa-label {
    color: #64748B;
    font-size: 0.82rem;
    font-weight: 500;
    display: block;
    margin-bottom: 0.15rem;
}
.pa-value {
    color: #0F172A;
    font-size: 0.95rem;
    font-weight: 500;
    display: block;
    margin-bottom: 0.55rem;
}
.pa-value.empty {
    color: #94A3B8;
    font-style: italic;
    font-weight: 400;
}
.pa-badge {
    display: inline-block;
    background: #DBEAFE;
    color: #1D4ED8;
    padding: 0.2rem 0.55rem;
    border-radius: 999px;
    font-size: 0.8rem;
    font-weight: 600;
    margin-right: 0.35rem;
}
.pa-subheading {
    color: #334155;
    font-size: 0.85rem;
    font-weight: 600;
    margin: 0.75rem 0 0.35rem 0;
}
</style>
"""


def _esc(text: object | None) -> str:
    if text is None:
        return "—"
    s = str(text).strip()
    return html.escape(s) if s else "—"


def _value_class(text: str) -> str:
    return "pa-value empty" if text == "—" else "pa-value"


def _field(col, label: str, value: str) -> None:
    col.markdown(f'<span class="pa-label">{_esc(label)}</span>', unsafe_allow_html=True)
    col.markdown(
        f'<span class="{_value_class(value)}">{_esc(value)}</span>',
        unsafe_allow_html=True,
    )


def _badge(text: str) -> str:
    return f'<span class="pa-badge">{_esc(text)}</span>'


def _provider_block(
    title: str,
    name: str | None,
    npi: str | None,
    phone: str | None = None,
    fax: str | None = None,
    extra: list[tuple[str, str]] | None = None,
) -> None:
    st.markdown(
        f'<div class="pa-subheading">{_esc(title)}</div>',
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns(2)
    _field(c1, "Name", name or "—")
    _field(c2, "NPI", npi or "—")
    c3, c4 = st.columns(2)
    _field(c3, "Phone", phone or "—")
    _field(c4, "Fax", fax or "—")
    for label, value in extra or []:
        if value:
            _field(st.columns(1)[0], label, value)


def _procedure_rows(procedures: list[ProcedureRow]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for proc in procedures:
        rows.append(
            {
                "Service": _procedure_title(proc),
                "Code": proc.code or "—",
                "Dates": _date_range(proc.start_date, proc.end_date),
                "ICD": proc.icd_code or "—",
            }
        )
    return rows


def _therapy_line(t: TherapyInfo) -> str:
    parts = [_humanize(t.type)]
    if t.sessions is not None:
        parts.append(
            f"{t.sessions} session{'s' if t.sessions != 1 else ''}"
        )
    if t.duration:
        parts.append(f"Duration: {t.duration}")
    if t.frequency:
        parts.append(f"Frequency: {t.frequency}")
    return " · ".join(parts)


def render_summary_cards(doc: FormDocument) -> None:
    """Layout 1: stacked section cards with white & blue form styling."""
    p = doc.section_iii_patient
    g = doc.section_ii_general
    sub = doc.section_i_submission
    rp = doc.section_iv_providers.requesting
    sp = doc.section_iv_providers.service
    svc = doc.section_v_services
    clinical = doc.section_vi_clinical

    st.markdown(_SUMMARY_STYLES, unsafe_allow_html=True)

    st.markdown('<div class="pa-wrap">', unsafe_allow_html=True)
    st.markdown(
        '<div class="pa-header">Prior Authorization Summary</div>',
        unsafe_allow_html=True,
    )

    # Patient
    with st.container(border=True):
        st.markdown('<div class="pa-section-title">Patient</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        _field(c1, "Name", p.name or "—")
        _field(c2, "Member ID", p.member_id or "—")
        c3, c4 = st.columns(2)
        _field(c3, "Date of birth", _fmt_date(p.dob))
        _field(c4, "Gender", _humanize(p.gender))
        if p.phone:
            _field(st.columns(1)[0], "Phone", p.phone)

    # Request
    with st.container(border=True):
        st.markdown('<div class="pa-section-title">Request</div>', unsafe_allow_html=True)
        badges = []
        if g.review_type:
            badges.append(_badge(_humanize(g.review_type)))
        if g.request_type:
            badges.append(_badge(_humanize(g.request_type)))
        if svc.setting:
            badges.append(_badge(_humanize(svc.setting)))
        if badges:
            st.markdown("".join(badges), unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        _field(c1, "Review type", _humanize(g.review_type))
        _field(c2, "Request type", _humanize(g.request_type))
        c3, c4 = st.columns(2)
        _field(c3, "Submission date", _fmt_date(sub.submission_date))
        _field(c4, "Issuer", sub.issuer_name or "—")

    # Providers (Section IV — two columns like the form)
    with st.container(border=True):
        st.markdown(
            '<div class="pa-section-title">Providers (Section IV)</div>',
            unsafe_allow_html=True,
        )
        left, right = st.columns(2)
        with left:
            _provider_block(
                "Requesting Provider or Facility",
                rp.name,
                rp.npi,
                rp.phone,
                rp.fax,
                extra=[
                    ("Contact", rp.contact_name or ""),
                    ("Contact phone", rp.contact_phone or ""),
                ],
            )
        with right:
            _provider_block(
                "Service Provider or Facility",
                sp.name,
                sp.npi,
                sp.phone,
                sp.fax,
                extra=[
                    ("Primary care", sp.primary_care_provider_name or ""),
                    ("PCP phone", sp.primary_care_provider_phone or ""),
                ],
            )

    # Services
    with st.container(border=True):
        st.markdown('<div class="pa-section-title">Services</div>', unsafe_allow_html=True)
        _field(st.columns(1)[0], "Setting", _humanize(svc.setting))

        if svc.procedures:
            st.markdown('<div class="pa-subheading">Procedures</div>', unsafe_allow_html=True)
            st.dataframe(
                _procedure_rows(svc.procedures),
                use_container_width=True,
                hide_index=True,
            )

        if svc.therapies:
            st.markdown('<div class="pa-subheading">Therapy</div>', unsafe_allow_html=True)
            for t in svc.therapies:
                st.markdown(
                    f'<span class="pa-value">{_esc(_therapy_line(t))}</span>',
                    unsafe_allow_html=True,
                )

    # Clinical (optional)
    if clinical.address:
        with st.container(border=True):
            st.markdown('<div class="pa-section-title">Clinical</div>', unsafe_allow_html=True)
            _field(st.columns(1)[0], "Service address", clinical.address)

    st.markdown("</div>", unsafe_allow_html=True)
