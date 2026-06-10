"""Optional Streamlit UI for form upload and Q&A."""

import sys
import tempfile
import time
from pathlib import Path

# Streamlit runs this file directly; ensure project root is on sys.path.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from src.agent.field_lookup import lookup_field
from src.agent.llm_client import OllamaClient
from src.agent.multi_form import MultiFormAnalyzer
from src.agent.qa import FormQA
from src.ui.summary_view import render_summary_cards
from src.config.settings import settings
from src.extract.document_extractor import DocumentExtractor
from src.extract.form_parser import FormParser
from src.extract.ocr_engine import OCREngine
from src.extract.ocr_engines import (
    OCR_ENGINE_INFO,
    OCR_MODE_INFO,
    available_fast_ocr_engines,
    available_ocr_engines,
    ocr_engine_label,
    ocr_mode_label,
)
from src.extract.schema import FormDocument
from src.index.structured_store import StructuredStore
from src.index.vector_index import VectorIndex
from src.pipeline.batch import BatchPipeline

st.set_page_config(page_title="Intelligent Form Agent", layout="wide")
st.title("Intelligent Form Agent")
st.caption("100% local — OCR, indexing, and Ollama LLM")

settings.ensure_dirs()


_PARSER_CACHE_VERSION = "ocr-v11-review-type-order"


def _make_ocr_engine(engine_name: str) -> OCREngine:
    """Build OCREngine; supports both old and new OCREngine signatures."""
    try:
        return OCREngine(engine=engine_name)
    except TypeError:
        return OCREngine()


@st.cache_resource
def _build_parser(
    ocr_mode: str,
    ocr_engine: str,
    fast_preprocess: bool,
    _cache_version: str,
) -> FormParser:
    if ocr_mode == "hybrid":
        return FormParser(
            extractor=DocumentExtractor(
                ocr_mode="hybrid",
                fast_engine=ocr_engine,
                fast_preprocess=False,
            )
        )
    return FormParser(
        extractor=DocumentExtractor(
            ocr=_make_ocr_engine(ocr_engine),
            fast_preprocess=fast_preprocess,
        )
    )


@st.cache_resource
def _cached_vector_index() -> VectorIndex:
    return VectorIndex(settings.chroma_path, settings.embedding_model)


def _form_id_from_name(filename: str) -> str:
    return Path(filename).stem.replace(" ", "_").replace(".", "_")


def _count_populated_fields(doc: FormDocument) -> int:
    fields = [
        doc.section_iii_patient.name,
        doc.section_iii_patient.member_id,
        doc.section_iii_patient.group_number,
        doc.section_ii_general.review_type,
        doc.section_iv_providers.requesting.name,
        doc.section_iv_providers.requesting.npi,
        doc.section_iv_providers.service.name,
    ]
    fields.extend(p.code for p in doc.section_v_services.procedures)
    return sum(1 for f in fields if f)


def _load_processed_docs() -> list[FormDocument]:
    docs: list[FormDocument] = []
    for path in sorted(settings.processed_dir.glob("*.json")):
        docs.append(FormDocument.model_validate_json(path.read_text()))
    return docs


def _indexed_form_count() -> int:
    if not settings.duckdb_path.exists():
        return 0
    store = StructuredStore(settings.duckdb_path)
    try:
        rows = store.query("SELECT COUNT(*) AS cnt FROM forms")
        return int(rows[0]["cnt"]) if rows else 0
    finally:
        store.close()


def _save_doc(doc: FormDocument) -> Path:
    out_path = settings.processed_dir / f"{doc.form_id}.json"
    out_path.write_text(doc.model_dump_json(indent=2))
    return out_path


def _index_docs(docs: list[FormDocument]) -> float:
    if not docs:
        return 0.0
    t0 = time.time()
    store = StructuredStore(settings.duckdb_path)
    vector_index = _cached_vector_index()
    try:
        store.upsert_many(docs)
        vector_index.index_forms(docs)
    finally:
        store.close()
    return time.time() - t0


def _render_ocr_engine_selector() -> tuple[str, str]:
    """OCR mode + engine dropdowns for extract / batch runs."""
    if "ocr_mode" not in st.session_state:
        st.session_state.ocr_mode = "hybrid"
    if "ocr_engine" not in st.session_state:
        st.session_state.ocr_engine = "rapidocr"

    mode = st.selectbox(
        "Extraction mode",
        options=["hybrid", "full"],
        format_func=ocr_mode_label,
        key="ocr_mode",
        help="Hybrid = fast layout OCR + Surya only on failed/low-confidence field crops.",
    )
    mode_info = OCR_MODE_INFO.get(mode, {})
    st.caption(f"{mode_info.get('speed', '')} · {mode_info.get('accuracy', '')}")

    if mode == "hybrid":
        engine_options = available_fast_ocr_engines()
        if st.session_state.ocr_engine not in engine_options:
            st.session_state.ocr_engine = engine_options[0]
        selected = st.selectbox(
            "Fast crop engine",
            options=engine_options,
            format_func=ocr_engine_label,
            key="ocr_engine",
            help="Used for full-page layout and field crops. Surya runs only on failed fields.",
        )
        st.caption("Accurate fallback: **Surya on low-confidence crops only** (not full document).")
    else:
        engine_options = available_ocr_engines()
        if st.session_state.ocr_engine not in engine_options:
            st.session_state.ocr_engine = engine_options[0]
        selected = st.selectbox(
            "OCR engine",
            options=engine_options,
            format_func=ocr_engine_label,
            key="ocr_engine",
            help="Single engine across the entire document.",
        )
        info = OCR_ENGINE_INFO.get(selected, {})
        st.caption(f"{info.get('speed', '')} · {info.get('accuracy', '')}")
        if selected != "surya":
            st.caption("Faster engines may lower confidence on checkboxes and handwriting.")

    return mode, selected


def _extract_with_progress(
    form_id: str, tmp_path: Path, ocr_mode: str, ocr_engine: str
) -> tuple[FormDocument, float]:
    parser = _build_parser(ocr_mode, ocr_engine, False, _PARSER_CACHE_VERSION)
    wall_t0 = time.time()
    with st.status("Processing form…", expanded=True) as status:
        st.write(f"**File:** `{tmp_path.name}`")
        if ocr_mode == "hybrid":
            st.write(f"**Mode:** `hybrid` — fast: `{ocr_engine}`, accurate: `surya` (crops only)")
        else:
            actual = getattr(parser.extractor.ocr, "engine", ocr_engine)
            st.write(f"**OCR engine:** `{ocr_engine}` (active: `{actual}`)")
        st.write("**Step 1/3** — Loading image / PDF")
        with st.spinner("Loading OCR models…"):
            parser.extractor.warmup()
        t0 = time.time()
        extracted = parser.extractor.extract(tmp_path)
        ocr_secs = time.time() - t0
        method = extracted.method or "unknown"
        st.write(
            f"**Step 2/3** — Text extraction complete "
            f"(`{method}`, {ocr_secs:.0f}s, {len(extracted.full_text):,} chars)"
        )
        if method == "surya":
            st.info("Surya OCR uses GPU — first run loads models and can take several minutes.")
        elif method == "rapidocr":
            st.caption("RapidOCR is faster; verify confidence and key fields if accuracy looks low.")
        elif method == "tesseract":
            st.caption(
                f"Tesseract OCR took {ocr_secs:.0f}s. For faster batch runs, try **RapidOCR**."
            )
        st.write("**Step 3/3** — Parsing form fields (Sections I–VI)")
        from src.ingest.loader import load_image

        image = extracted.image or load_image(tmp_path)
        doc = parser.parse_content(
            form_id=form_id,
            source_file=tmp_path.name,
            text=extracted.full_text,
            lines=extracted.lines,
            image=image,
            extraction_method=method,
        )
        total_secs = time.time() - wall_t0
        status.update(
            label=f"Extraction complete in {total_secs:.1f}s",
            state="complete",
            expanded=False,
        )
    return doc, total_secs


def _confidence_with_ocr(doc: FormDocument) -> str:
    ocr = doc.extraction_method or "—"
    return f"{doc.extraction_confidence:.0%} ({ocr})"


def _summary_key(form_id: str) -> str:
    return f"show_summary_{form_id}"


def _clear_summary(form_id: str) -> None:
    st.session_state[_summary_key(form_id)] = False


def _show_extraction_summary(
    doc: FormDocument,
    extract_seconds: float | None = None,
) -> None:
    if extract_seconds is not None:
        st.success(
            f"Extraction completed in **{extract_seconds:.1f}s** "
            f"({doc.extraction_method or 'unknown'} OCR)"
        )
    p = doc.section_iii_patient
    populated = _count_populated_fields(doc)
    cols = st.columns(4)
    cols[0].metric("Confidence · OCR", _confidence_with_ocr(doc))
    cols[1].metric("Fields found", populated)
    cols[2].metric("Procedures", len(doc.section_v_services.procedures))
    cols[3].metric(
        "Extraction time",
        f"{extract_seconds:.1f}s" if extract_seconds is not None else "—",
    )
    with st.expander("Key fields preview", expanded=False):
        st.json(
            {
                "patient_name": p.name,
                "member_id": p.member_id,
                "group_number": p.group_number,
                "gender": p.gender,
                "review_type": doc.section_ii_general.review_type,
                "request_type": doc.section_ii_general.request_type,
                "setting": doc.section_v_services.setting,
                "requesting_provider": doc.section_iv_providers.requesting.name,
                "requesting_npi": doc.section_iv_providers.requesting.npi,
                "procedures": [proc.code for proc in doc.section_v_services.procedures],
                "therapies": [t.type for t in doc.section_v_services.therapies],
            }
        )


def _forms_overview_table(docs: list[FormDocument]) -> None:
    rows = [
        {
            "form_id": d.form_id,
            "patient": d.section_iii_patient.name,
            "member_id": d.section_iii_patient.member_id,
            "review_type": d.section_ii_general.review_type,
            "setting": d.section_v_services.setting,
            "provider": d.section_iv_providers.requesting.name,
            "ocr": d.extraction_method or "—",
            "confidence": _confidence_with_ocr(d),
        }
        for d in docs
    ]
    st.dataframe(rows, use_container_width=True, hide_index=True)


def _render_ask_section(
    doc: FormDocument,
    form_id: str,
    llm: OllamaClient,
) -> None:
    st.subheader("Ask a question")
    question = st.text_input(
        "Question",
        placeholder="e.g. What is the patient name? Is he inpatient or outpatient?",
        key=f"ask_{form_id}",
    )
    col_ask, col_sum = st.columns(2)

    if col_ask.button("Ask", type="primary", key=f"ask_btn_{form_id}") and question:
        direct = lookup_field(doc, question)
        if direct:
            st.markdown(direct)
            st.caption("Answered from structured form data (no LLM).")
        elif not llm.is_available():
            st.error(
                "Ollama is not running and this question needs the LLM. "
                "Start it with: `open -a Ollama`, or ask about a specific field."
            )
        else:
            with st.status("Answering question…", expanded=True) as status:
                st.write("Indexing form for retrieval…")
                store = StructuredStore(settings.duckdb_path)
                vector_index = _cached_vector_index()
                store.upsert(doc)
                vector_index.index_form(doc)
                st.write("Querying Ollama…")
                qa = FormQA(store, vector_index, llm=llm)
                answer = qa.answer(question, form_id, doc)
                store.close()
                status.update(label="Answer ready", state="complete", expanded=False)
            st.markdown(answer)

    if col_sum.button("Summarize", key=f"sum_btn_{form_id}"):
        st.session_state[_summary_key(form_id)] = True
        st.rerun()

    if st.session_state.get(_summary_key(form_id)):
        st.markdown("#### Summary")
        render_summary_cards(doc)


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("How it works")
    st.markdown(
        """
        **Single form** — upload one form, review JSON, ask questions.

        **Batch & analyze** — upload multiple forms, extract & index them,
        then run cross-form analytics.
        """
    )
    llm = OllamaClient()
    if llm.is_available():
        st.success(f"Ollama ready ({settings.ollama_model})")
    else:
        st.warning("Ollama not running — LLM answers need `open -a Ollama`")
    st.caption("Field lookup and structured stats work without Ollama.")

    indexed = _indexed_form_count()
    processed = len(list(settings.processed_dir.glob("*.json")))
    st.metric("Processed JSON files", processed)
    st.metric("Indexed in DuckDB", indexed)

# ── Session state ────────────────────────────────────────────────────────────
if "single_extract_key" not in st.session_state:
    st.session_state.single_extract_key = None
if "single_doc" not in st.session_state:
    st.session_state.single_doc = None
if "single_form_id" not in st.session_state:
    st.session_state.single_form_id = None
if "single_tmp_path" not in st.session_state:
    st.session_state.single_tmp_path = None
if "batch_file_key" not in st.session_state:
    st.session_state.batch_file_key = None
if "batch_docs" not in st.session_state:
    st.session_state.batch_docs = []

with st.container(border=True):
    st.markdown("**Extraction settings**")
    _render_ocr_engine_selector()  # sets st.session_state.ocr_mode + ocr_engine

tab_single, tab_batch = st.tabs(["Single form", "Batch upload & analyze"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Single form
# ══════════════════════════════════════════════════════════════════════════════
with tab_single:
    uploaded = st.file_uploader(
        "Upload one form (PNG/JPG/PDF)",
        type=["png", "jpg", "jpeg", "pdf"],
        key="single_uploader",
    )

    if uploaded:
        file_key = f"{uploaded.name}:{uploaded.size}"
        extract_key = (
            f"{file_key}|{st.session_state.ocr_mode}|{st.session_state.ocr_engine}"
            f"|{_PARSER_CACHE_VERSION}"
        )
        suffix = Path(uploaded.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.getvalue())
            tmp_path = Path(tmp.name)

        form_id = _form_id_from_name(uploaded.name)
        st.session_state.single_form_id = form_id
        st.session_state.single_tmp_path = str(tmp_path)

        mode_label = (
            f"hybrid ({st.session_state.ocr_engine} + Surya crops)"
            if st.session_state.ocr_mode == "hybrid"
            else st.session_state.ocr_engine
        )
        needs_extract = (
            st.session_state.single_extract_key != extract_key
            or st.session_state.single_doc is None
        )
        if needs_extract:
            with st.spinner(f"Extracting with **{mode_label}**…"):
                st.session_state.single_doc, st.session_state.single_extract_seconds = (
                    _extract_with_progress(
                        form_id,
                        tmp_path,
                        st.session_state.ocr_mode,
                        st.session_state.ocr_engine,
                    )
                )
            st.session_state.single_extract_key = extract_key
            _clear_summary(form_id)
            _save_doc(st.session_state.single_doc)

        doc = st.session_state.single_doc
        form_id = st.session_state.single_form_id

        if st.button("Re-extract with selected OCR engine", key="single_reextract"):
            with st.spinner(f"Re-extracting with **{mode_label}**…"):
                st.session_state.single_doc, st.session_state.single_extract_seconds = (
                    _extract_with_progress(
                        form_id,
                        Path(st.session_state.single_tmp_path),
                        st.session_state.ocr_mode,
                        st.session_state.ocr_engine,
                    )
                )
            st.session_state.single_extract_key = extract_key
            _clear_summary(form_id)
            _save_doc(st.session_state.single_doc)
            st.rerun()

        _show_extraction_summary(doc, st.session_state.get("single_extract_seconds"))
        with st.expander("Full extracted JSON"):
            st.json(doc.model_dump(mode="json"))
        st.divider()
        _render_ask_section(doc, form_id, llm)
    else:
        st.info(
            "Upload a form to begin. **Hybrid mode** (default) uses fast OCR for layout "
            "and Surya only on failed field crops — much faster than full-page Surya."
        )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Batch upload & multi-form analysis
# ══════════════════════════════════════════════════════════════════════════════
with tab_batch:
    st.subheader("Upload multiple forms")
    uploaded_files = st.file_uploader(
        "Upload forms (PNG/JPG/PDF)",
        type=["png", "jpg", "jpeg", "pdf"],
        accept_multiple_files=True,
        key="batch_uploader",
    )

    batch_key = "|".join(f"{f.name}:{f.size}" for f in uploaded_files) if uploaded_files else ""

    skip_existing = st.checkbox(
        "Skip already-extracted forms (use cached JSON)",
        value=True,
        help=(
            "Loads existing JSON only when it was extracted with the **same OCR engine** "
            "selected above. Change engine or enable Force re-extract to run OCR again."
        ),
    )
    force_reextract = st.checkbox(
        "Force re-extract all (ignore cache)",
        value=False,
    )

    col_extract, col_reindex = st.columns(2)
    run_extract = col_extract.button(
        "Extract & index uploaded forms",
        type="primary",
        disabled=not uploaded_files,
    )
    run_reindex = col_reindex.button(
        "Re-index all processed JSON",
        help="Index every file in data/processed/ without re-running OCR",
    )

    if run_extract and uploaded_files:
        ocr_mode = st.session_state.ocr_mode
        ocr_engine = st.session_state.ocr_engine
        parser = _build_parser(
            ocr_mode,
            ocr_engine,
            ocr_mode != "hybrid" and ocr_engine == "surya",
            _PARSER_CACHE_VERSION,
        )
        pipeline = BatchPipeline(parser, vector_index=_cached_vector_index())
        warmup_label = (
            f"hybrid ({ocr_engine})"
            if ocr_mode == "hybrid"
            else ocr_engine
        )
        progress = st.progress(0, text=f"Loading {warmup_label} OCR (once)…")
        status = st.empty()
        timings: list[dict] = []

        with st.spinner(f"Warming up {warmup_label}…"):
            pipeline.warmup()

        files: list[tuple[str, Path]] = []
        for uploaded_file in uploaded_files:
            suffix = Path(uploaded_file.name).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.getvalue())
                files.append((_form_id_from_name(uploaded_file.name), Path(tmp.name)))

        docs: list[FormDocument] = []
        batch_t0 = time.time()
        for i, (form_id, tmp_path) in enumerate(files):
            uploaded_name = uploaded_files[i].name
            progress.progress(
                (i) / len(files),
                text=f"Form {i + 1}/{len(files)}: {uploaded_name}",
            )
            item = pipeline.extract_file(
                form_id,
                tmp_path,
                skip_existing=skip_existing and not force_reextract,
                force=force_reextract,
                ocr_engine=ocr_engine,
                ocr_mode=ocr_mode,
            )
            if item.doc:
                docs.append(item.doc)
            ocr_method = item.doc.extraction_method if item.doc else "—"
            conf = (
                f"{item.doc.extraction_confidence:.0%}"
                if item.doc
                else "—"
            )
            timings.append(
                {
                    "file": uploaded_name,
                    "status": item.status,
                    "ocr": ocr_method,
                    "confidence": conf,
                    "seconds": round(item.seconds, 1),
                }
            )
            status.caption(
                f"{'Skipped (cached)' if item.status == 'skipped' else 'Extracted'} "
                f"**{uploaded_name}** — OCR: {ocr_method}, confidence: {conf}, "
                f"{item.seconds:.0f}s"
            )

        progress.progress(0.95, text="Batch indexing (single embedding pass)…")
        index_secs = pipeline.index_docs(docs)
        total_secs = time.time() - batch_t0

        st.session_state.batch_file_key = batch_key
        st.session_state.batch_docs = docs
        progress.progress(1.0, text="Done")
        progress.empty()
        status.empty()

        extracted = sum(1 for t in timings if t["status"] == "extracted")
        skipped = sum(1 for t in timings if t["status"] == "skipped")
        extract_secs = sum(t["seconds"] for t in timings if t["status"] == "extracted")
        st.success(
            f"Done in **{total_secs:.0f}s** — "
            f"{extracted} extracted ({extract_secs:.0f}s OCR), {skipped} skipped, "
            f"{len(docs)} indexed (indexing: {index_secs:.1f}s)."
        )
        with st.expander("Per-form timing"):
            st.dataframe(timings, use_container_width=True, hide_index=True)
        if skipped and not force_reextract:
            st.caption(
                "Skipped forms used cached JSON (same OCR engine). "
                "Change OCR engine or check **Force re-extract** to run OCR again."
            )

    if run_reindex:
        docs = _load_processed_docs()
        if not docs:
            st.warning("No processed JSON files found in data/processed/.")
        else:
            index_secs = _index_docs(docs)
            st.session_state.batch_docs = docs
            st.success(f"Indexed {len(docs)} form(s) in {index_secs:.1f}s.")

    st.divider()
    st.subheader("Forms in index")

    all_docs = _load_processed_docs()
    if all_docs:
        _forms_overview_table(all_docs)
    else:
        st.caption("No forms yet — upload and extract, or run `make extract` from the CLI.")

    indexed_count = _indexed_form_count()
    if indexed_count >= 2:
        st.divider()
        st.subheader("Holistic analysis (all forms)")
        st.caption(f"Analyzing {indexed_count} indexed form(s) together.")

        analyze_q = st.text_input(
            "Cross-form question",
            placeholder='e.g. How many requests are urgent vs non-urgent? Which providers appear most?',
            key="analyze_question",
        )
        col_stats, col_llm = st.columns(2)

        if col_stats.button("Show stats", key="analyze_stats"):
            store = StructuredStore(settings.duckdb_path)
            try:
                st.markdown(MultiFormAnalyzer(store).analyze_structured())
            finally:
                store.close()

        if col_llm.button("Analyze with LLM", key="analyze_llm"):
            if not llm.is_available():
                st.error("Ollama is not running. Start it with: `open -a Ollama`")
            else:
                with st.spinner("Analyzing across forms…"):
                    store = StructuredStore(settings.duckdb_path)
                    try:
                        analyzer = MultiFormAnalyzer(store, llm=llm)
                        st.markdown(analyzer.analyze(analyze_q or None))
                    finally:
                        store.close()

        st.divider()
        st.subheader("Ask about one form from the batch")
        form_options = {d.form_id: d for d in all_docs}
        selected_id = st.selectbox("Select form", options=list(form_options.keys()))
        if selected_id:
            _render_ask_section(form_options[selected_id], selected_id, llm)

    elif indexed_count == 1:
        st.info("Upload and index at least 2 forms to enable holistic cross-form analysis.")
        if all_docs:
            st.divider()
            st.subheader("Ask about this form")
            d = all_docs[0]
            _render_ask_section(d, d.form_id, llm)
    elif uploaded_files and not run_extract:
        st.info("Click **Extract & index uploaded forms** to process your uploads.")
