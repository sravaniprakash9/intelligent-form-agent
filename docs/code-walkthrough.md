# Code Walkthrough

File-by-file guide to the `/src` directory. Read alongside [Architecture](architecture.md) and [End-to-End Flow](end-to-end-flow.md).

## Directory structure

```text
src/
├── cli.py                  # CLI entrypoint (Typer commands)
├── config/
│   └── settings.py         # Environment config (Pydantic Settings)
├── ingest/
│   ├── loader.py           # File listing and image/PDF loading
│   └── preprocess.py       # Image preprocessing (full + fast batch mode)
├── extract/
│   ├── schema.py           # Pydantic models (FormDocument)
│   ├── document_extractor.py  # Extraction orchestrator (hybrid + full modes)
│   ├── pdf_extractor.py    # PyMuPDF text extraction
│   ├── ocr_engine.py       # Surya, Tesseract, RapidOCR/Paddle/Easy crop OCR
│   ├── ocr_engines.py      # OCR mode registry and engine labels
│   ├── region_ocr.py       # RegionOCRReader: fast crop → Surya fallback
│   ├── field_refiner.py    # Surya crop retry for missing critical fields
│   ├── member_id_parser.py # Layout-aware Member ID vs Group # parsing
│   ├── hf_utils.py         # HuggingFace model download helpers
│   ├── form_parser.py      # OCR text → structured JSON
│   ├── checkbox_detector.py  # Checkbox detection (OpenCV + text fallbacks)
│   ├── table_parser.py     # Procedure table parsing
│   ├── therapy_parser.py   # Therapy sessions/duration (text + crop OCR)
│   └── sanitize.py         # Placeholder stripping + RapidOCR normalization
├── pipeline/
│   └── batch.py            # Batch extract + index with model reuse
├── index/
│   ├── structured_store.py # DuckDB storage
│   └── vector_index.py     # Chroma + sentence-transformers
├── agent/
│   ├── llm_client.py       # Ollama HTTP client
│   ├── router.py           # Query classification
│   ├── field_lookup.py     # Direct answers for known fields
│   ├── prompt_utils.py     # LLM context (excludes noisy raw_text)
│   ├── qa.py               # Single-form Q&A
│   ├── summarizer.py       # Form summaries
│   └── multi_form.py       # Cross-form analytics
└── ui/
    ├── app.py              # Streamlit UI (single + batch tabs)
    └── summary_view.py     # Blue/white section-card summary renderer
```

---

## `src/cli.py`

**Entry point for all commands.** Run as `python -m src.cli <command>`.

| Command | Function | What it does |
|---------|----------|--------------|
| `download-models` | `download_models()` | Pre-fetch Surya + embedding models |
| `extract` | `extract()` | OCR + parse all forms in input dir |
| `reextract-below-confidence` | `reextract_below_confidence()` | Re-OCR forms below confidence threshold |
| `index` | `index()` | Build DuckDB + Chroma from processed JSON |
| `ask` | `ask()` | Single-form Q&A |
| `summarize` | `summarize()` | Form summary |
| `analyze-all` | `analyze_all()` | Multi-form analytics |
| `list-forms` | `list_forms()` | Show raw and processed forms |

Key helpers:
- `_form_id_from_path()` — derives form ID from filename stem
- `_load_processed()` — loads `FormDocument` from `data/processed/`

---

## `src/config/settings.py`

Loads configuration from `.env` via Pydantic Settings.

```python
settings.ollama_host       # Ollama API URL
settings.ollama_model      # LLM model name
settings.embedding_model   # sentence-transformers model
settings.raw_dir           # data/raw
settings.processed_dir     # data/processed
settings.duckdb_path       # data/indexes/forms.duckdb
settings.chroma_path       # data/indexes/chroma
settings.ensure_dirs()     # create data directories
```

---

## `src/ingest/`

### `loader.py`
- `list_form_files(dir)` — find supported files
- `load_image(path)` — open PNG/JPG or render PDF page 0
- `pdf_to_image(path)` — PyMuPDF page render at 200 DPI

### `preprocess.py`
- `preprocess(image, fast=False)` — contrast/denoise before OCR
- `fast=True` skips denoise/CLAHE for quicker batch OCR (used in batch pipeline)

---

## `src/extract/`

### `schema.py`
Pydantic models for all form sections. `FormDocument` is the root model serialized to `data/processed/*.json`.

### `document_extractor.py`
Priority waterfall: PyMuPDF → hybrid or full OCR.

- **Hybrid** (`ocr_mode="hybrid"`): RapidOCR full page; never full-page Surya; `field_refiner` for crop fallback
- **Full**: Surya full page → optional Tesseract fallback
- Supports `fast_preprocess=True` for batch mode

### `form_parser.py`
Core parsing logic:
- Section-scoped parsing (I–VI)
- Texas checkbox fields via `detect_texas_form_fields()`
- Issuer name, submission date, provider phones
- Therapy sessions via `parse_therapy_sessions()`
- OCR sanitization via `sanitize.py`
- Expanded confidence scoring

### `ocr_engines.py` / `region_ocr.py` / `field_refiner.py`
- `ocr_engines.py` — registers hybrid vs full modes and fast engine options (RapidOCR, PaddleOCR, EasyOCR)
- `region_ocr.py` — `RegionOCRReader`: run fast crop OCR, fall back to Surya on low confidence
- `field_refiner.py` — after hybrid parse, Surya crop OCR for still-missing critical fields

### `member_id_parser.py`
Spatial column assignment so Member ID digits are not mistaken for Group #; crop OCR fallback.

### `provider_parser.py`
Section IV two-column layout: requesting (left) vs service (right) via header midpoint; phone/fax via label proximity; contact and PCP names near their labels.

### `checkbox_detector.py`
Texas Prior Auth form checkbox groups with layout-aware fallbacks:
- Review type, request type, gender, setting (inpatient/outpatient)
- Therapy types (physical, occupational, speech, etc.)
- Review type: `_resolve_review_type()` — OpenCV ink scores first; then **label order before `Review Type:`** (only `Urgent` → urgent; `Non-Urgent` before spurious `Urgent` → non-urgent); checkbox band comparison; text fallbacks last

### `table_parser.py`
Section V procedure rows:
- **CPT-anchored** parsing for scrambled RapidOCR (excludes diagnosis bleed like `Othermotorcycle`)
- Date-line fallback when CPT code is missing from OCR
- Groups lines by date rows; attaches trailing ICD lines
- Rejects years (2022, 2023) mistaken as CPT codes
- Normalizes mangled ICD (`.247.1` → `Z47.1`)

### `therapy_parser.py`
Therapy session count, type, and duration:
- `infer_therapy_from_session_column()` — session digit x-position between Physical / Occupational / Speech labels
- `parse_therapy_duration()` — reads `1 week` / `2 weeks`; does not treat bare session count as duration
- Text patterns after "Number of Sessions"
- Spatial OCR line matching near therapy row
- Hybrid crop OCR fallback for handwritten session counts

### `sanitize.py`
Strips OCR placeholders and normalizes RapidOCR/hybrid text:
- `| Urgent`, `(if different):`, `SECTION` as prev auth number
- CamelCase name splitting, mashed CPT/date splitting, section slicing by keywords

---

## `src/pipeline/batch.py`

`BatchPipeline` — shared model reuse for multi-form processing:
- `warmup()` — load Surya models once
- `extract_file()` — with `skip_existing` / `force` options
- `index_docs()` — batch DuckDB upsert + `index_forms()`
- `extract_and_index()` — full batch workflow

Used by Streamlit batch tab and `reextract-below-confidence` CLI.

---

## `src/index/`

### `structured_store.py`
- `upsert(doc)` — insert/replace form + procedure rows
- `upsert_many(docs)` — batch upsert in one transaction
- `query(sql)` — run SQL, return dicts

### `vector_index.py`
- `_chunks_for_form()` — patient, providers, setting, therapy, procedures (no raw OCR)
- `index_form(doc)` — embed and store chunks for one form
- `index_forms(docs)` — single batched embedding pass for multiple forms
- `search()` — cosine similarity; filters legacy `raw` chunks from old indexes

---

## `src/agent/`

### `field_lookup.py`
Centralized direct field lookup — bypasses LLM for known patterns:
- Patient, providers, setting, gender, therapy, procedures, ICD/CPT
- Setting checked before gender (fixes "is he inpatient or outpatient")
- Used by `FormQA`, CLI `--no-llm`, and Streamlit UI

### `prompt_utils.py`
- `key_fields_summary()` — compact authoritative fields for LLM
- `form_context_for_llm()` — structured JSON **without** `raw_text`

### `router.py`
`classify_query(question)` → `"lookup"` | `"semantic"` | `"aggregate"`

Expanded patterns: setting, gender, therapy, group number, issuer, clinical address.

### `qa.py`
`FormQA.answer()`:
1. **Always** tries `lookup_field()` first (regardless of route)
2. Falls back to vector search + LLM with sanitized context
3. System prompt instructs LLM to prefer structured fields over OCR noise

### `summarizer.py`
- `summarize()` — LLM summary via `form_context_for_llm()` (no raw OCR)
- `summarize_structured()` — sectioned markdown fallback (`--no-llm`, CLI Rich output)

### `summary_view.py`
`render_summary_cards()` — blue header, section cards (Patient includes Member ID), two-column Providers with phone/fax, badges, procedures table with **Code**/ICD columns, explicit therapy duration lines.

### `multi_form.py`
`default_stats()` includes `setting_counts` breakdown.

---

## `src/ui/app.py`

Streamlit app with two tabs:

**Extraction settings** (bordered container at top)
- **Hybrid** (default): RapidOCR + Surya crops only
- **Full**: Surya full-page OCR
- Parser cache version in extract key — re-extract after parser updates

**Single form**
- Upload → extract → metrics → Full extracted JSON → Q&A
- **Summarize** renders section cards below the Ask / Summarize buttons (session state)
- Key fields preview collapsed by default (includes `member_id`)
- Direct lookup works without Ollama

**Batch upload & analyze**
- Multi-file upload
- Extract & index with skip-cache and force re-extract options
- Cached OCR + embedding models (`@st.cache_resource`)
- Per-form timing table; **Forms in index** table includes `member_id`
- Holistic analysis (stats + LLM) when 2+ forms indexed
- Per-form Q&A dropdown from batch

Run: `make ui` → http://localhost:8501 (alternate port if busy: `--server.port 8502`). See [UI Guide](ui-guide.md).

---

## `tests/`

| File | Tests |
|------|-------|
| `test_field_lookup.py` | Direct lookup (setting, therapy, gender) |
| `test_table_parser.py` | Procedure rows, year-as-code rejection |
| `test_therapy_parser.py` | Session count, checkbox artifact rejection |
| `test_sanitize.py` | OCR placeholder stripping |
| `test_qa_router.py` | Query classification |
| `test_summarizer.py` | Structured markdown summary output |
| `test_member_id_parser.py` | Member ID column assignment |
| `test_region_ocr.py` | Fast crop OCR + Surya fallback |
| `test_rapidocr_parsing.py` | Hybrid normalization, Daniel Jarvis regression |
| `test_checkbox_text_fallback.py` | RapidOCR checkbox text fallbacks |
| `test_review_type_detection.py` | OpenCV-first review type; label-order before `Review Type:`; text fallback when inconclusive |
| `test_provider_parser.py` | Section IV two-column providers; phone/fax label proximity |
| `test_bobby_services.py` | CPT-anchored procedures; therapy type/duration on Bobby Juarez form |
| `test_batch_cache.py` | Batch extract cache behavior |
| `test_pdf_extractor.py` | PyMuPDF text extraction |

Run: `make test` or `pytest tests/ -v`

---

## Key extension points

| Want to... | Edit... |
|------------|---------|
| Add new form fields | `src/extract/schema.py` + `form_parser.py` |
| Add direct Q&A answers | `src/agent/field_lookup.py` + `router.py` |
| Improve OCR accuracy | `ocr_engine.py`, `region_ocr.py`, `field_refiner.py`, `preprocess.py` |
| Fix provider column mix-ups | `src/extract/provider_parser.py` |
| Change summary UI layout | `src/ui/summary_view.py` |
| Change LLM context | `src/agent/prompt_utils.py` |
| Add SQL analytics | `src/agent/multi_form.py` → `default_stats()` |
| Batch processing | `src/pipeline/batch.py` |

---

## Related docs

- [Extraction Pipeline](extraction-pipeline.md)
- [Indexing & Retrieval](indexing-and-retrieval.md)
- [Agent Layer](agent-layer.md)
- [UI Guide](ui-guide.md)
- [Design Document](design.md)
