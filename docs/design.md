# Design Document

## 1. Purpose

The **Intelligent Form Agent** is a locally runnable system that processes Texas Prior Authorization forms. It:

1. **Reads** form images and PDFs
2. **Extracts** structured data (patient, providers, procedures, therapies)
3. **Answers** questions about individual forms
4. **Summarizes** forms concisely
5. **Analyzes** multiple forms together for holistic insights

The assignment requires a QA and summarization pipeline. This project extends that with cross-form analytics, optional UI, and a fully offline-capable stack.

## 2. Requirements

### Functional

| Requirement | Implementation |
|-------------|----------------|
| Process structured and unstructured fields | Template parser + regex over OCR text; checkbox detector for review type |
| Answer questions about one form | `lookup_field` (always first) + vector search + Ollama |
| Summarize one form | `FormSummarizer` (LLM or structured fallback) |
| Holistic multi-form insights | `MultiFormAnalyzer` over DuckDB aggregates + Ollama |
| No cloud API dependency | Surya OCR, sentence-transformers, Ollama — all local |

### Non-functional

| Constraint | Decision |
|------------|----------|
| Privacy / offline | No OpenAI or cloud LLM; models cached after first download |
| Reproducibility | Canonical JSON schema (`FormDocument`) stored per form |
| Extensibility | Pydantic models, pluggable OCR engine, `--no-llm` flag |
| Assignment packaging | `/src`, `/data`, `/tests`, `/docs`, `requirements.txt`, `README.md` |

## 3. Design approach

### 3.1 Pipeline stages

The system is split into four independent stages. Each stage writes artifacts to disk so later stages can be re-run without repeating expensive OCR.

```text
Stage 1: INGEST     — load PNG/JPG/PDF from data/raw/
Stage 2: EXTRACT    — OCR + parse → data/processed/<form_id>.json
Stage 3: INDEX      — DuckDB + Chroma → data/indexes/
Stage 4: AGENT      — ask / summarize / analyze-all
```

This separation means:
- OCR (slow) runs once per form
- Indexing can be rebuilt after parser changes without re-OCR
- Q&A can be tested interactively without re-indexing

### 3.2 Extraction strategy

Forms arrive as scanned images (PNG) or PDFs. Text extraction follows a **priority waterfall**:

1. **PyMuPDF** — embedded/selectable PDF text (fast, no ML)
2. **Surya OCR** — HuggingFace document OCR for images and scanned PDFs (GPU/MPS on Apple Silicon)
3. **Tesseract** — optional fallback if Surya fails (`ENABLE_TESSERACT_FALLBACK=true`)

After text is obtained, a **template parser** maps content to Sections I–VI of the Texas Prior Authorization form using regex, section slicing, table parsing, therapy session OCR, OCR sanitization, and OpenCV checkbox detection.

### 3.3 Dual indexing strategy

Two indexes serve different query types:

| Index | Technology | Best for |
|-------|------------|----------|
| Structured | DuckDB | Counts, filters, cross-form SQL analytics |
| Semantic | Chroma + sentence-transformers | Fuzzy questions, context retrieval for LLM |

**Why both?** Structured fields (patient name, setting, therapy) answer quickly via `lookup_field` or SQL. Open-ended questions benefit from vector search over **structured chunks** (not raw OCR) plus LLM synthesis with a key-fields summary.

### 3.4 Agent / LLM strategy

**Ollama** (`llama3.1:8b` by default) handles:
- Natural-language answers with section citations
- Form summaries
- Narrative multi-form analysis

A **`--no-llm` flag** provides structured output without Ollama for offline demos and testing extraction quality in isolation.

`lookup_field()` in `src/agent/field_lookup.py` runs **before** routing for every question. Query routing (`src/agent/router.py`) classifies fallback questions as:
- **lookup** — direct field match (setting, therapy, gender, member ID, NPI, etc.)
- **semantic** — vector search + LLM (context excludes `raw_text`)
- **aggregate** — multi-form SQL stats + LLM

## 4. Data model

All extracted data conforms to `FormDocument` (Pydantic) in `src/extract/schema.py`:

```text
FormDocument
├── section_i_submission      (issuer, submission date)
├── section_ii_general        (review type, request type, urgency)
├── section_iii_patient       (name, DOB, member ID, group number)
├── section_iv_providers      (requesting + service provider details)
├── section_v_services        (therapies, procedures, setting)
├── section_vi_clinical       (address, notes)
├── extraction_confidence
├── extraction_method         (surya | pymupdf | tesseract)
└── raw_text                  (full OCR text — stored for debugging, not sent to LLM/index)
```

JSON files in `data/processed/` are the **source of truth** for validation.

## 5. Technology choices

| Component | Choice | Rationale |
|-----------|--------|-----------|
| OCR | Surya 0.4.5 | Strong document OCR, runs locally via HuggingFace |
| Embeddings | all-MiniLM-L6-v2 | Small, fast, good enough for form chunks |
| Vector DB | Chroma | Simple persistent local store, no server |
| Analytics DB | DuckDB | SQL over structured fields, embedded, no server |
| LLM | Ollama + llama3.1:8b | Local, no API keys, good instruction following |
| CLI | Typer + Rich | Clean commands, readable terminal output |
| UI (optional) | Streamlit | Quick upload + Q&A demo |

## 6. Creative extensions

Beyond core assignment requirements:

- **Streamlit UI** (`src/ui/app.py`) — single-form and batch upload tabs with holistic analysis
- **Batch pipeline** (`src/pipeline/batch.py`) — shared OCR warmup, skip-cache, batch index
- **`reextract-below-confidence`** — targeted re-OCR for low-confidence forms
- **Texas checkbox detector** — review type, setting, gender, therapies with OCR fallbacks
- **Therapy session OCR** — image crop fallback for handwritten session counts
- **Direct field lookup** — instant answers for setting, therapy, gender, etc. without LLM
- **Confidence scoring** — extraction confidence on each `FormDocument`
- **`--no-llm` mode** — full pipeline without Ollama for grading extraction separately from QA

## 7. Known limitations

| Limitation | Cause | Mitigation |
|------------|-------|------------|
| Member ID / NPI often null | OCR label-value alignment on dense forms | Tune regex in `form_parser.py` |
| Handwritten session counts | Surya misses digits in checkbox column | `therapy_parser.py` image crop OCR |
| Missing CPT on some rows | OCR gap in procedure table | Re-extract; check `raw_text` |
| Slow first extract | Surya model load + GPU warmup | `BatchPipeline.warmup()`; skip-existing in UI |
| HuggingFace download hangs | XET transfer on some Macs | Set `export HF_HUB_DISABLE_XET=1` |
| Stale UI confidence counts | Streamlit session cache | Hard refresh; click Re-index all |

## 8. Future improvements

- Native ARM64 Python on Apple Silicon for faster OCR
- Layout-aware parsing using OCR bounding boxes
- Fine-tuned NER model for provider/patient fields
- RAG with re-ranking for higher QA accuracy
- PDF multi-page support beyond first page

## Related docs

- [Architecture](architecture.md) — diagrams and module map
- [End-to-End Flow](end-to-end-flow.md) — step-by-step execution path
- [Setup & Execution Guide](setup-and-execution.md) — how to run everything
