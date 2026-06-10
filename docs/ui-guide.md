# Streamlit UI Guide

The Streamlit app (`src/ui/app.py`) provides a browser interface for single-form extraction, batch processing, Q&A, summaries, and cross-form analytics. Everything runs locally — OCR, indexing, and Ollama LLM.

**Start:** `make ui` → http://localhost:8501 (if port 8501 is busy: `streamlit run src/ui/app.py --server.port 8502`)

---

## Layout overview

```text
Sidebar                    Main area
────────                   ─────────
How it works               Extraction settings (mode + engine)
Ollama status              ┌─ Single form ─┬─ Batch upload & analyze ─┐
Processed / indexed counts └───────────────┴──────────────────────────┘
```

### Sidebar

| Item | Purpose |
|------|---------|
| Ollama status | Green = LLM ready; yellow = field lookup still works without Ollama |
| Processed JSON files | Count of `data/processed/*.json` |
| Indexed in DuckDB | Count of rows in `data/indexes/forms.duckdb` |

---

## Extraction settings

Shared by both tabs. Settings are stored in session state and included in the extract cache key.

| Setting | Options | Notes |
|---------|---------|-------|
| **Extraction mode** | `hybrid` (default), `full` | Hybrid = fast full-page OCR + Surya only on low-confidence field crops |
| **Fast crop engine** (hybrid) | RapidOCR, PaddleOCR, EasyOCR | Used for layout and crops; Surya runs only when needed |
| **OCR engine** (full) | Surya, Tesseract, etc. | Single engine across the entire document |

**Hybrid mode** is the default — typically ~1–2 min/form vs 30+ min for full Surya.

After parser logic changes, use **Re-extract** or **Force re-extract** so cached JSON is rebuilt. The UI cache version (`_PARSER_CACHE_VERSION` in `app.py`) invalidates stale parser results automatically on upload.

---

## Tab 1 — Single form

### Workflow

1. Upload one PNG, JPG, or PDF.
2. Extraction runs automatically with the selected OCR mode/engine.
3. Review metrics, key-field preview, and full extracted JSON.
4. Ask questions or click **Summarize**.

### After extraction

| Section | Content |
|---------|---------|
| Metrics | Confidence · OCR, fields found, procedures, extraction time |
| Key fields preview | Patient, member ID, review type, setting, providers, procedures |
| Full extracted JSON | Complete `FormDocument` schema |
| Ask a question | Field lookup first; LLM fallback when Ollama is running |
| Summary | Card layout below Ask/Summarize buttons (see below) |

### Re-extract

**Re-extract with selected OCR engine** re-runs OCR and parsing with the current mode/engine settings. Use this after changing extraction settings or when parser cache version bumps.

---

## Tab 2 — Batch upload & analyze

### Extract & index

1. Upload multiple forms.
2. Optional: **Skip already-extracted forms** (loads cached JSON when OCR engine matches).
3. Optional: **Force re-extract all** (ignores cache).
4. Click **Extract & index uploaded forms**.

`BatchPipeline` warms up OCR models once, extracts each file, then indexes all docs in a single embedding pass. Per-form timing is shown in an expander.

**Re-index all processed JSON** rebuilds DuckDB + Chroma from every file in `data/processed/` without re-running OCR.

### Forms in index

Table of all processed forms:

| Column | Source |
|--------|--------|
| form_id | Filename stem |
| patient | Section III name |
| member_id | Section III member ID |
| review_type | Section II urgent / non-urgent |
| setting | Section V inpatient / outpatient |
| provider | Requesting provider name |
| ocr | Extraction method (rapidocr, surya, etc.) |
| confidence | Extraction confidence + OCR method |

### Holistic analysis (2+ indexed forms)

| Action | Behavior |
|--------|----------|
| **Show stats** | SQL aggregates via `MultiFormAnalyzer.analyze_structured()` — no LLM |
| **Analyze with LLM** | Cross-form question answered by Ollama using indexed data |

### Ask about one form

Select a form from the dropdown to use the same Ask / Summarize section as the single-form tab.

---

## Summary cards (`summary_view.py`)

Click **Summarize** in the Ask section. The summary renders **below** the Ask/Summarize buttons (after metrics and Full extracted JSON).

| Section | Fields |
|---------|--------|
| Header | Patient name, review type badge, setting badge |
| Patient | Name, Member ID, DOB, gender, phone |
| Request | Request type, issuer, submission date |
| Providers | Two columns — Requesting (left) and Service (right) with phone/fax |
| Services | Procedures table (Start, End, **Code**, ICD, Qty), therapy lines |
| Clinical | Clinical reason (when present) |

Therapy lines show session count and duration explicitly, e.g. `3 sessions · Duration: 1 week`.

CLI `summarize --no-llm` produces equivalent markdown from the same JSON fields.

---

## Q&A behavior

1. **Direct lookup** — `lookup_field()` answers known fields instantly (patient name, member ID, review type, setting, therapies, etc.) without Ollama.
2. **LLM Q&A** — When lookup cannot answer, the form is indexed on the fly and `FormQA` queries Ollama. Requires `open -a Ollama`.

Field lookup and structured stats work offline. LLM answers need Ollama running.

---

## Common tasks

| Task | Steps |
|------|-------|
| Quick demo on one form | Single tab → upload → review JSON → Ask or Summarize |
| Process assignment forms | Batch tab → upload all → Extract & index → Show stats |
| Fix stale extraction after code change | Change OCR engine or enable Force re-extract; or click Re-extract on single tab |
| Compare OCR engines | Change engine in Extraction settings → Re-extract |
| Rebuild index without OCR | Batch tab → Re-index all processed JSON |

---

## Related docs

- [Setup & Execution Guide](setup-and-execution.md) — install and run
- [End-to-End Flow](end-to-end-flow.md) — pipeline from image to answer
- [Agent Layer](agent-layer.md) — Q&A routing and summarization
- [Demo Queries](demo_queries.md) — example questions for assignment demos
- [Troubleshooting](troubleshooting.md) — Ollama, OCR, stale cache issues
