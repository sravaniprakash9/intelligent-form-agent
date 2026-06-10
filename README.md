# Intelligent Form Agent

A **fully local** Intelligent Form Agent that reads Texas Prior Authorization forms, extracts structured data, answers questions, summarizes individual forms, and provides holistic insights across multiple forms.

No cloud APIs required — uses **hybrid OCR** (RapidOCR + selective Surya), **Ollama**, **sentence-transformers**, **Chroma**, and **DuckDB**.

## Features

- Extract key-value fields, checkboxes, and procedure tables from form images/PDFs
- **Hybrid OCR** (default in UI): fast full-page OCR + Surya only on failed field crops (~1–2 min/form vs 30+ min full Surya)
- Direct field lookup for common questions (setting, therapy, gender, NPI, etc.) — no LLM required
- Answer open-ended questions about a single form (with citations)
- Structured summaries (CLI markdown + Streamlit blue/white section cards)
- Cross-form analytics (urgency counts, settings, top providers, etc.)
- Streamlit UI with **single-form** and **batch multi-form** tabs, OCR mode selector (Hybrid / Full)
- Batch extract/index with skip-cache and low-confidence re-extract

## Project Structure

```text
/src        - main agent code
/data       - raw forms, processed JSON, indexes
/notebooks  - experiments
/tests      - unit tests
/docs       - full documentation (start here)
```

## Documentation

**New to the project?** Read the docs in this order:

| # | Document | Description |
|---|----------|-------------|
| 1 | [Documentation Index](docs/README.md) | Table of contents for all docs |
| 2 | [Setup & Execution Guide](docs/setup-and-execution.md) | **Install, configure, and run the project** |
| 3 | [Design Document](docs/design.md) | Goals, approach, and design decisions |
| 4 | [End-to-End Flow](docs/end-to-end-flow.md) | What happens from raw image to answer |
| 5 | [Architecture](docs/architecture.md) | System diagram and module map |

### By topic

| Topic | Document |
|-------|----------|
| How to run / setup | [Setup & Execution Guide](docs/setup-and-execution.md) |
| Design & requirements | [Design Document](docs/design.md) |
| Pipeline walkthrough | [End-to-End Flow](docs/end-to-end-flow.md) |
| System architecture | [Architecture](docs/architecture.md) |
| OCR & parsing | [Extraction Pipeline](docs/extraction-pipeline.md) |
| DuckDB + Chroma | [Indexing & Retrieval](docs/indexing-and-retrieval.md) |
| Q&A, summary, analytics | [Agent Layer](docs/agent-layer.md) |
| Streamlit UI | [UI Guide](docs/ui-guide.md) |
| Source code guide | [Code Walkthrough](docs/code-walkthrough.md) |
| Assignment demos | [Demo Queries](docs/demo_queries.md) |
| Validate with questions | [Validation Guide](docs/validation-guide.md) |
| Fix common errors | [Troubleshooting](docs/troubleshooting.md) |

## Quick Start

```bash
brew install python@3.11 ollama
cd intelligent-form-agent
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env

export HF_HUB_DISABLE_XET=1   # recommended on macOS
open -a Ollama && ollama pull llama3.1:8b

python -m src.cli download-models
cp /path/to/forms/*.png data/raw/
python -m src.cli extract --input data/raw
python -m src.cli index
```

See [Setup & Execution Guide](docs/setup-and-execution.md) for full instructions.

## Usage

```bash
# List forms
python -m src.cli list-forms

# Ask a question
python -m src.cli ask "What is the patient name?" --form <form_id>

# Summarize
python -m src.cli summarize --form <form_id>

# Multi-form analysis
python -m src.cli analyze-all --question "How many requests are urgent vs non-urgent?"

# Re-extract forms below 100% confidence, then re-index all
python -m src.cli reextract-below-confidence --threshold 1.0

# Optional UI (single form + batch upload & analyze)
make ui   # http://localhost:8501

# Tests
make test
```

Assignment demo commands: [Demo Queries](docs/demo_queries.md)

## Pipeline

```text
Form file (data/raw/)
  → PyMuPDF text (PDFs with embedded text)
  → else Hybrid OCR (default): RapidOCR full page → Surya on failed crops only
  → else Full mode: Surya full-page OCR (slower, higher accuracy on hard scans)
  → else Tesseract (optional fallback)
  → template parser + checkboxes + therapy OCR → JSON (data/processed/)
  → DuckDB + Chroma index (structured chunks only, no raw OCR)
  → lookup_field (direct answers) → else Ollama (QA / summary / analytics)
```

## Prerequisites

- **Python 3.11** — `brew install python@3.11`
- [Ollama](https://ollama.com) — local LLM
- ~15–20 GB disk (packages + models)
- Poppler (PDFs only): `brew install poppler`

## Creative Extensions

- `--no-llm` flag for offline structured output without Ollama
- **Hybrid OCR** with RapidOCR/PaddleOCR/EasyOCR fast engines and Surya crop fallback
- Streamlit UI with batch upload, cross-form analytics, formatted summary cards, and cached OCR models
- Texas-form checkbox detector (OpenCV-first; RapidOCR text fallbacks only when ink scores are inconclusive)
- Member ID layout-aware parsing; RapidOCR name/date/code normalization
- Therapy session extraction via layout-aware OCR + image crop fallback
- `reextract-below-confidence` CLI for targeted re-OCR
- Confidence scoring on extraction

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `Ollama not running` | `open -a Ollama` |
| HuggingFace download hangs | `export HF_HUB_DISABLE_XET=1` |
| Slow first extract | Use **Hybrid** mode in UI (default); run `download-models` first |
| LLM contradicts JSON on setting/therapy | Use `--no-llm`; re-run `index` |
| Stale confidence in Streamlit UI | Hard refresh; Re-index all processed JSON |
| Low OCR accuracy | Re-extract: `reextract-below-confidence --threshold 1.0` |
| Out of memory | Use `mistral:7b` in `.env` |

Full guide: [Troubleshooting](docs/troubleshooting.md)

## Docker (optional)

```bash
docker compose up -d ollama
docker compose exec ollama ollama pull llama3.1:8b
docker compose build form-agent
docker compose run --rm form-agent python -m src.cli extract --input data/raw
```
