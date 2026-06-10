# Setup & Execution Guide

Complete instructions to install, configure, and run the Intelligent Form Agent from scratch.

## Table of contents

1. [Prerequisites](#1-prerequisites)
2. [Install system dependencies](#2-install-system-dependencies)
3. [Python environment](#3-python-environment)
4. [Configure the project](#4-configure-the-project)
5. [Download models](#5-download-models)
6. [Add form data](#6-add-form-data)
7. [Run the pipeline](#7-run-the-pipeline)
8. [Assignment demos](#8-assignment-demos)
9. [Optional: Streamlit UI](#9-optional-streamlit-ui)
10. [Optional: Docker](#10-optional-docker)
11. [Daily workflow cheat sheet](#11-daily-workflow-cheat-sheet)

---

## 1. Prerequisites

| Requirement | Notes |
|-------------|-------|
| **macOS** (tested) or Linux | Windows may work with manual adjustments |
| **Python 3.11** | Required — several ML packages pin to 3.11 |
| **~15–20 GB disk** | Python packages + Surya + Ollama model |
| **Internet (first run only)** | To download HuggingFace and Ollama models |
| **8 GB+ RAM** | 16 GB recommended for OCR + LLM |

---

## 2. Install system dependencies

### macOS (Homebrew)

```bash
# Python 3.11
brew install python@3.11

# Ollama (local LLM) — prefer the official app for Apple Silicon
brew install ollama
# OR download from https://ollama.com

# Tesseract (optional OCR fallback)
brew install tesseract

# Poppler (only needed for PDF processing)
brew install poppler
```

### Apple Silicon note

If your Mac has an M-series chip, use the **Ollama desktop app** (`open -a Ollama`) for native ARM64 performance. The app includes a universal binary that uses the GPU.

---

## 3. Python environment

```bash
cd intelligent-form-agent

# Create virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Upgrade pip and install dependencies
pip install -U pip setuptools wheel
pip install -r requirements.txt
```

**Alternative (Makefile):**

```bash
make setup
```

Verify installation:

```bash
python --version          # Python 3.11.x
python -c "import torch; print(torch.backends.mps.is_available())"  # True on Apple Silicon
```

---

## 4. Configure the project

```bash
cp .env.example .env
```

Key settings in `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama API endpoint |
| `OLLAMA_MODEL` | `llama3.1:8b` | Chat model name |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model for Chroma |
| `RAW_DIR` | `./data/raw` | Input forms directory |
| `PROCESSED_DIR` | `./data/processed` | Extracted JSON output |
| `INDEX_DIR` | `./data/indexes` | DuckDB + Chroma storage |
| `ENABLE_TESSERACT_FALLBACK` | `true` | Use Tesseract if Surya fails |

**Important (macOS):** HuggingFace XET downloads can hang on some Macs. Add this to your shell before running:

```bash
export HF_HUB_DISABLE_XET=1
```

Add it to `~/.zshrc` to make it permanent.

---

## 5. Download models

### Ollama LLM

```bash
# Start Ollama (pick one)
open -a Ollama          # macOS app (recommended)
# OR
ollama serve &          # terminal

# Pull the chat model (one-time, ~4.9 GB)
ollama pull llama3.1:8b

# Verify
ollama list
curl http://localhost:11434/api/tags
```

### HuggingFace models (Surya OCR + embeddings)

```bash
source .venv/bin/activate
export HF_HUB_DISABLE_XET=1

python -m src.cli download-models
```

This pre-downloads:
- `vikp/surya_det2` — text detection
- `vikp/surya_rec` — text recognition
- `sentence-transformers/all-MiniLM-L6-v2` — embeddings

After this step, the stack can run **fully offline**.

---

## 6. Add form data

Copy form images or PDFs into `data/raw/`:

```bash
cp /path/to/your/forms/*.png data/raw/
```

Supported formats: `.png`, `.jpg`, `.jpeg`, `.tiff`, `.tif`, `.pdf`

List what the agent sees:

```bash
python -m src.cli list-forms
```

Each file gets a **form ID** equal to its filename stem (e.g. `00c33de5-461e-4642-966b-73ee20ef0d27_TX_page_1`).

---

## 7. Run the pipeline

Always activate the environment first:

```bash
cd intelligent-form-agent
source .venv/bin/activate
export HF_HUB_DISABLE_XET=1
```

### Step 1 — Extract (OCR + parse → JSON)

```bash
python -m src.cli extract --input data/raw
```

| Flag | Description |
|------|-------------|
| `--input`, `-i` | Input directory (default: `data/raw`) |
| `--force` | Re-extract even if JSON already exists |

**Output:** `data/processed/<form_id>.json` per form

**Expected time:** CLI `extract` uses full Surya (~3–10+ minutes per PNG). The Streamlit UI defaults to **Hybrid** mode (~1–2 min/form: RapidOCR + Surya crops). Subsequent runs skip existing files unless `--force`.

### Step 1b — Re-extract low-confidence forms (optional)

```bash
python -m src.cli reextract-below-confidence --threshold 1.0
```

Re-OCRs only forms below the confidence threshold, then re-indexes all processed JSON. Useful after parser improvements.

| Flag | Description |
|------|-------------|
| `--threshold`, `-t` | Re-extract forms below this confidence (default: 1.0) |
| `--input`, `-i` | Raw forms directory (default: `data/raw`) |

### Step 2 — Index (DuckDB + Chroma)

```bash
python -m src.cli index
```

**Output:**
- `data/indexes/forms.duckdb` — structured SQL store
- `data/indexes/chroma/` — vector embeddings

### Step 3 — Query

```bash
# List available forms
python -m src.cli list-forms

# Ask a question (requires Ollama)
python -m src.cli ask "What is the patient name?" --form <form_id>

# Summarize
python -m src.cli summarize --form <form_id>

# Multi-form analysis
python -m src.cli analyze-all --question "How many requests are urgent vs non-urgent?"
```

### Offline mode (no Ollama)

Direct field lookup works without Ollama for setting, therapy, gender, patient, providers, and more:

```bash
python -m src.cli ask "is he inpatient or outpatient" --form <form_id> --no-llm
python -m src.cli ask "what kind of therapy" --form <form_id> --no-llm
python -m src.cli ask "how many therapy sessions" --form <form_id> --no-llm
python -m src.cli summarize --form <form_id> --no-llm
python -m src.cli analyze-all --no-llm
```

---

## 8. Assignment demos

Replace `<form_id>` with output from `list-forms`. Example form ID from sample data:

`00c33de5-461e-4642-966b-73ee20ef0d27_TX_page_1`

### Demo 1 — Single-form Q&A

```bash
python -m src.cli ask "What is the patient name and member ID?" --form 00c33de5-461e-4642-966b-73ee20ef0d27_TX_page_1
python -m src.cli ask "What is the requesting provider NPI?" --form 00c33de5-461e-4642-966b-73ee20ef0d27_TX_page_1
python -m src.cli ask "Is this request marked as urgent?" --form 00c33de5-461e-4642-966b-73ee20ef0d27_TX_page_1
```

### Demo 2 — Single-form summary

```bash
python -m src.cli summarize --form 00c33de5-461e-4642-966b-73ee20ef0d27_TX_page_1
```

### Demo 3 — Holistic multi-form analysis

```bash
python -m src.cli analyze-all --question "How many requests are urgent vs non-urgent?"
python -m src.cli analyze-all --question "Which requesting providers appear on multiple forms?"
python -m src.cli analyze-all --question "List all unique ICD codes across forms."
```

More examples: [Demo Queries](demo_queries.md)  
How to validate answers: [Validation Guide](validation-guide.md)

---

## 9. Streamlit UI

```bash
make ui
# Opens http://localhost:8501
```

**Extraction settings** (top of page):
- **Hybrid** (default) — RapidOCR full page + Surya only on failed field crops (~1–2 min/form)
- **Full** — Surya full-page OCR (slower, best for hard scans)
- Fast engine selector: RapidOCR (default), PaddleOCR, EasyOCR

Two tabs:

| Tab | What it does |
|-----|--------------|
| **Single form** | Upload → extract → metrics → JSON → Q&A → **Summarize** (blue section cards below buttons; Patient includes Member ID) |
| **Batch upload & analyze** | Multi-file upload, extract & index (skip-cache / force re-extract), **Forms in index** table (includes `member_id`), holistic analysis, per-form Q&A |

The batch tab caches OCR and embedding models for faster multi-form runs. Direct lookup answers work without Ollama.

**Tips:**
- If confidence counts look stale after a CLI re-extract, hard-refresh the browser and click **Re-index all processed JSON**.
- After parser code changes, click **Re-extract with selected OCR engine** (cache includes parser version).
- If port 8501 is busy: `streamlit run src/ui/app.py --server.port 8502`

---

## 10. Optional: Docker

```bash
docker compose up -d ollama
docker compose exec ollama ollama pull llama3.1:8b
docker compose build form-agent
docker compose run --rm form-agent python -m src.cli extract --input data/raw
```

---

## 11. Daily workflow cheat sheet

```bash
# 1. Start services
open -a Ollama
cd intelligent-form-agent && source .venv/bin/activate
export HF_HUB_DISABLE_XET=1

# 2. Process new forms
cp new_form.png data/raw/
python -m src.cli extract --input data/raw
python -m src.cli index

# 3. Ask questions
python -m src.cli list-forms
python -m src.cli ask "YOUR QUESTION" --form <form_id>

# 4. Re-extract forms below 100% confidence (after parser fixes)
python -m src.cli reextract-below-confidence --threshold 1.0

# 5. Run tests
make test
```

---

## Related docs

- [End-to-End Flow](end-to-end-flow.md) — what each step does internally
- [Troubleshooting](troubleshooting.md) — fix common errors
- [Architecture](architecture.md) — system design overview
