# Documentation Index

Welcome to the **Intelligent Form Agent** documentation. Read these docs in order if you are new to the project.

## Start here

| Document | What you'll learn |
|----------|-------------------|
| [Design Document](design.md) | Goals, requirements, design decisions, and high-level approach |
| [Setup & Execution Guide](setup-and-execution.md) | **Full setup, install, and run instructions** — start here to execute the project |
| [End-to-End Flow](end-to-end-flow.md) | What happens from raw form image to final answer, step by step |

## Architecture & code

| Document | What you'll learn |
|----------|-------------------|
| [Architecture](architecture.md) | System diagram, module map, technology choices |
| [Extraction Pipeline](extraction-pipeline.md) | OCR, parsing, checkboxes, therapy sessions, sanitization |
| [Indexing & Retrieval](indexing-and-retrieval.md) | DuckDB structured store, Chroma vector index (no raw OCR chunks) |
| [Agent Layer](agent-layer.md) | `lookup_field`, Q&A routing, summarization, multi-form analytics |
| [Code Walkthrough](code-walkthrough.md) | File-by-file guide to every module in `/src` |

## Running & validating

| Document | What you'll learn |
|----------|-------------------|
| [UI Guide](ui-guide.md) | Streamlit tabs — single form, batch, summaries, cross-form analytics |
| [Demo Queries](demo_queries.md) | Example commands for assignment demos + direct lookup queries |
| [Validation Guide](validation-guide.md) | How to verify the system works by asking questions |
| [Troubleshooting](troubleshooting.md) | Common errors and fixes (Ollama, HuggingFace, OCR, stale UI) |

## Quick reference

```text
Raw form (PNG/PDF)
  → extract                        → data/processed/*.json
  → reextract-below-confidence     → re-OCR low-confidence forms (optional)
  → index                          → data/indexes/ (DuckDB + Chroma)
  → ask / summarize / analyze-all  → lookup_field first, then LLM
```

**Minimum commands to run the project:**

```bash
source .venv/bin/activate
export HF_HUB_DISABLE_XET=1
open -a Ollama
python -m src.cli extract --input data/raw
python -m src.cli index
python -m src.cli ask "is he inpatient or outpatient" --form <form_id> --no-llm
```

**Streamlit UI** (single form + batch upload, Hybrid OCR default): `make ui` → http://localhost:8501

See [Setup & Execution Guide](setup-and-execution.md) for complete instructions.
