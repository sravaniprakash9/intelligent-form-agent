# Setup (Quick Reference)

> **Full guide:** See [Setup & Execution Guide](setup-and-execution.md) for complete step-by-step instructions.

## Quick start

```bash
brew install python@3.11 ollama tesseract

cd intelligent-form-agent
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env

export HF_HUB_DISABLE_XET=1   # important on macOS

open -a Ollama
ollama pull llama3.1:8b

python -m src.cli download-models
cp /path/to/forms/*.png data/raw/
python -m src.cli extract
python -m src.cli index
make ui   # optional: Streamlit UI (Hybrid OCR default, summarize cards)
```

## First-run downloads

| Model | How | Size |
|-------|-----|------|
| Surya OCR | `download-models` or first `extract` | ~500 MB |
| all-MiniLM-L6-v2 | `download-models` or first `index` | ~90 MB |
| llama3.1:8b | `ollama pull` | ~4.9 GB |

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server |
| `OLLAMA_MODEL` | `llama3.1:8b` | Chat model |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Embeddings |
| `RAW_DIR` | `./data/raw` | Input forms |
| `PROCESSED_DIR` | `./data/processed` | Extracted JSON |
| `ENABLE_TESSERACT_FALLBACK` | `true` | Tesseract if Surya fails |

## Related docs

- [Setup & Execution Guide](setup-and-execution.md) — **complete instructions**
- [Troubleshooting](troubleshooting.md) — common errors
- [Documentation Index](README.md) — all docs
