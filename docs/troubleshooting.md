# Troubleshooting

Common issues and fixes when running the Intelligent Form Agent.

---

## Ollama

### `Ollama not running`

```bash
open -a Ollama
# or
ollama serve
```

Verify:
```bash
curl http://localhost:11434/api/tags
ollama list
```

### Model not found

```bash
ollama pull llama3.1:8b
```

### Ollama crashes on Apple Silicon

Install the **official Ollama app** from https://ollama.com instead of the x86 Homebrew bottle. The app ships a universal ARM64 binary.

### Out of memory with LLM

Use a smaller model in `.env`:
```
OLLAMA_MODEL=mistral:7b
```

Then: `ollama pull mistral:7b`

---

## HuggingFace / model downloads

### Download hangs indefinitely

On some Macs, HuggingFace XET transfer stalls. Fix:

```bash
export HF_HUB_DISABLE_XET=1
```

Add to `~/.zshrc` for persistence. Then:

```bash
python -m src.cli download-models
```

### `huggingface.co` DNS / network error

```bash
nslookup huggingface.co
```

Fix network/DNS, then:
```bash
export HF_HUB_DISABLE_XET=1
python -m src.cli download-models
```

### Surya models not cached and offline

```
Surya models not cached and huggingface.co is unreachable
```

Connect to internet once and run `download-models`. Models cache in `~/.cache/huggingface/hub/`.

---

## OCR / extraction

### Slow first extract

**Full Surya mode:** Expect **3–30+ minutes per PNG** on first run. Surya loads two models and runs full-page inference.

**Hybrid mode (Streamlit default):** RapidOCR full page + Surya only on failed field crops. Expect **~1–2 minutes per form**. Use **Hybrid + RapidOCR** in the UI extraction settings.

Speed up:
```bash
python -m src.cli download-models   # pre-download Surya + embeddings
```

For batch processing, use the Streamlit **Batch upload & analyze** tab with **Hybrid** mode (cached models, skip-existing) or `BatchPipeline` via CLI.

### Parser fixes not reflected after re-extract

The UI caches extracts by OCR mode, engine, and parser version (`_PARSER_CACHE_VERSION`). Click **Re-extract with selected OCR engine** or change settings to bust cache. CLI: `extract --force`.

### All forms show non-urgent (urgent checkbox clearly checked)

Older hybrid runs used text layout fallbacks that forced `non_urgent` when RapidOCR placed `Urgent` between Clinical Reason and Review Type. **Force re-extract** after parser updates.

### All forms show urgent (non-urgent checkbox clearly checked)

A bad `_detect_left_right_row` scan below the `Review Type:` label biased toward `urgent`. Review type now uses **label order before `Review Type:`** (`Non-Urgent` present → non-urgent; only `Urgent` → urgent). **Force re-extract** after parser version `ocr-v11-review-type-order`.

### RapidOCR accuracy looks low

1. Verify key fields in processed JSON (`member_id`, NPIs, setting, therapy sessions)
2. Hybrid mode should run Surya crop fallback for missing critical fields — check `extraction_method` contains `hybrid:rapidocr+surya-crops`
3. Switch to **Full** Surya mode for difficult scans (much slower)
4. Re-extract after parser updates: `python -m src.cli extract --input data/raw --force`

### Re-extract low-confidence forms

After parser fixes, re-OCR only forms below a threshold:
```bash
python -m src.cli reextract-below-confidence --threshold 1.0
```
Expect ~1–10 min per form depending on hardware. Re-indexes all processed JSON when done.

### Low OCR accuracy / many null fields

1. Check `raw_text` in processed JSON — is the value in the OCR output?
2. If yes → parser issue → edit `src/extract/form_parser.py`
3. If no → OCR issue → try `--force` re-extract or higher-resolution source image

### Tesseract fallback used instead of Surya

Check `extraction_method` in JSON. If `"tesseract"`, Surya failed in full mode. Check logs for the error. Ensure Surya models are downloaded.

Hybrid mode typically shows `hybrid:rapidocr+surya-crops(...)` — this is expected and preferred for speed.

### `ENABLE_TESSERACT_FALLBACK=false` and Surya fails

Either fix Surya or enable fallback in `.env`:
```
ENABLE_TESSERACT_FALLBACK=true
```

---

## Indexing

### `No processed forms. Run extract first.`

```bash
python -m src.cli extract --input data/raw
```

### Index hangs on embedding download

Same XET fix:
```bash
export HF_HUB_DISABLE_XET=1
python -m src.cli index
```

### Chroma errors after model change

Delete vector index and rebuild:
```bash
rm -rf data/indexes/chroma/
python -m src.cli index
```

---

## Q&A / agent

### Answer says "None" for a visible field

Extraction gap — check JSON ground truth:
```bash
cat data/processed/<form_id>.json | python -m json.tool
```

### LLM gives wrong answer but JSON is correct

Test without LLM to isolate:
```bash
python -m src.cli ask "is he inpatient or outpatient" --form <form_id> --no-llm
```

If `--no-llm` is correct, extraction and `lookup_field` are fine. Common LLM issues:
- Stale Chroma index with old `raw` OCR chunks → `python -m src.cli index`
- Ollama model drift → try `llama3.1:8b` refresh

### LLM says both inpatient and outpatient

Fixed in current code: raw OCR is excluded from LLM prompts and vector index. Re-index:
```bash
python -m src.cli index
```

### Therapy sessions wrong (e.g. 2 instead of 4)

OCR checkbox artifacts like `[ 2 Physical Therapy` were previously mistaken for session count. Re-extract after parser fix:
```bash
python -m src.cli extract --input data/raw --force
python -m src.cli index
```

### Streamlit shows stale confidence counts

Hard-refresh the browser (Cmd+Shift+R). Click **Re-index all processed JSON** in the batch tab sidebar.

### `Form '<id>' not found`

```bash
python -m src.cli list-forms
```

Use exact form ID from processed list. Run `extract` if missing.

---

## Python environment

### `python3.11 not found`

```bash
brew install python@3.11
python3.11 -m venv .venv
```

### Wrong Python version in venv

Recreate venv:
```bash
rm -rf .venv
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Copied venv from another Mac doesn't work

Always recreate `.venv` on a new machine — compiled packages (torch, chromadb) are architecture-specific.

### Apple Silicon running under Rosetta (slow)

If `file .venv/bin/python` shows `x86_64`, consider installing native ARM Homebrew at `/opt/homebrew` and recreating the venv.

---

## Tests

```bash
make test
# or targeted:
pytest tests/test_field_lookup.py tests/test_therapy_parser.py tests/test_table_parser.py tests/test_sanitize.py -v
```

---

## Getting help

1. Check [Validation Guide](validation-guide.md) to isolate which stage fails
2. Check [Setup & Execution Guide](setup-and-execution.md) for correct run order
3. Inspect `data/processed/<form_id>.json` as ground truth
4. Run with `--no-llm` to test without Ollama

---

## Related docs

- [Setup & Execution Guide](setup-and-execution.md)
- [Validation Guide](validation-guide.md)
