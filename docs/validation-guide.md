# Validation Guide

How to verify the Intelligent Form Agent works correctly by asking questions and comparing answers to ground truth.

## Validation strategy

Validate at three levels:

```text
Level 1: Extraction   → Does JSON match the source form image?
Level 2: Indexing     → Are forms listed and indexed without errors?
Level 3: Q&A          → Do answers match the extracted JSON?
```

A wrong answer at Level 3 may be an extraction problem (Level 1), not a QA problem.

---

## Step 1: Verify extraction

```bash
source .venv/bin/activate
export HF_HUB_DISABLE_XET=1

python -m src.cli list-forms
```

Pick a form ID, then inspect its JSON:

```bash
cat data/processed/0a01f77b-85f9-48c9-89bf-4b095bebb438_TX_page_1.json | python -m json.tool
```

Open the matching PNG in `data/raw/` side by side. Check:

| Field | JSON path |
|-------|-----------|
| Patient name | `section_iii_patient.name` |
| Gender | `section_iii_patient.gender` |
| Group number | `section_iii_patient.group_number` |
| Member ID | `section_iii_patient.member_id` |
| Review type | `section_ii_general.review_type` |
| Service setting | `section_v_services.setting` |
| Therapies | `section_v_services.therapies[]` |
| Requesting NPI | `section_iv_providers.requesting.npi` |
| Procedures | `section_v_services.procedures[]` |
| Issuer | `section_i_submission.issuer` |
| Extraction method | `extraction_method` (expect `surya` for PNGs) |
| Confidence | `extraction_confidence` (0.85–1.0 after recent parser fixes) |

Re-extract forms below target confidence:

```bash
python -m src.cli reextract-below-confidence --threshold 1.0
```

---

## Step 2: Verify indexing

```bash
python -m src.cli index
```

Expect output like:
```text
Indexed 0a01f77b-85f9-48c9-89bf-4b095bebb438_TX_page_1
...
Indexing complete.
```

Check files exist:
```bash
ls -la data/indexes/forms.duckdb data/indexes/chroma/
```

---

## Step 3: Ask validation questions

### Direct lookup (no LLM) — tests extraction + field_lookup

```bash
FORM=0a01f77b-85f9-48c9-89bf-4b095bebb438_TX_page_1

python -m src.cli ask "is he inpatient or outpatient" --form $FORM --no-llm
python -m src.cli ask "what kind of therapy" --form $FORM --no-llm
python -m src.cli ask "how many therapy sessions" --form $FORM --no-llm
python -m src.cli ask "What is the patient name?" --form $FORM --no-llm
python -m src.cli ask "What is the requesting provider NPI?" --form $FORM --no-llm
```

Compare each answer to the JSON. Example:

| Question | JSON value | Expected answer |
|----------|------------|-----------------|
| Setting | `outpatient` | `Service setting: outpatient (Section V)` |
| Therapy type | `physical_therapy` | `Therapy: physical therapy (Section V)` |
| Sessions | `4` | `Therapy sessions: 4 (Section V)` |
| Patient name | `Daniel Jarvis` | `Patient name: Daniel Jarvis (Section III)` |

### LLM-backed (full pipeline)

```bash
python -m src.cli ask "What is the group number?" --form $FORM
python -m src.cli ask "What procedure code and dates are listed?" --form $FORM
python -m src.cli ask "Is this request marked as urgent?" --form $FORM
```

Verify answers cite sections and match JSON values. LLM context excludes `raw_text` to avoid OCR checkbox noise.

### Cross-form validation

```bash
python -m src.cli analyze-all --no-llm
python -m src.cli analyze-all --question "How many requests are urgent vs non-urgent?"
python -m src.cli analyze-all --question "How many inpatient vs outpatient?"
```

Compare structured output to manual JSON inspection.

---

## Sample validation results (from project run)

These were executed against 11 sample forms after recent fixes:

| Question | Form | Answer |
|----------|------|--------|
| Setting | Daniel Jarvis | outpatient ✓ |
| Therapy type | Daniel Jarvis | physical therapy ✓ |
| Therapy sessions | Daniel Jarvis | 4 ✓ |
| Patient name | Form 1 | Dustin Carey ✓ |
| Group number | Form 1 | 85735 ✓ |
| Urgent count (all forms) | — | 8 urgent, 3 non-urgent ✓ |
| All forms at 100% confidence | — | 11/11 ✓ |

Known gaps (extraction, not QA):
- Member ID often `null`
- Requesting NPI often `null`
- Some procedure CPT codes `null` when OCR misses the value

---

## Validation checklist

Use this checklist before submission:

- [ ] `python -m src.cli extract` completes without errors
- [ ] `data/processed/` has one JSON per raw form
- [ ] `extraction_confidence` ≥ 0.85 on sample forms (or re-extract)
- [ ] `python -m src.cli index` completes without errors
- [ ] `python -m src.cli list-forms` shows all forms
- [ ] Ollama running: `curl http://localhost:11434/api/tags`
- [ ] Demo 1: single-form Q&A returns cited answer
- [ ] Demo 1b: `--no-llm` setting/therapy answers match JSON
- [ ] Demo 2: summarize returns patient + providers + procedures + therapy
- [ ] Demo 3: analyze-all returns cross-form counts (including settings)
- [ ] Answers match `data/processed/*.json` ground truth
- [ ] `pytest tests/test_field_lookup.py tests/test_therapy_parser.py -v` passes

---

## Diagnosing wrong answers

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| Answer says "None" but form has value | OCR/parser missed field | Check `raw_text` in JSON; tune parser; re-extract |
| LLM says both inpatient and outpatient | Old index had raw OCR chunks | Re-run `index`; verify no `_raw` chunks used |
| `--no-llm` correct but LLM wrong | LLM context issue | Fixed in current code (no raw_text); restart Ollama |
| Therapy sessions = 2 instead of 4 | OCR checkbox artifact | Fixed in `therapy_parser.py`; re-extract |
| UI shows forms below 100% confidence | Stale Streamlit session | Hard refresh; Re-index all processed JSON |
| "Ollama not running" | Ollama stopped | `open -a Ollama` |
| "Form not found" | Wrong form ID or missing extract | Run `list-forms`, then `extract` |
| Slow/hanging on first ask | Embedding model downloading | Set `HF_HUB_DISABLE_XET=1`; run `download-models` |

---

## Related docs

- [Demo Queries](demo_queries.md) — assignment demo commands
- [Setup & Execution Guide](setup-and-execution.md) — how to run everything
- [Troubleshooting](troubleshooting.md) — error fixes
