# Demo Queries

Example commands for the three required assignment demonstrations.  
Replace `<form_id>` with output from `python -m src.cli list-forms`.

**Example form IDs from sample data:**
- `00c33de5-461e-4642-966b-73ee20ef0d27_TX_page_1` (Dustin Carey)
- `0a01f77b-85f9-48c9-89bf-4b095bebb438_TX_page_1` (Daniel Jarvis — outpatient, physical therapy, 4 sessions)

Before running, ensure:
```bash
source .venv/bin/activate
export HF_HUB_DISABLE_XET=1
open -a Ollama
```

---

## 1. Single-form Q&A

```bash
python -m src.cli ask "What is the patient name and member ID?" --form <form_id>
python -m src.cli ask "What is the requesting provider NPI?" --form <form_id>
python -m src.cli ask "What is the group number?" --form <form_id>
python -m src.cli ask "What procedure code and dates are listed?" --form <form_id>
python -m src.cli ask "Is this request marked as urgent?" --form <form_id>
```

### Direct lookup demos (no LLM — instant, authoritative)

```bash
python -m src.cli ask "is he inpatient or outpatient" --form <form_id> --no-llm
python -m src.cli ask "what kind of therapy" --form <form_id> --no-llm
python -m src.cli ask "how many therapy sessions" --form <form_id> --no-llm
python -m src.cli ask "what is the patient gender" --form <form_id> --no-llm
```

**Expected:** Direct answer citing Section II, III, IV, or V fields.

**Sample output:**
```text
Service setting: outpatient (Section V)
Therapy: physical therapy (Section V)
Therapy sessions: 4 (Section V)
Patient name: Dustin Carey (Section III)
According to Section III, the group number is: 85735
```

---

## 2. Single-form summary

```bash
python -m src.cli summarize --form <form_id>
python -m src.cli summarize --form <form_id> --no-llm
```

**Expected:** Sectioned summary covering patient, request, providers, services (procedures, therapy), and key dates.

**CLI sample output** (`--no-llm`):
```text
### Patient
- **Name:** Daniel Jarvis
- **Member ID:** 62106
- **DOB:** Mar 20, 1992

### Request
- **Review type:** Non-Urgent
- **Request type:** Initial Request

### Services
- **Setting:** Outpatient
- **Therapy:** Physical Therapy · 4 sessions · 2 weeks
```

**Streamlit UI:** Click **Summarize** for blue/white section cards below the Ask / Summarize buttons (Patient card shows Member ID).

---

## 3. Multi-form holistic analysis

```bash
python -m src.cli analyze-all --question "How many requests are urgent vs non-urgent?"
python -m src.cli analyze-all --question "How many inpatient vs outpatient requests?"
python -m src.cli analyze-all --question "Which requesting providers appear on multiple forms?"
python -m src.cli analyze-all --question "List all unique ICD codes across forms."
```

**Expected:** Aggregate counts and narrative insights from DuckDB stats (including `setting_counts`) + Ollama.

**Sample output:**
```text
8 urgent requests and 3 non-urgent requests (out of 11 forms)
Setting breakdown: 9 outpatient, 2 inpatient
```

Also available in Streamlit **Batch upload & analyze** tab.

---

## Offline demos (no Ollama)

```bash
python -m src.cli summarize --form <form_id> --no-llm
python -m src.cli analyze-all --no-llm
python -m src.cli ask "is he inpatient or outpatient" --form <form_id> --no-llm
python -m src.cli ask "what kind of therapy" --form <form_id> --no-llm
```

---

## Validate answers

Compare CLI output to ground truth JSON:

```bash
cat data/processed/<form_id>.json | python -m json.tool
```

See [Validation Guide](validation-guide.md) for a full checklist.

---

## Related docs

- [Setup & Execution Guide](setup-and-execution.md) — how to run the project
- [Validation Guide](validation-guide.md) — verify correctness
- [Agent Layer](agent-layer.md) — how Q&A works internally
