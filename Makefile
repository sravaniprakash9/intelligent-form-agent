.PHONY: setup extract index ask summarize analyze test ui

PYTHON := /usr/local/opt/python@3.11/bin/python3.11

setup:
	$(PYTHON) -m venv .venv
	.venv/bin/pip install -U pip setuptools wheel
	.venv/bin/pip install -r requirements.txt
	@echo "Run: cp .env.example .env && ollama pull llama3.1:8b"

extract:
	.venv/bin/python -m src.cli extract --input data/raw

index:
	.venv/bin/python -m src.cli index

ask:
	.venv/bin/python -m src.cli ask "$(Q)" --form $(FORM)

summarize:
	.venv/bin/python -m src.cli summarize --form $(FORM)

analyze:
	.venv/bin/python -m src.cli analyze-all --question "$(Q)"

test:
	.venv/bin/pytest tests/ -v

ui:
	PYTHONPATH=. HF_HUB_DISABLE_XET=1 .venv/bin/streamlit run src/ui/app.py --server.headless true
