"""CLI entrypoint for the Intelligent Form Agent."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.markdown import Markdown

from src.agent.llm_client import OllamaClient
from src.agent.multi_form import MultiFormAnalyzer
from src.agent.field_lookup import lookup_field
from src.agent.qa import FormQA
from src.agent.summarizer import FormSummarizer
from src.config.settings import settings
from src.extract.hf_utils import can_use_surya, download_surya_models
from src.extract.document_extractor import DocumentExtractor
from src.extract.form_parser import FormParser
from src.pipeline.batch import BatchPipeline
from src.extract.schema import FormDocument
from src.index.structured_store import StructuredStore
from src.index.vector_index import VectorIndex
from src.ingest.loader import list_form_files

app = typer.Typer(help="Intelligent Form Agent — local document understanding")
console = Console()


def _form_id_from_path(path: Path) -> str:
    return path.stem.replace(" ", "_").replace(".", "_")


def _load_processed(form_id: str) -> FormDocument:
    path = settings.processed_dir / f"{form_id}.json"
    if not path.exists():
        raise typer.BadParameter(f"Form '{form_id}' not found. Run extract first.")
    return FormDocument.model_validate_json(path.read_text())


@app.command()
def download_models() -> None:
    """Pre-download HuggingFace models (Surya OCR + embeddings) for offline use."""
    console.print("[bold]Downloading Surya OCR models...[/bold]")
    try:
        download_surya_models()
    except RuntimeError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    console.print("[bold]Downloading embedding model...[/bold]")
    from sentence_transformers import SentenceTransformer

    SentenceTransformer(settings.embedding_model)
    console.print("[green]All models downloaded.[/green]")


@app.command()
def extract(
    input: Path = typer.Option(settings.raw_dir, "--input", "-i", help="Input directory"),
    force: bool = typer.Option(False, "--force", help="Re-extract existing forms"),
) -> None:
    """Extract structured JSON from all forms in input directory."""
    settings.ensure_dirs()
    files = list_form_files(input)
    if not files:
        console.print(f"[yellow]No forms found in {input}[/yellow]")
        raise typer.Exit(1)

    if not can_use_surya() and settings.enable_tesseract_fallback:
        console.print(
            "[yellow]Warning: huggingface.co unreachable and Surya models not cached.[/yellow]\n"
            "[yellow]Using Tesseract fallback. Fix DNS and run: python -m src.cli download-models[/yellow]"
        )
    elif not can_use_surya():
        console.print("[red]HuggingFace unreachable and Tesseract fallback disabled.[/red]")
        raise typer.Exit(1)

    parser = FormParser()
    for path in files:
        form_id = _form_id_from_path(path)
        out_path = settings.processed_dir / f"{form_id}.json"
        if out_path.exists() and not force:
            console.print(f"[dim]Skipping {path.name} (already extracted)[/dim]")
            continue

        console.print(f"Extracting [bold]{path.name}[/bold]...")
        doc = parser.parse_file(form_id, path)
        out_path.write_text(doc.model_dump_json(indent=2))
        console.print(
            f"  → {out_path.name} (method: {doc.extraction_method}, "
            f"confidence: {doc.extraction_confidence:.0%})"
        )

    console.print(f"[green]Done. {len(files)} form(s) processed.[/green]")


@app.command(name="reextract-below-confidence")
def reextract_below_confidence(
    threshold: float = typer.Option(
        1.0, "--threshold", "-t", help="Re-extract forms with confidence below this (1.0 = 100%%)"
    ),
) -> None:
    """Re-extract and re-index only forms whose saved confidence is below the threshold."""
    settings.ensure_dirs()
    parser = FormParser(extractor=DocumentExtractor(fast_preprocess=True))
    pipeline = BatchPipeline(parser)

    to_extract: list[tuple[str, Path]] = []
    for json_path in sorted(settings.processed_dir.glob("*.json")):
        doc = FormDocument.model_validate_json(json_path.read_text())
        if doc.extraction_confidence >= threshold:
            console.print(f"[dim]Skipping {doc.form_id} ({doc.extraction_confidence:.0%})[/dim]")
            continue
        raw_path = settings.raw_dir / doc.source_file
        if not raw_path.exists():
            raw_path = settings.raw_dir / f"{doc.form_id}.png"
        if not raw_path.exists():
            for ext in (".png", ".jpg", ".jpeg", ".pdf"):
                candidate = settings.raw_dir / f"{doc.form_id}{ext}"
                if candidate.exists():
                    raw_path = candidate
                    break
        if not raw_path.exists():
            console.print(f"[yellow]No raw file for {doc.form_id}, skipping[/yellow]")
            continue
        to_extract.append((doc.form_id, raw_path))

    if not to_extract:
        console.print(f"[green]No forms below {threshold:.0%} confidence.[/green]")
        raise typer.Exit(0)

    console.print(f"[bold]Re-extracting {len(to_extract)} form(s) below {threshold:.0%}…[/bold]")
    pipeline.warmup()
    docs: list[FormDocument] = []
    for form_id, path in to_extract:
        console.print(f"Extracting [bold]{path.name}[/bold]…")
        item = pipeline.extract_file(form_id, path, skip_existing=False, force=True)
        if item.doc:
            docs.append(item.doc)
            console.print(
                f"  → {item.doc.extraction_confidence:.0%} confidence ({item.seconds:.0f}s)"
            )

    console.print("[bold]Re-indexing all processed forms…[/bold]")
    all_docs = [
        FormDocument.model_validate_json(p.read_text())
        for p in sorted(settings.processed_dir.glob("*.json"))
    ]
    index_secs = pipeline.index_docs(all_docs)
    console.print(
        f"[green]Done. Re-extracted {len(docs)}, indexed {len(all_docs)} "
        f"(index {index_secs:.1f}s).[/green]"
    )


@app.command()
def index() -> None:
    """Build DuckDB + vector indexes from processed JSON."""
    settings.ensure_dirs()
    json_files = list(settings.processed_dir.glob("*.json"))
    if not json_files:
        console.print("[yellow]No processed forms. Run extract first.[/yellow]")
        raise typer.Exit(1)

    store = StructuredStore(settings.duckdb_path)
    vector_index = VectorIndex(settings.chroma_path, settings.embedding_model)

    docs = [FormDocument.model_validate_json(path.read_text()) for path in json_files]
    store.upsert_many(docs)
    vector_index.index_forms(docs)
    for doc in docs:
        console.print(f"Indexed {doc.form_id}")

    store.close()
    console.print(f"[green]Indexing complete. {len(docs)} form(s).[/green]")


@app.command()
def ask(
    question: str = typer.Argument(..., help="Question about the form"),
    form: str = typer.Option(..., "--form", "-f", help="Form ID"),
    no_llm: bool = typer.Option(False, "--no-llm", help="Use lookup only"),
) -> None:
    """Answer a question about a single form."""
    doc = _load_processed(form)
    store = StructuredStore(settings.duckdb_path)
    vector_index = VectorIndex(settings.chroma_path, settings.embedding_model)

    if no_llm:
        qa = FormQA(store, vector_index, llm=OllamaClient())
        direct = lookup_field(doc, question)
        console.print(direct or "No direct lookup match. Remove --no-llm for LLM answer.")
    else:
        llm = OllamaClient()
        if not llm.is_available():
            console.print("[red]Ollama not running. Start with: ollama serve[/red]")
            raise typer.Exit(1)
        qa = FormQA(store, vector_index, llm=llm)
        answer = qa.answer(question, form, doc)
        console.print(Markdown(answer))

    store.close()


@app.command()
def summarize(
    form: str = typer.Option(..., "--form", "-f", help="Form ID"),
    no_llm: bool = typer.Option(False, "--no-llm", help="Structured summary without LLM"),
) -> None:
    """Generate a summary of one form."""
    doc = _load_processed(form)
    summarizer = FormSummarizer()

    if no_llm:
        console.print(Markdown(summarizer.summarize_structured(doc)))
    else:
        llm = OllamaClient()
        if not llm.is_available():
            console.print("[yellow]Ollama unavailable — using structured summary[/yellow]")
            console.print(Markdown(summarizer.summarize_structured(doc)))
        else:
            summarizer = FormSummarizer(llm=llm)
            console.print(Markdown(summarizer.summarize(doc)))


@app.command(name="analyze-all")
def analyze_all(
    question: str = typer.Option(
        "Provide holistic insights across all prior authorization forms.",
        "--question",
        "-q",
    ),
    no_llm: bool = typer.Option(False, "--no-llm", help="Structured analytics only"),
) -> None:
    """Holistic analysis across all indexed forms."""
    if not settings.duckdb_path.exists():
        console.print("[yellow]No index found. Run index first.[/yellow]")
        raise typer.Exit(1)

    store = StructuredStore(settings.duckdb_path)
    analyzer = MultiFormAnalyzer(store)

    if no_llm:
        console.print(analyzer.analyze_structured())
    else:
        llm = OllamaClient()
        if not llm.is_available():
            console.print("[yellow]Ollama unavailable — using structured analytics[/yellow]")
            console.print(analyzer.analyze_structured())
        else:
            analyzer = MultiFormAnalyzer(store, llm=llm)
            console.print(Markdown(analyzer.analyze(question)))

    store.close()


@app.command()
def list_forms() -> None:
    """List available raw and processed forms."""
    raw = list_form_files(settings.raw_dir)
    processed = list(settings.processed_dir.glob("*.json"))

    console.print("[bold]Raw forms:[/bold]")
    for p in raw:
        console.print(f"  {p.name} → form_id: {_form_id_from_path(p)}")

    console.print("\n[bold]Processed forms:[/bold]")
    for p in processed:
        console.print(f"  {p.stem}")


if __name__ == "__main__":
    app()
