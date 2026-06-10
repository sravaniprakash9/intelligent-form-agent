"""Batch extract and index pipeline with shared model reuse."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from src.config.settings import settings
from src.extract.form_parser import FormParser
from src.extract.schema import FormDocument
from src.index.structured_store import StructuredStore
from src.index.vector_index import VectorIndex


@dataclass
class BatchItemResult:
    form_id: str
    source_name: str
    status: str  # extracted | skipped | loaded
    seconds: float = 0.0
    doc: FormDocument | None = None


@dataclass
class BatchResult:
    items: list[BatchItemResult] = field(default_factory=list)
    index_seconds: float = 0.0

    @property
    def docs(self) -> list[FormDocument]:
        return [item.doc for item in self.items if item.doc is not None]

    @property
    def extracted_count(self) -> int:
        return sum(1 for i in self.items if i.status == "extracted")

    @property
    def skipped_count(self) -> int:
        return sum(1 for i in self.items if i.status in ("skipped", "loaded"))


def processed_json_path(form_id: str) -> Path:
    return settings.processed_dir / f"{form_id}.json"


def _extraction_cache_matches(cached_method: str, ocr_mode: str, ocr_engine: str) -> bool:
    """True when processed JSON matches the selected extraction settings."""
    if not cached_method:
        return False
    if ocr_mode == "hybrid":
        if not ocr_engine:
            return cached_method.startswith("hybrid:")
        return cached_method.startswith(f"hybrid:{ocr_engine}")
    if not ocr_engine:
        return True
    return cached_method == ocr_engine or cached_method.startswith(f"{ocr_engine}+")


def load_processed_doc(form_id: str) -> FormDocument | None:
    path = processed_json_path(form_id)
    if not path.exists():
        return None
    return FormDocument.model_validate_json(path.read_text())


class BatchPipeline:
    def __init__(
        self,
        parser: FormParser,
        store: StructuredStore | None = None,
        vector_index: VectorIndex | None = None,
    ) -> None:
        self.parser = parser
        self._store = store
        self._vector_index = vector_index

    def warmup(self) -> None:
        """Load OCR models once before processing multiple forms."""
        self.parser.extractor.warmup()

    def extract_file(
        self,
        form_id: str,
        path: Path,
        *,
        skip_existing: bool = True,
        force: bool = False,
        ocr_engine: str | None = None,
        ocr_mode: str | None = None,
    ) -> BatchItemResult:
        import time
        from src.ingest.loader import load_image

        t0 = time.time()
        existing = load_processed_doc(form_id)
        if skip_existing and not force and existing is not None:
            cached = (existing.extraction_method or "").lower()
            mode = (ocr_mode or "full").lower()
            wanted = (ocr_engine or "").lower()
            cache_ok = _extraction_cache_matches(cached, mode, wanted)
            if cache_ok:
                return BatchItemResult(
                    form_id=form_id,
                    source_name=path.name,
                    status="skipped",
                    seconds=time.time() - t0,
                    doc=existing,
                )

        extracted = self.parser.extractor.extract(path)
        image = extracted.image or load_image(path)
        doc = self.parser.parse_content(
            form_id=form_id,
            source_file=path.name,
            text=extracted.full_text,
            lines=extracted.lines,
            image=image,
            extraction_method=extracted.method,
        )
        processed_json_path(form_id).write_text(doc.model_dump_json(indent=2))
        return BatchItemResult(
            form_id=form_id,
            source_name=path.name,
            status="extracted",
            seconds=time.time() - t0,
            doc=doc,
        )

    def index_docs(self, docs: list[FormDocument]) -> float:
        import time

        if not docs:
            return 0.0
        t0 = time.time()
        store = self._store or StructuredStore(settings.duckdb_path)
        vector_index = self._vector_index or VectorIndex(
            settings.chroma_path, settings.embedding_model
        )
        own_store = self._store is None
        try:
            store.upsert_many(docs)
            vector_index.index_forms(docs)
        finally:
            if own_store:
                store.close()
        return time.time() - t0

    def extract_and_index(
        self,
        files: list[tuple[str, Path]],
        *,
        skip_existing: bool = True,
        force: bool = False,
    ) -> BatchResult:
        self.warmup()
        result = BatchResult()
        for form_id, path in files:
            result.items.append(
                self.extract_file(form_id, path, skip_existing=skip_existing, force=force)
            )
        result.index_seconds = self.index_docs(result.docs)
        return result
