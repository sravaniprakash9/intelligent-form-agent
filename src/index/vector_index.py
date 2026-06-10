"""Chroma vector index with local sentence-transformers embeddings."""

from __future__ import annotations

from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer

from src.extract.schema import FormDocument


class VectorIndex:
    def __init__(self, persist_dir: Path, embedding_model: str) -> None:
        persist_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name="form_chunks",
            metadata={"hnsw:space": "cosine"},
        )
        self.embedder = SentenceTransformer(embedding_model)

    def _chunks_for_form(self, doc: FormDocument) -> list[tuple[str, str, dict]]:
        chunks: list[tuple[str, str, dict]] = []
        base_meta = {"form_id": doc.form_id, "source_file": doc.source_file}

        fields = [
            ("submission", (
                f"Issuer: {doc.section_i_submission.issuer_name}, "
                f"Submission date: {doc.section_i_submission.submission_date}"
            )),
            ("patient", (
                f"Patient: {doc.section_iii_patient.name}, "
                f"DOB: {doc.section_iii_patient.dob}, "
                f"Member ID: {doc.section_iii_patient.member_id}, "
                f"Gender: {doc.section_iii_patient.gender}, "
                f"Group #: {doc.section_iii_patient.group_number}"
            )),
            ("general", (
                f"Review type: {doc.section_ii_general.review_type}, "
                f"Request type: {doc.section_ii_general.request_type}"
            )),
            ("requesting_provider", (
                f"Requesting provider: {doc.section_iv_providers.requesting.name}, "
                f"NPI: {doc.section_iv_providers.requesting.npi}"
            )),
            ("service_provider", (
                f"Service provider: {doc.section_iv_providers.service.name}, "
                f"NPI: {doc.section_iv_providers.service.npi}, "
                f"PCP: {doc.section_iv_providers.service.primary_care_provider_name}"
            )),
            ("setting", f"Service setting: {doc.section_v_services.setting}"),
            ("clinical", f"Clinical address: {doc.section_vi_clinical.address}"),
        ]
        for section, content in fields:
            if content and "None" not in content:
                chunks.append((f"{doc.form_id}_{section}", content, {**base_meta, "section": section}))

        for i, therapy in enumerate(doc.section_v_services.therapies):
            content = (
                f"Therapy: {therapy.type}, "
                f"Sessions: {therapy.sessions}, Duration: {therapy.duration}"
            )
            chunks.append((f"{doc.form_id}_therapy_{i}", content, {**base_meta, "section": "therapy"}))

        for i, proc in enumerate(doc.section_v_services.procedures):
            content = (
                f"Procedure: {proc.planned_service}, Code: {proc.code}, "
                f"Dates: {proc.start_date} to {proc.end_date}, "
                f"Diagnosis: {proc.diagnosis_description}, ICD: {proc.icd_code}"
            )
            chunks.append((f"{doc.form_id}_proc_{i}", content, {**base_meta, "section": "procedure"}))

        # Do not index raw OCR text — checkbox noise (e.g. "Inpatient [7] Outpatient [7]")
        # pollutes semantic search and contradicts structured checkbox fields.

        return chunks

    def _replace_form_chunks(self, chunks: list[tuple[str, str, dict]]) -> None:
        if not chunks:
            return
        form_ids = {meta["form_id"] for _, _, meta in chunks}
        for form_id in form_ids:
            existing = self.collection.get(where={"form_id": form_id})
            if existing["ids"]:
                self.collection.delete(ids=existing["ids"])
        ids, texts, metas = zip(*chunks)
        embeddings = self.embedder.encode(
            list(texts), show_progress_bar=False, batch_size=64
        ).tolist()
        self.collection.add(
            ids=list(ids),
            documents=list(texts),
            metadatas=list(metas),
            embeddings=embeddings,
        )

    def index_form(self, doc: FormDocument) -> None:
        chunks = self._chunks_for_form(doc)
        self._replace_form_chunks(chunks)

    def index_forms(self, docs: list[FormDocument]) -> None:
        """Index multiple forms with a single embedding batch (much faster than one-by-one)."""
        all_chunks: list[tuple[str, str, dict]] = []
        for doc in docs:
            all_chunks.extend(self._chunks_for_form(doc))
        self._replace_form_chunks(all_chunks)

    def search(self, query: str, form_id: str | None = None, top_k: int = 5) -> list[dict]:
        embedding = self.embedder.encode([query], show_progress_bar=False).tolist()
        where = {"form_id": form_id} if form_id else None
        # Over-fetch so we can drop legacy raw-OCR chunks from older indexes.
        n_results = top_k * 3
        results = self.collection.query(
            query_embeddings=embedding,
            n_results=n_results,
            where=where,
        )
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        ranked = [
            {"text": d, "metadata": m}
            for d, m in zip(docs, metas)
            if m.get("section") != "raw"
        ]
        return ranked[:top_k]
