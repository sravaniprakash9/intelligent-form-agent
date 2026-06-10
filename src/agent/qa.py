"""Single-form question answering."""

from __future__ import annotations

from src.agent.field_lookup import lookup_field
from src.agent.llm_client import OllamaClient
from src.agent.prompt_utils import form_context_for_llm
from src.agent.router import classify_query
from src.extract.schema import FormDocument
from src.index.structured_store import StructuredStore
from src.index.vector_index import VectorIndex


SYSTEM_PROMPT = (
    "You are a medical prior authorization form assistant. "
    "Answer ONLY using the provided context. Cite the section when possible. "
    "If the answer is not in the context, say you cannot find it. "
    "IMPORTANT: Prefer the Key fields summary and structured JSON over context chunks. "
    "Checkbox fields (setting, review_type, request_type, gender, therapies) in structured "
    "JSON are authoritative. Ignore OCR artifacts like '[ 7 ]', '[]', or '| Label' in chunks."
)


class FormQA:
    def __init__(
        self,
        store: StructuredStore,
        vector_index: VectorIndex,
        llm: OllamaClient | None = None,
    ) -> None:
        self.store = store
        self.vector_index = vector_index
        self.llm = llm or OllamaClient()

    def _lookup_field(self, doc: FormDocument, question: str) -> str | None:
        return lookup_field(doc, question)

    def answer(self, question: str, form_id: str, doc: FormDocument | None = None) -> str:
        if doc:
            direct = lookup_field(doc, question)
            if direct:
                return direct

        route = classify_query(question)
        if route == "lookup" and doc:
            return "I could not find that field in the structured form data."

        chunks = self.vector_index.search(question, form_id=form_id, top_k=5)
        context = "\n\n".join(c["text"] for c in chunks)
        form_json = form_context_for_llm(doc) if doc else ""

        prompt = f"""Context chunks:
{context}

{form_json}

Question: {question}
"""
        return self.llm.generate(prompt, system=SYSTEM_PROMPT)
