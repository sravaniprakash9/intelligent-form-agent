"""Cross-form analytics and holistic insights."""

from __future__ import annotations

import json

from src.agent.llm_client import OllamaClient
from src.index.structured_store import StructuredStore


ANALYTICS_SYSTEM = (
    "You analyze multiple prior authorization forms. "
    "Use the provided statistics and records to answer holistically. "
    "Be concise and cite counts or examples."
)

_DISPLAY_LABELS: dict[str, str] = {
    "non_urgent": "Non-urgent",
    "urgent": "Urgent",
    "initial": "Initial",
    "extension_renewal_amendment": "Extension / renewal / amendment",
    "outpatient": "Outpatient",
    "inpatient": "Inpatient",
}


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    s = str(value).strip().lower()
    return not s or s in ("none", "null", "unknown", "name:")


def _display_label(value: object) -> str | None:
    if _is_missing(value):
        return None
    key = str(value).strip()
    return _DISPLAY_LABELS.get(key, key.replace("_", " ").title())


class MultiFormAnalyzer:
    def __init__(self, store: StructuredStore, llm: OllamaClient | None = None) -> None:
        self.store = store
        self.llm = llm or OllamaClient()

    def default_stats(self) -> dict:
        stats = {}
        stats["total_forms"] = self.store.query("SELECT COUNT(*) AS cnt FROM forms")[0]["cnt"]
        stats["review_type_counts"] = self.store.query(
            "SELECT review_type, COUNT(*) AS cnt FROM forms GROUP BY review_type"
        )
        stats["request_type_counts"] = self.store.query(
            "SELECT request_type, COUNT(*) AS cnt FROM forms GROUP BY request_type"
        )
        stats["setting_counts"] = self.store.query(
            """
            SELECT setting, COUNT(*) AS cnt FROM forms
            WHERE setting IS NOT NULL AND TRIM(setting) != ''
            GROUP BY setting
            """
        )
        stats["top_requesting_providers"] = self.store.query(
            """
            SELECT requesting_provider, COUNT(*) AS cnt
            FROM forms
            WHERE requesting_provider IS NOT NULL
              AND TRIM(requesting_provider) != ''
              AND LOWER(TRIM(requesting_provider)) NOT IN ('none', 'name:', 'name')
            GROUP BY requesting_provider
            ORDER BY cnt DESC
            LIMIT 5
            """
        )
        stats["patients"] = self.store.query(
            "SELECT form_id, patient_name, member_id, review_type FROM forms"
        )
        stats["procedures"] = self.store.query(
            "SELECT form_id, planned_service, code, icd_code FROM procedures LIMIT 20"
        )
        return stats

    def analyze(self, question: str | None = None) -> str:
        stats = self.default_stats()
        q = question or "Provide holistic insights across all forms."

        prompt = f"""Statistics:
{json.dumps(stats, indent=2, default=str)}

Question: {q}
"""
        return self.llm.generate(prompt, system=ANALYTICS_SYSTEM)

    def _breakdown_lines(self, rows: list[dict], key: str) -> list[str]:
        lines: list[str] = []
        for row in sorted(rows, key=lambda r: r.get("cnt", 0), reverse=True):
            label = _display_label(row.get(key))
            if label is None:
                continue
            lines.append(f"  • {label}: {row['cnt']}")
        return lines

    def analyze_structured(self) -> str:
        stats = self.default_stats()
        sections: list[str] = [
            f"Total forms: {stats['total_forms']}",
            "",
            "Review type",
            *self._breakdown_lines(stats["review_type_counts"], "review_type"),
            "",
            "Service setting",
            *self._breakdown_lines(stats["setting_counts"], "setting"),
            "",
            "Top requesting providers",
        ]
        provider_lines = [
            f"  • {row['requesting_provider']}: {row['cnt']}"
            for row in stats["top_requesting_providers"]
            if not _is_missing(row.get("requesting_provider"))
        ]
        sections.extend(provider_lines or ["  • (no provider names extracted)"])
        return "\n".join(sections)
