"""DuckDB store for structured form fields and cross-form analytics."""

from __future__ import annotations

import json
from pathlib import Path

import duckdb

from src.extract.schema import FormDocument


class StructuredStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = duckdb.connect(str(db_path))
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS forms (
                form_id VARCHAR PRIMARY KEY,
                source_file VARCHAR,
                submission_date DATE,
                review_type VARCHAR,
                request_type VARCHAR,
                patient_name VARCHAR,
                patient_dob DATE,
                patient_gender VARCHAR,
                member_id VARCHAR,
                requesting_provider VARCHAR,
                requesting_npi VARCHAR,
                service_provider VARCHAR,
                service_npi VARCHAR,
                setting VARCHAR,
                clinical_address VARCHAR,
                extraction_confidence DOUBLE,
                raw_json VARCHAR
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS procedures (
                form_id VARCHAR,
                planned_service VARCHAR,
                code VARCHAR,
                start_date DATE,
                end_date DATE,
                diagnosis_description VARCHAR,
                icd_code VARCHAR
            )
        """)

    def upsert_many(self, docs: list[FormDocument]) -> None:
        """Upsert multiple forms in one transaction."""
        if not docs:
            return
        self.conn.execute("BEGIN")
        try:
            for doc in docs:
                self.upsert(doc)
            self.conn.execute("COMMIT")
        except Exception:
            self.conn.execute("ROLLBACK")
            raise

    def upsert(self, doc: FormDocument) -> None:
        self.conn.execute("DELETE FROM forms WHERE form_id = ?", [doc.form_id])
        self.conn.execute("DELETE FROM procedures WHERE form_id = ?", [doc.form_id])

        self.conn.execute(
            """
            INSERT INTO forms VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                doc.form_id,
                doc.source_file,
                doc.section_i_submission.submission_date,
                doc.section_ii_general.review_type,
                doc.section_ii_general.request_type,
                doc.section_iii_patient.name,
                doc.section_iii_patient.dob,
                doc.section_iii_patient.gender,
                doc.section_iii_patient.member_id,
                doc.section_iv_providers.requesting.name,
                doc.section_iv_providers.requesting.npi,
                doc.section_iv_providers.service.name,
                doc.section_iv_providers.service.npi,
                doc.section_v_services.setting,
                doc.section_vi_clinical.address,
                doc.extraction_confidence,
                doc.model_dump_json(),
            ],
        )

        for proc in doc.section_v_services.procedures:
            self.conn.execute(
                """
                INSERT INTO procedures VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    doc.form_id,
                    proc.planned_service,
                    proc.code,
                    proc.start_date,
                    proc.end_date,
                    proc.diagnosis_description,
                    proc.icd_code,
                ],
            )

    def load_all_json(self, processed_dir: Path) -> list[FormDocument]:
        docs: list[FormDocument] = []
        for path in sorted(processed_dir.glob("*.json")):
            docs.append(FormDocument.model_validate_json(path.read_text()))
        return docs

    def query(self, sql: str) -> list[dict]:
        result = self.conn.execute(sql)
        cols = [d[0] for d in result.description]
        return [dict(zip(cols, row)) for row in result.fetchall()]

    def close(self) -> None:
        self.conn.close()
