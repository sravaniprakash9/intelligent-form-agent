"""Application configuration."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    data_dir: Path = Path("./data")
    raw_dir: Path = Path("./data/raw")
    processed_dir: Path = Path("./data/processed")
    index_dir: Path = Path("./data/indexes")

    ocr_engine: str = "surya"
    ocr_mode: str = "hybrid"
    hybrid_fast_engine: str = "rapidocr"
    hybrid_crop_confidence_threshold: float = 0.65
    hybrid_enable_surya_fallback: bool = True
    hybrid_warmup_surya: bool = False
    enable_tesseract_fallback: bool = True
    pdf_min_text_chars: int = 100
    log_level: str = "INFO"

    @property
    def duckdb_path(self) -> Path:
        return self.index_dir / "forms.duckdb"

    @property
    def chroma_path(self) -> Path:
        return self.index_dir / "chroma"

    def ensure_dirs(self) -> None:
        for path in (self.raw_dir, self.processed_dir, self.index_dir):
            path.mkdir(parents=True, exist_ok=True)


settings = Settings()
