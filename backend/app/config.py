"""Application configuration powered by environment variables."""
from __future__ import annotations

from pathlib import Path
from typing import List

from pydantic_settings import BaseSettings
from pydantic import Field


class AppSettings(BaseSettings):
    """Centralized settings for the FastAPI service."""

    model_path: Path = Path("models/baseline.joblib")
    transformer_dir: Path = Path("models/transformer")
    use_transformer: bool = True  
    frontend_dir: Path = Path("frontend")
    database_path: Path = Path("data/app.db")
    feedback_path: Path = Path("data/feedback.jsonl")  # Deprecated, kept for backward compatibility
    history_path: Path = Path("data/prediction_history.jsonl")  # Deprecated, kept for backward compatibility
    eval_metrics_path: Path = Path("reports/eval_metrics.json")
    history_summary_path: Path = Path("reports/history_summary.json")
    max_file_records: int = 1000
    stats_max_history: int = 100
    allow_origins: List[str] = Field(default_factory=lambda: ["*"])

    class Config:
        env_prefix = "APP_"
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = AppSettings()
