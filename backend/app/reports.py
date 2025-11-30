"""Helpers for loading evaluation and history reports for the dashboard."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Optional

from sqlalchemy.orm import Session

from .database import PredictionRecord as DBPredictionRecord, SessionLocal


class ReportLoader:
    """Load JSON reports from disk with graceful fallbacks."""

    def __init__(self, eval_metrics_path: Path, history_summary_path: Path) -> None:
        self.eval_metrics_path = eval_metrics_path
        self.history_summary_path = history_summary_path
        self._eval_fallback = _fallback_path(eval_metrics_path)
        self._history_fallback = _fallback_path(history_summary_path)

    def load_eval_metrics(self) -> Dict[str, Any]:
        """Load evaluation metrics, ensuring accuracy is present."""
        metrics = self._load(self.eval_metrics_path) or self._load(self._eval_fallback) or {}
        # Ensure accuracy is available - it might be in classification_report
        if metrics and "accuracy" not in metrics:
            # Try to extract accuracy from classification_report if available
            if "classification_report" in metrics and isinstance(metrics["classification_report"], dict):
                if "accuracy" in metrics["classification_report"]:
                    metrics["accuracy"] = metrics["classification_report"]["accuracy"]
        return metrics

    def load_history_summary(self) -> Dict[str, Any]:
        """Load history summary from database or fallback to JSON file."""
        # Try to generate from database first
        db_summary = self._generate_history_summary_from_db()
        if db_summary and db_summary.get("total_predictions", 0) > 0:
            return db_summary
        
        # Fallback to JSON file if database is empty
        return self._load(self.history_summary_path) or self._load(self._history_fallback) or {}

    def _generate_history_summary_from_db(self) -> Dict[str, Any]:
        """Generate history summary from prediction records in database."""
        db = SessionLocal()
        try:
            # Get all records
            records = db.query(DBPredictionRecord).order_by(DBPredictionRecord.timestamp).all()
            
            if not records:
                return {
                    "total_predictions": 0,
                    "label_counts": {},
                    "date_counts": {},
                    "first_timestamp": None,
                    "last_timestamp": None,
                    "average_text_length": 0,
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                }
            
            # Process records
            label_counter = Counter(record.label for record in records)
            date_counter: Dict[str, int] = defaultdict(int)
            lengths: list[int] = []
            timestamps: list[datetime] = []
            
            for record in records:
                text = record.text or ""
                lengths.append(len(text))
                
                if record.timestamp:
                    timestamps.append(record.timestamp)
                    date_counter[record.timestamp.strftime("%Y-%m-%d")] += 1
                else:
                    date_counter["unknown"] += 1
            
            timestamps.sort()
            
            summary: Dict[str, Any] = {
                "total_predictions": len(records),
                "label_counts": dict(label_counter),
                "date_counts": dict(sorted(date_counter.items())),
                "first_timestamp": timestamps[0].isoformat() if timestamps else None,
                "last_timestamp": timestamps[-1].isoformat() if timestamps else None,
                "average_text_length": round(mean(lengths), 2) if lengths else 0,
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }
            return summary
        finally:
            db.close()

    @staticmethod
    def _load(path: Optional[Path]) -> Optional[Dict[str, Any]]:
        if not path or not path.exists():
            return None
        try:
            with path.open("r", encoding="utf-8") as fh:
                return json.load(fh)
        except (OSError, json.JSONDecodeError):
            return None


def _fallback_path(path: Path) -> Path:
    """Generate fallback path by inserting .sample before the file extension."""
    if path.suffix:
        # For files with extension, insert .sample before the extension
        # e.g., eval_metrics.json -> eval_metrics.sample.json
        return path.with_name(f"{path.stem}.sample{path.suffix}")
    # For files without extension, append .sample.json
    return Path(f"{path}.sample.json")
