"""Utilities for tracking prediction statistics with optional persistence."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Dict, Optional

from sqlalchemy import func
from sqlalchemy.orm import Session

from .database import PredictionRecord as DBPredictionRecord, SessionLocal


class PredictionRecord:
    """Data class for prediction record (for backward compatibility)."""
    def __init__(
        self,
        text: str,
        label: str,
        scores: Dict[str, float],
        timestamp: Optional[str] = None,
    ) -> None:
        self.text = text
        self.label = label
        self.scores = scores
        self.timestamp = timestamp or datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, object]:
        return {
            "text": self.text,
            "label": self.label,
            "scores": self.scores,
            "timestamp": self.timestamp,
        }


class StatsTracker:
    """Thread-safe tracker that stores aggregate stats in SQLite database."""

    def __init__(self, max_history: int = 50, history_path: Optional[Path] = None) -> None:
        """Initialize stats tracker.
        
        Args:
            max_history: Maximum number of recent predictions to keep in memory.
            history_path: Deprecated, kept for backward compatibility. Database path is now in config.
        """
        self.max_history = max_history
        self._lock = Lock()
        self._default_labels = ["negative", "neutral", "positive"]

    def _get_session(self) -> Session:
        """Get database session."""
        return SessionLocal()

    def record(self, text: str, label: str, scores: Dict[str, float]) -> PredictionRecord:
        """Record a new prediction."""
        truncated_text = _truncate_text(text)
        timestamp = datetime.now(timezone.utc)
        
        with self._lock:
            db = self._get_session()
            try:
                db_record = DBPredictionRecord(
                    text=truncated_text,
                    label=label,
                    scores=scores,
                    timestamp=timestamp,
                )
                db.add(db_record)
                db.commit()
                db.refresh(db_record)
                
                # Convert to legacy PredictionRecord format
                record = PredictionRecord(
                    text=db_record.text,
                    label=db_record.label,
                    scores=db_record.scores,
                    timestamp=db_record.timestamp.isoformat() if db_record.timestamp else "",
                )
                return record
            finally:
                db.close()

    def snapshot(self) -> Dict[str, object]:
        """Get current statistics snapshot."""
        db = self._get_session()
        try:
            # Get total count
            total = db.query(DBPredictionRecord).count()
            
            # Get label distribution
            label_counts = (
                db.query(DBPredictionRecord.label, func.count(DBPredictionRecord.id))
                .group_by(DBPredictionRecord.label)
                .all()
            )
            counts = Counter({label: count for label, count in label_counts})
            
            labels = sorted(counts.keys()) or self._default_labels
            distribution = {
                label: (counts[label] / total if total else 0.0)
                for label in labels
            }
            
            # Get recent predictions
            recent_records = (
                db.query(DBPredictionRecord)
                .order_by(DBPredictionRecord.timestamp.desc())
                .limit(self.max_history)
                .all()
            )
            history = [record.to_dict() for record in recent_records]
            
            return {
                "total_predictions": total,
                "label_distribution": distribution,
                "recent_predictions": history,
            }
        finally:
            db.close()

    def reset(self) -> None:
        """Reset all statistics (delete all records)."""
        db = self._get_session()
        try:
            db.query(DBPredictionRecord).delete()
            db.commit()
        finally:
            db.close()


def _truncate_text(text: str, max_length: int = 240) -> str:
    text = text.strip()
    if len(text) <= max_length:
        return text
    return text[: max_length - 1].rstrip() + "…"
