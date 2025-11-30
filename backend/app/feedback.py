"""Feedback storage utilities for active learning."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional

from sqlalchemy.orm import Session

from .database import FeedbackEntry as DBFeedbackEntry, SessionLocal


class FeedbackEntry:
    """Data class for feedback entry (for backward compatibility)."""
    def __init__(
        self,
        text: str,
        predicted_label: str,
        user_label: Optional[str] = None,
        scores: Optional[Dict[str, float]] = None,
        notes: Optional[str] = None,
        timestamp: Optional[str] = None,
    ) -> None:
        self.text = text
        self.predicted_label = predicted_label
        self.user_label = user_label
        self.scores = scores
        self.notes = notes
        self.timestamp = timestamp or datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, object]:
        return {
            "text": self.text,
            "predicted_label": self.predicted_label,
            "user_label": self.user_label,
            "scores": self.scores,
            "notes": self.notes,
            "timestamp": self.timestamp,
        }


class FeedbackStore:
    """Thread-safe store backed by SQLite database."""

    def __init__(self, path: Path, cache_size: int = 200) -> None:
        """Initialize feedback store.
        
        Args:
            path: Deprecated, kept for backward compatibility. Database path is now in config.
            cache_size: Deprecated, kept for backward compatibility.
        """
        self._lock = Lock()

    def _get_session(self) -> Session:
        """Get database session."""
        return SessionLocal()

    def append(
        self,
        *,
        text: str,
        predicted_label: str,
        user_label: Optional[str] = None,
        scores: Optional[Dict[str, float]] = None,
        notes: Optional[str] = None,
    ) -> FeedbackEntry:
        """Add a new feedback entry."""
        with self._lock:
            db = self._get_session()
            try:
                db_entry = DBFeedbackEntry(
                    text=text.strip(),
                    predicted_label=predicted_label,
                    user_label=user_label,
                    scores=scores,
                    notes=notes.strip() if notes else None,
                    timestamp=datetime.now(timezone.utc),
                )
                db.add(db_entry)
                db.commit()
                db.refresh(db_entry)
                
                # Convert to legacy FeedbackEntry format
                entry = FeedbackEntry(
                    text=db_entry.text,
                    predicted_label=db_entry.predicted_label,
                    user_label=db_entry.user_label,
                    scores=db_entry.scores,
                    notes=db_entry.notes,
                    timestamp=db_entry.timestamp.isoformat() if db_entry.timestamp else "",
                )
                return entry
            finally:
                db.close()

    def recent(self, limit: Optional[int] = None) -> List[Dict[str, object]]:
        """Get recent feedback entries."""
        db = self._get_session()
        try:
            query = db.query(DBFeedbackEntry).order_by(DBFeedbackEntry.timestamp.desc())
            if limit is not None:
                query = query.limit(limit)
            entries = query.all()
            return [entry.to_dict() for entry in entries]
        finally:
            db.close()

    def count(self) -> int:
        """Get total count of feedback entries."""
        db = self._get_session()
        try:
            return db.query(DBFeedbackEntry).count()
        finally:
            db.close()
