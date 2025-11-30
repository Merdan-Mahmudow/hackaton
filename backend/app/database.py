"""Database configuration and models."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

from sqlalchemy import JSON, Column, DateTime, Float, Integer, String, Text, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from .config import settings

Base = declarative_base()


class FeedbackEntry(Base):
    """SQLAlchemy model for feedback entries."""
    __tablename__ = "feedback_entries"

    id = Column(Integer, primary_key=True, index=True)
    text = Column(Text, nullable=False)
    predicted_label = Column(String(50), nullable=False)
    user_label = Column(String(50), nullable=True)
    scores = Column(JSON, nullable=True)
    notes = Column(Text, nullable=True)
    timestamp = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc), server_default=None)

    def to_dict(self) -> Dict[str, object]:
        """Convert to dictionary."""
        return {
            "text": self.text,
            "predicted_label": self.predicted_label,
            "user_label": self.user_label,
            "scores": self.scores,
            "notes": self.notes,
            "timestamp": self.timestamp.isoformat() if self.timestamp else "",
        }


class PredictionRecord(Base):
    """SQLAlchemy model for prediction records."""
    __tablename__ = "prediction_records"

    id = Column(Integer, primary_key=True, index=True)
    text = Column(Text, nullable=False)
    label = Column(String(50), nullable=False)
    scores = Column(JSON, nullable=False)
    timestamp = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc), server_default=None)

    def to_dict(self) -> Dict[str, object]:
        """Convert to dictionary."""
        return {
            "text": self.text,
            "label": self.label,
            "scores": self.scores,
            "timestamp": self.timestamp.isoformat() if self.timestamp else "",
        }


# Database setup
def get_database_url() -> str:
    """Get database URL from settings."""
    db_path = getattr(settings, "database_path", Path("data/app.db"))
    if isinstance(db_path, str):
        db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{db_path.absolute()}"


engine = create_engine(
    get_database_url(),
    connect_args={"check_same_thread": False},  # Needed for SQLite
    echo=False,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db() -> None:
    """Initialize database tables."""
    Base.metadata.create_all(bind=engine)


def get_db():
    """Dependency for getting database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

