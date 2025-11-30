"""Pydantic schemas for the API."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, constr


class PredictRequest(BaseModel):
    text: constr(strip_whitespace=True, min_length=1, max_length=2000) = Field(
        ..., description="Русскоязычный текст обращения"
    )


class EntityInfo(BaseModel):
    """Информация об извлеченных сущностях."""
    persons: List[str] = Field(default_factory=list, description="Список найденных имен людей")
    organizations: List[str] = Field(default_factory=list, description="Список найденных названий организаций")
    locations: List[str] = Field(default_factory=list, description="Список найденных названий мест")
    from_person: Optional[str] = Field(None, description="Имя отправителя обращения")
    to_organization: Optional[str] = Field(None, description="Название организации-получателя")
    to_person: Optional[str] = Field(None, description="Имя получателя обращения")


class PredictResponse(BaseModel):
    label: str = Field(..., description="Предсказанный класс тональности")
    scores: Dict[str, float] = Field(..., description="Вероятности по каждому классу")
    entities: Optional[EntityInfo] = Field(None, description="Извлеченные имена и названия компаний")
    model_name: Optional[str] = Field(None, description="Название использованной модели")


class BatchPredictRequest(BaseModel):
    texts: List[constr(strip_whitespace=True, min_length=1, max_length=2000)] = Field(
        ..., description="Список отзывов"
    )


class BatchPredictResponse(BaseModel):
    predictions: List[PredictResponse]


class FeedbackRequest(BaseModel):
    text: constr(strip_whitespace=True, min_length=1, max_length=2000)
    predicted_label: constr(strip_whitespace=True, min_length=1)
    user_label: Optional[constr(strip_whitespace=True, min_length=1)] = None
    scores: Optional[Dict[str, float]] = None
    notes: Optional[constr(strip_whitespace=True, min_length=1, max_length=500)] = None


class PredictionHistoryItem(BaseModel):
    text: str
    label: str
    scores: Dict[str, float]
    timestamp: str


class StatsResponse(BaseModel):
    total_predictions: int
    label_distribution: Dict[str, float]
    recent_predictions: List[PredictionHistoryItem]


class FeedbackItem(BaseModel):
    text: str
    predicted_label: str
    user_label: Optional[str]
    scores: Optional[Dict[str, float]] = None
    notes: Optional[str] = None
    timestamp: str


class FeedbackResponse(BaseModel):
    status: str
    entry: FeedbackItem


class FeedbackListResponse(BaseModel):
    total_items: int
    items: List[FeedbackItem]


class ModelInfoResponse(BaseModel):
    model_path: str
    algorithm: Optional[str] = None
    vectorizer: Optional[str] = None
    classes: List[str]
    metrics: Optional[Dict[str, Any]] = None
    test_size: Optional[float] = None
    random_state: Optional[int] = None


class FilePrediction(BaseModel):
    row: int = Field(..., ge=0, description="Индекс строки в исходном файле")
    text: str = Field(..., description="Текст обращения")
    label: str = Field(..., description="Предсказанная тональность")
    scores: Dict[str, float] = Field(..., description="Вероятности по классам")
    entities: Optional[EntityInfo] = Field(None, description="Извлеченные имена и названия компаний")
    model_name: Optional[str] = Field(None, description="Название использованной модели")


class FilePredictionSummary(BaseModel):
    input_rows: int = Field(..., description="Количество строк в CSV")
    processed_rows: int = Field(..., description="Сколько строк обработано")
    skipped_rows: int = Field(..., description="Сколько строк пропущено")
    class_counts: Dict[str, int] = Field(
        ..., description="Количество предсказаний по классам"
    )


class FilePredictResponse(BaseModel):
    summary: FilePredictionSummary
    predictions: List[FilePrediction]


class ConfusionMatrixPayload(BaseModel):
    labels: List[str]
    matrix: List[List[int]]


class EvalMetricsResponse(BaseModel):
    dataset: str
    model: str
    num_records: int
    accuracy: float
    macro_f1: float
    classification_report: Dict[str, Any]  # Can contain Dict[str, float] for classes or float for accuracy/macro avg
    labels: List[str]
    confusion_matrix: ConfusionMatrixPayload
    generated_at: Optional[str] = None


class HistorySummaryResponse(BaseModel):
    total_predictions: int
    label_counts: Dict[str, int]
    date_counts: Dict[str, int]
    first_timestamp: Optional[str]
    last_timestamp: Optional[str]
    average_text_length: float
    generated_at: Optional[str] = None
