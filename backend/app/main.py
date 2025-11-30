"""FastAPI service for the sentiment classifier."""
from __future__ import annotations

import io
import logging
from collections import Counter

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .config import settings
from .database import init_db
from .entity_extractor import get_entity_extractor
from .feedback import FeedbackStore
from .model import SentimentModel
from .reports import ReportLoader
from .schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    EntityInfo,
    EvalMetricsResponse,
    FilePredictResponse,
    FeedbackListResponse,
    FeedbackRequest,
    FeedbackResponse,
    HistorySummaryResponse,
    ModelInfoResponse,
    PredictRequest,
    PredictResponse,
    StatsResponse,
)
from .stats import StatsTracker

MAX_FILE_RECORDS = settings.max_file_records

logging.basicConfig(level=logging.INFO)
app = FastAPI(title="ML-Web Sentiment API", version="1.0.0")
logger = logging.getLogger(__name__)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if settings.frontend_dir.exists():
    app.mount("/ui", StaticFiles(directory=settings.frontend_dir, html=True), name="ui")

sentiment_model: SentimentModel | None = None
stats_tracker = StatsTracker(
    max_history=settings.stats_max_history, history_path=settings.history_path
)
feedback_store = FeedbackStore(settings.feedback_path, cache_size=200)
report_loader = ReportLoader(
    eval_metrics_path=settings.eval_metrics_path,
    history_summary_path=settings.history_summary_path,
)


@app.on_event("startup")
def startup() -> None:
    """Initialize database and load model on startup."""
    # Initialize database
    init_db()
    logger.info("Database initialized")
    
    # Load model
    global sentiment_model
    # Prioritize transformer model if enabled, fallback to baseline
    if settings.use_transformer and settings.transformer_dir.exists() and (settings.transformer_dir / "config.json").exists():
        target_path = settings.transformer_dir
        logger.info("Loading transformer model from %s", target_path)
        sentiment_model = SentimentModel(target_path)
    elif settings.model_path.exists():
        target_path = settings.model_path
        logger.info("Loading baseline model from %s", target_path)
        sentiment_model = SentimentModel(target_path)
    else:
        logger.warning(
            "No model found. Transformer dir: %s, Baseline: %s. Using KeywordFallbackModel.",
            settings.transformer_dir,
            settings.model_path,
        )
        sentiment_model = SentimentModel(settings.model_path)


@app.get("/health")
def healthcheck() -> dict:
    return {"status": "ok"}


def _require_model() -> SentimentModel:
    if sentiment_model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded")
    return sentiment_model


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    """Predict sentiment for a single text using the trained model."""
    model = _require_model()
    try:
        result = model.classify(request.text)
        
        # Извлекаем имена и компании
        entity_extractor = get_entity_extractor()
        entities_full = entity_extractor.extract(request.text)
        entities_simple = entity_extractor.extract_entities_simple(request.text)
        
        entity_info = EntityInfo(
            organizations=entities_full["organizations"],
            locations=entities_full["locations"],
            to_organization=entities_simple["to_organization"],
            to_person=entities_simple["to_person"],
        )
        
        model_name = model.get_model_name()
        
        logger.debug(
            "Prediction for text (length=%d): label=%s, scores=%s, entities=%s, model=%s",
            len(request.text),
            result["label"],
            result["scores"],
            entity_info.dict(),
            model_name,
        )
        stats_tracker.record(request.text, result["label"], result["scores"])
        return PredictResponse(**result, model_name=model_name)
    except Exception as exc:
        logger.error("Error during prediction: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка при предсказании тональности: {str(exc)}",
        ) from exc


@app.post("/predict_batch", response_model=BatchPredictResponse)
def predict_batch(request: BatchPredictRequest) -> BatchPredictResponse:
    """Predict sentiment for multiple texts in batch."""
    model = _require_model()
    try:
        predictions = model.classify_batch(request.texts)
        
        # Извлекаем сущности для каждого текста
        entity_extractor = get_entity_extractor()
        model_name = model.get_model_name()
        response_predictions = []
        for text, pred in zip(request.texts, predictions):
            entities_full = entity_extractor.extract(text)
            entities_simple = entity_extractor.extract_entities_simple(text)
            
            entity_info = EntityInfo(
                persons=entities_full["persons"],
                organizations=entities_full["organizations"],
                locations=entities_full["locations"],
                from_person=entities_simple["from_person"],
                to_organization=entities_simple["to_organization"],
                to_person=entities_simple["to_person"],
            )
            
            response_predictions.append(PredictResponse(**pred, entities=entity_info, model_name=model_name))
            stats_tracker.record(text, pred["label"], pred["scores"])
        
        logger.debug(
            "Batch prediction: %d texts processed, results: %s",
            len(request.texts),
            [p["label"] for p in predictions],
        )
        return BatchPredictResponse(predictions=response_predictions)
    except Exception as exc:
        logger.error("Error during batch prediction: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка при пакетном предсказании тональности: {str(exc)}",
        ) from exc


@app.post("/predict_file", response_model=FilePredictResponse)
async def predict_file(file: UploadFile = File(...)) -> FilePredictResponse:
    """Обработка CSV с колонкой text."""

    try:
        content = await file.read()
    except Exception as exc:  # pragma: no cover - FastAPI handles IO
        raise HTTPException(status_code=400, detail="Не удалось прочитать файл") from exc

    if not content:
        raise HTTPException(status_code=400, detail="Файл пустой")

    try:
        decoded = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=400, detail="CSV должен быть в кодировке UTF-8") from exc

    try:
        dataframe = pd.read_csv(io.StringIO(decoded))
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Не удалось распарсить CSV: {exc}",
        ) from exc

    if "text" not in dataframe.columns:
        raise HTTPException(status_code=400, detail="CSV должен содержать колонку 'text'")

    input_rows = len(dataframe)
    dataframe = dataframe.dropna(subset=["text"])
    if dataframe.empty:
        raise HTTPException(status_code=400, detail="В колонке 'text' нет строк для обработки")

    if len(dataframe) > MAX_FILE_RECORDS:
        dataframe = dataframe.head(MAX_FILE_RECORDS)

    model = _require_model()
    texts = dataframe["text"].astype(str).tolist()
    raw_predictions = model.classify_batch(texts)

    # Извлекаем сущности для каждого текста
    entity_extractor = get_entity_extractor()
    model_name = model.get_model_name()
    items = []
    for idx, text, pred in zip(dataframe.index.tolist(), texts, raw_predictions):
        entities_full = entity_extractor.extract(text)
        entities_simple = entity_extractor.extract_entities_simple(text)
        
        entity_info = EntityInfo(
            persons=entities_full["persons"],
            organizations=entities_full["organizations"],
            locations=entities_full["locations"],
            from_person=entities_simple["from_person"],
            to_organization=entities_simple["to_organization"],
            to_person=entities_simple["to_person"],
        )
        
        stats_tracker.record(text, pred["label"], pred["scores"])
        items.append({
            "row": int(idx),
            "text": text,
            "label": pred["label"],
            "scores": pred["scores"],
            "entities": entity_info,
            "model_name": model_name,
        })

    class_counts = Counter(item["label"] for item in items)
    summary = {
        "input_rows": int(input_rows),
        "processed_rows": len(items),
        "skipped_rows": max(0, int(input_rows) - len(items)),
        "class_counts": dict(class_counts),
    }

    return FilePredictResponse(summary=summary, predictions=items)


@app.get("/")
def root() -> dict:
    return {
        "message": "Добро пожаловать в сервис анализа тональности",
        "endpoints": [
            "/predict",
            "/predict_batch",
            "/predict_file",
            "/stats",
            "/model",
            "/reports/metrics",
            "/reports/history",
            "/health",
        ],
    }


@app.get("/stats", response_model=StatsResponse)
def stats() -> StatsResponse:
    summary = stats_tracker.snapshot()
    return StatsResponse(**summary)


@app.get("/model", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    """Возвращает информацию о загруженной модели."""
    model = _require_model()
    metadata = model.metadata
    
    
    model_path = metadata.get("model_path") or metadata.get("model_dir") or str(model.model_path)
    
    
    algorithm = metadata.get("algorithm")
    if not algorithm and metadata.get("model_type") == "transformer":
        base_model = metadata.get("base_model", "")
        if base_model:
            
            algorithm = base_model.split("/")[-1] if "/" in base_model else base_model
        else:
            algorithm = "transformer"
    
    
    if not algorithm:
        adapter_name = getattr(model.adapter, "__class__", type(model.adapter)).__name__
        if "Fallback" in adapter_name:
            algorithm = "KeywordFallback"
        elif "Transformer" in adapter_name:
            algorithm = "Transformer"
        elif "Joblib" in adapter_name:
            algorithm = metadata.get("algorithm", "Baseline")
    
    vectorizer = metadata.get("vectorizer")
    classes = metadata.get("classes", model.labels)
    metrics = metadata.get("metrics")
    test_size = metadata.get("test_size")
    random_state = metadata.get("random_state")
    
    return ModelInfoResponse(
        model_path=model_path,
        algorithm=algorithm,
        vectorizer=vectorizer,
        classes=classes,
        metrics=metrics,
        test_size=test_size,
        random_state=random_state,
    )


@app.get("/reports/metrics", response_model=EvalMetricsResponse)
def evaluation_report() -> EvalMetricsResponse:
    payload = report_loader.load_eval_metrics()
    if not payload:
        raise HTTPException(
            status_code=404,
            detail="Метрики ещё не сгенерированы. Запустите make evaluate после обучения модели.",
        )
    return EvalMetricsResponse(**payload)


@app.get("/reports/history", response_model=HistorySummaryResponse)
def history_report() -> HistorySummaryResponse:
    payload = report_loader.load_history_summary()
    if not payload:
        raise HTTPException(
            status_code=404,
            detail="Нет агрегированного отчёта. Выполните make history-report для генерации.",
        )
    return HistorySummaryResponse(**payload)


@app.post("/feedback", response_model=FeedbackResponse)
def submit_feedback(request: FeedbackRequest) -> FeedbackResponse:
    entry = feedback_store.append(
        text=request.text,
        predicted_label=request.predicted_label,
        user_label=request.user_label,
        scores=request.scores,
        notes=request.notes,
    )
    return FeedbackResponse(status="stored", entry=entry.to_dict())


@app.get("/feedback", response_model=FeedbackListResponse)
def list_feedback(limit: int = 50) -> FeedbackListResponse:
    limit = max(1, min(limit, 200))
    return FeedbackListResponse(
        total_items=feedback_store.count(), items=feedback_store.recent(limit)
    )
