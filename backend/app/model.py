"""Utilities to load and run the sentiment classifier."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List

try:  # pragma: no cover - optional dependency
    import joblib
except ImportError:  # pragma: no cover
    joblib = None

try:  # pragma: no cover - optional dependency
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from transformers.pipelines import TextClassificationPipeline
except ImportError:  # pragma: no cover
    AutoModelForSequenceClassification = None
    AutoTokenizer = None
    TextClassificationPipeline = None


class BaseAdapter:
    classes_: List[str]

    def predict(self, texts: Iterable[str]) -> List[str]:  # pragma: no cover - interface
        raise NotImplementedError

    def predict_proba(self, texts: Iterable[str]):  # pragma: no cover - interface
        raise NotImplementedError


class JoblibAdapter(BaseAdapter):
    def __init__(self, pipeline, metadata_path: Path | None = None) -> None:
        self.pipeline = pipeline
        pipeline_classes = list(getattr(pipeline, "classes_", ["neutral", "positive", "negative"]))
        
        # Try to use classes order from metadata if available
        if metadata_path and metadata_path.exists():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                if "classes" in metadata and isinstance(metadata["classes"], list):
                    # Ensure all pipeline classes are in metadata classes
                    metadata_classes = metadata["classes"]
                    if set(pipeline_classes) == set(metadata_classes):
                        # Use metadata order, but verify indices match
                        self.classes_ = metadata_classes
                    else:
                        self.classes_ = pipeline_classes
                else:
                    self.classes_ = pipeline_classes
            except (json.JSONDecodeError, KeyError):
                self.classes_ = pipeline_classes
        else:
            self.classes_ = pipeline_classes
        
        # Store pipeline classes for reordering if needed
        self.pipeline_classes_ = pipeline_classes

    def predict(self, texts: Iterable[str]) -> List[str]:
        # Use predict_proba to ensure correct class order mapping
        proba = self.predict_proba(texts)
        predictions = []
        for row in proba:
            max_idx = max(range(len(row)), key=lambda idx: row[idx])
            predictions.append(self.classes_[max_idx])
        return predictions

    def predict_proba(self, texts: Iterable[str]):
        proba = self.pipeline.predict_proba(list(texts))
        # Ensure probabilities are in the same order as self.classes_
        if self.pipeline_classes_ != self.classes_:
            # Reorder probabilities to match self.classes_
            reordered_proba = []
            for row in proba:
                reordered_row = [row[self.pipeline_classes_.index(cls)] for cls in self.classes_]
                reordered_proba.append(reordered_row)
            return reordered_proba
        return proba


class KeywordFallbackAdapter(BaseAdapter):
    """Simple keyword-based classifier used until a trained model is available."""

    classes_ = ["neutral", "positive", "negative"]

    positive_keywords = {
        "нравится",
        "спасибо",
        "удобно",
        "люблю",
        "хорошо",
        "стабильно",
        "отремонтирована",
    }
    negative_keywords = {
        "ужас",
        "плохо",
        "вылетает",
        "невозможно",
        "запутался",
        "молчит",
        "устаревшая",
        "ошибка",
        "проблема",
    }

    def predict(self, texts: Iterable[str]) -> List[str]:
        return [self._predict_text(text) for text in texts]

    def predict_proba(self, texts: Iterable[str]):
        return [self._scores(text) for text in texts]

    def _predict_text(self, text: str) -> str:
        probs = self._scores(text)
        max_index = max(range(len(probs)), key=lambda idx: probs[idx])
        return self.classes_[max_index]

    def _scores(self, text: str) -> List[float]:
        text_lower = text.lower()
        pos_hits = sum(word in text_lower for word in self.positive_keywords)
        neg_hits = sum(word in text_lower for word in self.negative_keywords)
        total = pos_hits + neg_hits
        if total == 0:
            # Порядок: neutral, positive, negative
            return [0.6, 0.2, 0.2]
        neg_score = neg_hits / total
        pos_score = pos_hits / total
        neu_score = max(0.0, 1.0 - (neg_score + pos_score) / 2)
        # Порядок: neutral, positive, negative
        scores = [neu_score, pos_score, neg_score]
        total_score = sum(scores)
        if total_score == 0:
            return [1 / 3, 1 / 3, 1 / 3]
        return [score / total_score for score in scores]


class TransformerAdapter(BaseAdapter):
    def __init__(self, model_dir: Path) -> None:
        if AutoTokenizer is None or AutoModelForSequenceClassification is None:
            raise ImportError("transformers is required to load transformer models")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        if TextClassificationPipeline is None:  # pragma: no cover
            raise ImportError("transformers pipelines are required")
        self.pipeline = TextClassificationPipeline(
            model=self.model,
            tokenizer=self.tokenizer,
            function_to_apply="softmax",
            top_k=None,
        )
        # Сначала пытаемся загрузить классы из метаданных
        metadata_path = model_dir / "metadata.json"
        if metadata_path.exists():
            try:
                import json
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                if "classes" in metadata and isinstance(metadata["classes"], list):
                    self.classes_ = metadata["classes"]
                elif "id2label" in metadata:
                    # Используем id2label из метаданных
                    id2label = metadata["id2label"]
                    ordered_keys = sorted(id2label.keys(), key=lambda k: int(k) if isinstance(k, (int, str)) and str(k).isdigit() else 0)
                    self.classes_ = [id2label[key] for key in ordered_keys]
                else:
                    self.classes_ = self._get_classes_from_config()
            except (json.JSONDecodeError, KeyError, ValueError):
                self.classes_ = self._get_classes_from_config()
        else:
            self.classes_ = self._get_classes_from_config()
    
    def _get_classes_from_config(self) -> List[str]:
        """Получить классы из конфигурации модели."""
        if hasattr(self.model.config, "id2label"):
            labels_map = self.model.config.id2label
            try:
                ordered_keys = sorted(labels_map.keys(), key=lambda key: int(key))
            except (TypeError, ValueError):
                ordered_keys = sorted(labels_map.keys())
            return [labels_map[key] for key in ordered_keys]
        else:
            return [f"LABEL_{idx}" for idx in range(self.model.config.num_labels)]

    def predict(self, texts: Iterable[str]) -> List[str]:
        proba = self.predict_proba(texts)
        predictions = []
        for row in proba:
            max_index = max(range(len(row)), key=lambda idx: row[idx])
            predictions.append(self.classes_[max_index])
        return predictions

    def predict_proba(self, texts: Iterable[str]):
        outputs = self.pipeline(
            list(texts),
            truncation=True,
            padding=True,
            top_k=None,  # Get all scores instead of deprecated return_all_scores
        )
        proba_rows = []
        for sample_scores in outputs:
            mapping = {item["label"]: float(item["score"]) for item in sample_scores}
            row = [mapping.get(label, 0.0) for label in self.classes_]
            total = sum(row)
            if total:
                row = [val / total for val in row]
            proba_rows.append(row)
        return proba_rows


class SentimentModel:
    """Wrapper around a trained pipeline with a rule-based fallback."""

    def __init__(self, model_path: Path, metadata_path: Path | None = None):
        self.model_path = model_path
        if metadata_path is not None:
            self.metadata_path = metadata_path
        elif model_path.is_dir():
            self.metadata_path = model_path / "metadata.json"
        else:
            self.metadata_path = model_path.with_name("metadata.json")
        self.adapter = self._build_adapter(model_path)
        self.labels: List[str] = list(
            getattr(self.adapter, "classes_", ["neutral", "positive", "negative"])
        )
        self.metadata = self._load_metadata()

    def _build_adapter(self, model_path: Path) -> BaseAdapter:
        if model_path.is_dir() and (model_path / "config.json").exists():
            try:
                return TransformerAdapter(model_path)
            except Exception:  # pragma: no cover - fallback handled gracefully
                pass
        if joblib is not None and model_path.exists():
            try:
                pipeline = joblib.load(model_path)
                return JoblibAdapter(pipeline, metadata_path=self.metadata_path)
            except Exception:  # pragma: no cover - fallback handled below
                pass
        return KeywordFallbackAdapter()

    def _load_metadata(self) -> Dict[str, object]:
        if not self.metadata_path.exists():
            return {
                "model_path": str(self.model_path),
                "algorithm": getattr(self.adapter, "__class__", type(self.adapter)).__name__,
                "classes": self.labels,
            }
        try:
            return json.loads(self.metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {"model_path": str(self.model_path), "classes": self.labels}

    def predict(self, text: str) -> Dict[str, float]:
        """Return class probabilities for a single text."""
        proba = self.adapter.predict_proba([text])[0]
        return {label: float(score) for label, score in zip(self.labels, proba)}

    def predict_label(self, text: str) -> str:
        return self.adapter.predict([text])[0]

    def classify(self, text: str) -> Dict[str, object]:
        probabilities = self.predict(text)
        label = max(probabilities.items(), key=lambda pair: pair[1])[0]
        return {"label": label, "scores": probabilities}

    def classify_batch(self, texts: List[str]) -> List[Dict[str, object]]:
        proba = self.adapter.predict_proba(texts)
        results = []
        for row in proba:
            # Find index with maximum probability
            max_idx = max(range(len(row)), key=lambda idx: row[idx])
            label = self.labels[max_idx]
            scores = {cls: float(score) for cls, score in zip(self.labels, row)}
            results.append({"label": label, "scores": scores})
        return results

    def get_model_name(self) -> str:
        """Возвращает название модели для отображения."""
        # Для трансформера
        if "base_model" in self.metadata:
            base_model = self.metadata.get("base_model", "")
            model_type = self.metadata.get("model_type", "transformer")
            if base_model:
                # Извлекаем короткое название из полного пути модели
                model_short = base_model.split("/")[-1] if "/" in base_model else base_model
                return f"{model_type} ({model_short})"
            return model_type
        
        # Для baseline модели
        if "algorithm" in self.metadata:
            algorithm = self.metadata.get("algorithm", "baseline")
            vectorizer = self.metadata.get("vectorizer", "")
            if vectorizer:
                return f"{algorithm} + {vectorizer}"
            return algorithm
        
        # Для fallback модели
        adapter_name = getattr(self.adapter, "__class__", type(self.adapter)).__name__
        if "Fallback" in adapter_name:
            return "KeywordFallbackModel"
        
        # По умолчанию
        return adapter_name.replace("Adapter", "")
