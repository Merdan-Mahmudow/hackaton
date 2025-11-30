"""Train a simple TF-IDF + Logistic Regression sentiment classifier."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

DATA_PATH = Path("data/train.csv")
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "baseline.joblib"
METADATA_PATH = MODEL_DIR / "metadata.json"


def load_dataset(path: Path) -> Tuple[pd.Series, pd.Series]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("Dataset must contain 'text' and 'label' columns")
    texts = df["text"]
    # Маппинг числовых меток на названия классов: 0 -> neutral, 1 -> positive, 2 -> negative
    label_mapping = {0: "neutral", 1: "positive", 2: "negative"}
    labels = df["label"].map(label_mapping).fillna(df["label"].astype(str))
    return texts, labels


def build_pipeline(max_iter: int = 5000) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=1,
                    max_df=0.95,
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=max_iter,
                    multi_class="auto",
                    class_weight="balanced",
                ),
            ),
        ]
    )


def train(test_size: float = 0.2, random_state: int = 42, max_iter: int = 5000) -> None:
    texts, labels = load_dataset(DATA_PATH)
    X_train, X_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )

    pipeline = build_pipeline(max_iter=max_iter)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    report_dict = classification_report(
        y_test,
        y_pred,
        output_dict=True,
        digits=4,
    )
    print("Classification report:\n")
    print(classification_report(y_test, y_pred, digits=4))

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    metadata = {
        "model_path": str(MODEL_PATH),
        "algorithm": "LogisticRegression",
        "vectorizer": "TfidfVectorizer",
        "classes": sorted(labels.unique().tolist()),
        "test_size": test_size,
        "random_state": random_state,
        "metrics": report_dict,
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(f"\nModel saved to {MODEL_PATH}")
    print(f"Metadata saved to {METADATA_PATH}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-iter", type=int, default=5000, help="Maximum number of iterations for LogisticRegression")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(test_size=args.test_size, random_state=args.random_state, max_iter=args.max_iter)
