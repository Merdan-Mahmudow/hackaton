"""Evaluate a trained sentiment model on a labeled CSV dataset."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

# Add project root to path for backend imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.app.model import SentimentModel

DEFAULT_DATA = Path("data/sample_reviews.csv")
DEFAULT_MODEL = Path("models/baseline.joblib")
DEFAULT_REPORT = Path("reports/eval_metrics.json")


def load_dataset(path: Path, text_column: str, label_column: str) -> Tuple[pd.Series, pd.Series]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path)
    missing_columns = {text_column, label_column} - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"Dataset must contain columns {missing_columns}. Available columns: {sorted(df.columns)}"
        )
    texts = df[text_column].astype(str)
    labels = df[label_column].astype(str)
    return texts, labels


def evaluate_model(
    model_path: Path,
    data_path: Path,
    text_column: str,
    label_column: str,
) -> Dict[str, object]:
    model = SentimentModel(model_path)
    texts, labels = load_dataset(data_path, text_column, label_column)
    predictions = [pred["label"] for pred in model.classify_batch(texts.tolist())]

    report = classification_report(labels, predictions, output_dict=True, digits=4)
    accuracy = accuracy_score(labels, predictions)
    macro_f1 = f1_score(labels, predictions, average="macro")
    labels_sorted = sorted(np.unique(np.concatenate([labels, predictions])))
    matrix = confusion_matrix(labels, predictions, labels=labels_sorted).tolist()

    return {
        "dataset": str(data_path),
        "model": str(model_path),
        "num_records": int(len(labels)),
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "classification_report": report,
        "labels": labels_sorted,
        "confusion_matrix": {
            "labels": labels_sorted,
            "matrix": matrix,
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save the evaluation report to disk (only print to stdout)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = evaluate_model(
        model_path=args.model,
        data_path=args.data,
        text_column=args.text_column,
        label_column=args.label_column,
    )

    print("Evaluation summary:\n")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))

    if not args.no_save:
        report_path = args.report
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
        print(f"\nReport saved to {report_path}")


if __name__ == "__main__":
    main()

