"""Fine-tune a Hugging Face transformer on the sentiment dataset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

DATA_PATH = Path("data/train.csv")
OUTPUT_DIR = Path("models/transformer")
REPORT_PATH = Path("reports/transformer_metrics.json")
DEFAULT_MODEL = "cointegrated/rubert-tiny"


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
    # Маппинг числовых меток на названия классов: 0 -> neutral, 1 -> positive, 2 -> negative
    label_mapping = {0: "neutral", 1: "positive", 2: "negative"}
    labels = df[label_column].map(label_mapping).fillna(df[label_column].astype(str))
    return texts, labels


def prepare_datasets(
    texts: pd.Series,
    labels: pd.Series,
    test_size: float,
    random_state: int,
) -> Tuple[Dataset, Dataset, Dict[str, int]]:
    # Убеждаемся, что классы в правильном порядке: neutral, positive, negative
    expected_classes = ["neutral", "positive", "negative"]
    unique_labels = set(labels.unique().tolist())
    
    # Проверяем наличие всех ожидаемых классов
    for cls in unique_labels:
        if cls not in expected_classes:
            print(f"Предупреждение: найден неожиданный класс '{cls}'")
    
    # Создаем маппинг с фиксированным порядком (neutral=0, positive=1, negative=2)
    label2id = {}
    for idx, cls in enumerate(expected_classes):
        if cls in unique_labels:
            label2id[cls] = idx
    
    # Если есть дополнительные классы, добавляем их в конец
    for cls in unique_labels:
        if cls not in label2id:
            label2id[cls] = len(label2id)
            print(f"Добавлен дополнительный класс '{cls}' с ID {label2id[cls]}")
    
    df = pd.DataFrame({"text": texts, "label": labels.map(label2id)})
    train_df, val_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df["label"],
        random_state=random_state,
    )
    train_dataset = Dataset.from_pandas(train_df, preserve_index=False)
    val_dataset = Dataset.from_pandas(val_df, preserve_index=False)
    return train_dataset, val_dataset, label2id


def tokenize_dataset(
    dataset: Dataset,
    tokenizer: AutoTokenizer,
    max_length: int,
) -> Dataset:
    def tokenize(batch: Dict[str, list[str]]):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )

    tokenized = dataset.map(tokenize, batched=True)
    tokenized = tokenized.remove_columns(["text"])
    tokenized.set_format("torch")
    return tokenized


def compute_metrics(eval_pred):
    from evaluate import load as load_metric
    from sklearn.metrics import f1_score

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy_metric = load_metric("accuracy")
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"]
    macro_f1 = f1_score(labels, predictions, average="macro")
    return {"accuracy": accuracy, "macro_f1": macro_f1}


def train_transformer(args: argparse.Namespace) -> Dict[str, object]:
    texts, labels = load_dataset(args.data, args.text_column, args.label_column)
    train_dataset, val_dataset, label2id = prepare_datasets(
        texts,
        labels,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_tokenized = tokenize_dataset(train_dataset, tokenizer, args.max_length)
    val_tokenized = tokenize_dataset(val_dataset, tokenizer, args.max_length)

    id2label = {idx: label for label, idx in label2id.items()}
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
    )

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    eval_metrics = trainer.evaluate()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    metadata = {
        "model_type": "transformer",
        "base_model": args.model_name,
        "model_dir": str(args.output_dir),
        "classes": list(label2id.keys()),
        "label2id": label2id,
        "id2label": {idx: label for label, idx in label2id.items()},
        "max_length": args.max_length,
        "metrics": eval_metrics,
        "test_size": args.test_size,
        "random_state": args.random_state,
    }
    metadata_path = args.output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(eval_metrics, indent=2, ensure_ascii=False))

    print("Training complete. Metrics:\n")
    print(json.dumps(eval_metrics, indent=2, ensure_ascii=False))
    print(f"\nModel saved to {args.output_dir}")
    print(f"Metadata saved to {metadata_path}")
    print(f"Evaluation report saved to {REPORT_PATH}")

    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    train_transformer(cli_args)
