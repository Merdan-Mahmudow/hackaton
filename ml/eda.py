"""Quick exploratory data analysis for the sentiment dataset."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd


def compute_stats(df: pd.DataFrame) -> dict:
    lengths = df["text"].str.len()
    word_lengths = df["text"].str.split().str.len()
    label_counts = Counter(df["label"])

    summary = {
        "num_samples": int(len(df)),
        "labels": label_counts,
        "avg_length_chars": float(lengths.mean()),
        "avg_length_words": float(word_lengths.mean()),
        "min_length_chars": int(lengths.min()),
        "max_length_chars": int(lengths.max()),
    }

    most_common_tokens = (
        df["text"].str.lower().str.replace(r"[^\w\s]", "", regex=True).str.split().explode()
    )
    top_tokens = Counter(most_common_tokens.dropna()).most_common(10)
    summary["top_tokens"] = top_tokens
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="EDA for the sentiment dataset")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/sample_reviews.csv"),
        help="Path to CSV file with columns `text` and `label`.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/eda_summary.json"),
        help="Where to save the JSON summary.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("Input CSV must contain `text` and `label` columns")

    summary = compute_stats(df)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
