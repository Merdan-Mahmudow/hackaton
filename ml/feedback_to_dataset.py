"""Convert collected feedback JSONL into a CSV dataset for retraining."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import List


def load_feedback(path: Path) -> List[dict]:
    if not path.exists():
        return []
    entries: List[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


def to_rows(entries: List[dict]) -> List[dict]:
    rows = []
    for entry in entries:
        text = (entry.get("text") or "").strip()
        label = entry.get("user_label") or entry.get("predicted_label")
        if not text or not label:
            continue
        rows.append({"text": text, "label": label})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/feedback.jsonl"),
        help="Path to the feedback JSONL file produced by the API.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/feedback_dataset.csv"),
        help="Where to store the resulting CSV dataset.",
    )
    args = parser.parse_args()

    entries = load_feedback(args.input)
    if not entries:
        print("No feedback entries found. Nothing to export.")
        return

    rows = to_rows(entries)
    if not rows:
        print("Feedback entries did not contain usable text/labels.")
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["text", "label"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Exported {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
