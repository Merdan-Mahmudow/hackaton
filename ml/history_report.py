"""Generate a summary report from the prediction history log."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/prediction_history.jsonl"),
        help="Path to the JSONL file produced by the API stats tracker.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/history_summary.json"),
        help="Where to save the aggregated metrics.",
    )
    return parser.parse_args()


def load_records(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        return []
    records: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def summarize(records: List[Dict[str, object]]) -> Dict[str, object]:
    if not records:
        return {
            "total_predictions": 0,
            "label_counts": {},
            "date_counts": {},
            "first_timestamp": None,
            "last_timestamp": None,
            "average_text_length": 0,
        }

    label_counter = Counter(record.get("label", "unknown") for record in records)
    date_counter: Dict[str, int] = defaultdict(int)
    lengths: List[int] = []
    timestamps: List[datetime] = []

    for record in records:
        text = str(record.get("text", ""))
        lengths.append(len(text))
        timestamp_raw = record.get("timestamp")
        parsed = _parse_timestamp(timestamp_raw)
        if parsed:
            timestamps.append(parsed)
            date_counter[parsed.strftime("%Y-%m-%d")] += 1
        else:
            date_counter["unknown"] += 1

    timestamps.sort()
    summary: Dict[str, object] = {
        "total_predictions": len(records),
        "label_counts": dict(label_counter),
        "date_counts": dict(sorted(date_counter.items())),
        "first_timestamp": timestamps[0].isoformat() if timestamps else None,
        "last_timestamp": timestamps[-1].isoformat() if timestamps else None,
        "average_text_length": round(mean(lengths), 2) if lengths else 0,
    }
    return summary


def _parse_timestamp(raw: Optional[str]) -> Optional[datetime]:
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None


def main() -> None:
    args = parse_args()
    records = load_records(args.input)
    summary = summarize(records)
    summary["generated_at"] = datetime.now(timezone.utc).isoformat()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
