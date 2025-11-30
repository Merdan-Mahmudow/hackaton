import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from backend.app.stats import StatsTracker


class StatsTrackerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = TemporaryDirectory()
        self.history_path = Path(self.tmp.name) / "history.jsonl"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_records_and_snapshot(self) -> None:
        tracker = StatsTracker(max_history=3, history_path=self.history_path)
        tracker.record("Текст", "neutral", {"negative": 0.1, "neutral": 0.8, "positive": 0.1})
        tracker.record("Это плохо", "negative", {"negative": 0.9, "neutral": 0.05, "positive": 0.05})

        snapshot = tracker.snapshot()
        self.assertEqual(snapshot["total_predictions"], 2)
        self.assertAlmostEqual(snapshot["label_distribution"]["negative"], 0.5)
        self.assertEqual(len(snapshot["recent_predictions"]), 2)

        file_lines = self.history_path.read_text(encoding="utf-8").strip().splitlines()
        labels = [json.loads(line)["label"] for line in file_lines]
        self.assertEqual(len(labels), 2)
        self.assertIn("negative", labels)

    def test_reload_from_history(self) -> None:
        tracker = StatsTracker(max_history=2, history_path=self.history_path)
        tracker.record("ok", "neutral", {"negative": 0.1, "neutral": 0.8, "positive": 0.1})
        tracker.record("great", "positive", {"negative": 0.1, "neutral": 0.1, "positive": 0.8})

        reloaded = StatsTracker(max_history=1, history_path=self.history_path)
        snapshot = reloaded.snapshot()
        self.assertEqual(snapshot["total_predictions"], 2)
        self.assertEqual(len(snapshot["recent_predictions"]), 1)
        self.assertEqual(snapshot["recent_predictions"][0]["label"], "positive")


if __name__ == "__main__":
    unittest.main()
