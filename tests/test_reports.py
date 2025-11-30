import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from backend.app.reports import ReportLoader


class ReportLoaderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = TemporaryDirectory()
        self.base = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_loads_direct_reports(self) -> None:
        eval_path = self.base / "metrics.json"
        history_path = self.base / "history.json"
        eval_path.write_text(json.dumps({"accuracy": 0.9}), encoding="utf-8")
        history_path.write_text(json.dumps({"total_predictions": 5}), encoding="utf-8")

        loader = ReportLoader(eval_path, history_path)
        self.assertEqual(loader.load_eval_metrics()["accuracy"], 0.9)
        self.assertEqual(loader.load_history_summary()["total_predictions"], 5)

    def test_fallback_is_used_when_primary_missing(self) -> None:
        eval_path = self.base / "missing_eval.json"
        history_path = self.base / "missing_history.json"
        (self.base / "missing_eval.sample.json").write_text(
            json.dumps({"macro_f1": 0.82}),
            encoding="utf-8",
        )
        (self.base / "missing_history.sample.json").write_text(
            json.dumps({"total_predictions": 12}),
            encoding="utf-8",
        )

        loader = ReportLoader(eval_path, history_path)
        metrics = loader.load_eval_metrics()
        history = loader.load_history_summary()
        self.assertEqual(metrics["macro_f1"], 0.82)
        self.assertEqual(history["total_predictions"], 12)


if __name__ == "__main__":
    unittest.main()
