import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from backend.app.feedback import FeedbackStore


class FeedbackStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = TemporaryDirectory()
        self.path = Path(self.tmp.name) / "feedback.jsonl"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_append_and_recent_entries(self) -> None:
        store = FeedbackStore(self.path, cache_size=5)
        store.append(text="Спасибо за сервис", predicted_label="positive", user_label="positive")
        store.append(
            text="Приложение вылетает",
            predicted_label="negative",
            user_label="negative",
            notes="Повторяется каждый день",
        )

        self.assertEqual(store.count(), 2)
        recent = store.recent()
        self.assertEqual(len(recent), 2)
        self.assertEqual(recent[0]["predicted_label"], "negative")
        self.assertIn("timestamp", recent[0])

        # file should contain serialized entries
        content = self.path.read_text(encoding="utf-8").strip().splitlines()
        labels = [json.loads(line)["predicted_label"] for line in content]
        self.assertEqual(len(labels), 2)
        self.assertIn("negative", labels)

    def test_bootstrap_cache(self) -> None:
        store = FeedbackStore(self.path, cache_size=1)
        store.append(text="test", predicted_label="neutral")
        store.append(text="ещё", predicted_label="positive")

        restored = FeedbackStore(self.path, cache_size=1)
        self.assertEqual(restored.count(), 2)
        recent = restored.recent()
        self.assertEqual(len(recent), 1)
        self.assertEqual(recent[0]["predicted_label"], "positive")


if __name__ == "__main__":
    unittest.main()
