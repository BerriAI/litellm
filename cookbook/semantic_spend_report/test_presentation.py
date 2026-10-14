import unittest
from typing import Final

from common import JSON
from presentation import scopes_for, validation_errors

RECORD: Final[dict[str, JSON]] = {
    "request_id": "req-a",
    "session_id": "session-a",
    "instruction": "Please propose dependency caching.",
    "response": "Use a lockfile hash in the cache key.",
}
ITEM: Final[dict[str, JSON]] = {
    "scope_id": "session:session-a",
    "title": "Proposed Dependency Cache Design",
    "overview": "Discussed a dependency caching strategy.",
    "bullets": [{"text": "Proposed lockfile-based cache keys", "request_ids": ["req-a"]}],
    "quotes": [
        {"text": "Please propose dependency caching.", "speaker": "user", "request_id": "req-a"},
        {"text": "Use a lockfile hash", "speaker": "assistant", "request_id": "req-a"},
    ],
}


class PresentationValidationTests(unittest.TestCase):
    def test_exact_quotes_and_scoped_findings_are_accepted(self) -> None:
        self.assertEqual(validation_errors(ITEM, frozenset({"req-a"}), {"req-a": RECORD}), ())

    def test_paraphrased_quote_is_rejected(self) -> None:
        item: Final = {
            **ITEM,
            "quotes": [
                {"text": "Use a hash of the lockfile", "speaker": "assistant", "request_id": "req-a"},
                {"text": "Please propose dependency caching.", "speaker": "user", "request_id": "req-a"},
            ],
        }
        self.assertIn(
            "Quote is not an exact source substring", validation_errors(item, frozenset({"req-a"}), {"req-a": RECORD})
        )

    def test_quote_attributed_to_wrong_speaker_is_rejected(self) -> None:
        item: Final = {
            **ITEM,
            "quotes": [
                {"text": "Use a lockfile hash", "speaker": "user", "request_id": "req-a"},
                {"text": "Please propose dependency caching.", "speaker": "assistant", "request_id": "req-a"},
            ],
        }
        self.assertIn(
            "Quote is not an exact source substring", validation_errors(item, frozenset({"req-a"}), {"req-a": RECORD})
        )

    def test_out_of_scope_citation_is_rejected(self) -> None:
        item: Final = {**ITEM, "bullets": [{"text": "A valid short finding", "request_ids": ["req-outside"]}]}
        self.assertIn("Finding has invalid citations", validation_errors(item, frozenset({"req-a"}), {"req-a": RECORD}))

    def test_wall_of_text_is_rejected(self) -> None:
        item: Final = {**ITEM, "overview": "word " * 26, "bullets": [{"text": "word " * 16, "request_ids": ["req-a"]}]}
        errors: Final = validation_errors(item, frozenset({"req-a"}), {"req-a": RECORD})
        self.assertIn("Overview must have 1 to 25 words", errors)
        self.assertIn("Finding exceeds 15 words", errors)

    def test_scopes_preserve_session_category_intersections(self) -> None:
        records: Final = (
            RECORD,
            {**RECORD, "request_id": "req-b"},
            {**RECORD, "request_id": "req-c", "session_id": "session-b"},
        )
        classifications: Final = (
            {"request_id": "req-a", "category": "ci"},
            {"request_id": "req-b", "category": "docs"},
            {"request_id": "req-c", "category": "ci"},
        )
        scopes: Final = {s["scope_id"]: s["request_ids"] for s in scopes_for(records, classifications)}
        self.assertEqual(scopes["session:session-a"], ["req-a", "req-b"])
        self.assertEqual(scopes["category:ci"], ["req-a", "req-c"])
        self.assertEqual(scopes["group:session-a:ci"], ["req-a"])


if __name__ == "__main__":
    unittest.main()
