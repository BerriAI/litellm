import unittest
from decimal import Decimal
from typing import Final

from common import JSON
from pipeline import (
    aggregate,
    classification_input,
    latest_instruction,
    response_text,
    stream_for,
)


def record(request_id: str, session_id: str, spend: str) -> dict[str, JSON]:
    return {
        "request_id": request_id,
        "session_id": session_id,
        "spend_usd": spend,
        "instruction": "Write the guide",
        "response": "A proposed guide",
        "messages": [{"role": "user", "content": "Earlier context"}, {"role": "user", "content": "Write the guide"}],
    }


class SemanticSpendTests(unittest.TestCase):
    def test_multiple_sessions_and_categories_reconcile_without_float_rounding(self) -> None:
        records: Final = (record("a", "one", "0.1"), record("b", "one", "0.2"), record("c", "two", "0.00000001"))
        classifications: Final = (
            {"request_id": "a", "category": "code"},
            {"request_id": "b", "category": "docs"},
            {"request_id": "c", "category": "code"},
        )
        categories: Final = ({"id": "code", "name": "Code"}, {"id": "docs", "name": "Docs"})
        results: Final = aggregate(records, classifications, categories)
        self.assertEqual(Decimal(str(results[0]["spend_usd"])), Decimal("0.10000001"))
        self.assertEqual(results[0]["session_count"], 2)
        self.assertEqual(results[1]["request_count"], 1)
        self.assertEqual(sum(Decimal(str(r["spend_usd"])) for r in results), Decimal("0.30000001"))

    def test_unknown_spend_remains_in_total(self) -> None:
        results: Final = aggregate(
            (record("a", "one", "0.8"),),
            ({"request_id": "a", "category": "unknown"},),
            ({"id": "unknown", "name": "Unknown"}, {"id": "docs", "name": "Docs"}),
        )
        self.assertEqual(results[0]["spend_usd"], "0.8")
        self.assertEqual(results[0]["percent"], 100.0)
        self.assertEqual(results[1]["request_count"], 0)

    def test_missing_cost_is_visible_and_not_counted_as_known_cost(self) -> None:
        results: Final = aggregate(
            ({**record("a", "one", "0"), "spend_usd": None},),
            ({"request_id": "a", "category": "unknown"},),
            ({"id": "unknown", "name": "Unknown"},),
        )
        self.assertEqual(results[0]["missing_cost_count"], 1)

    def test_latest_instruction_ignores_old_repeated_history(self) -> None:
        messages: Final = (
            {"role": "user", "content": "Optimize CI"},
            {"role": "assistant", "content": [{"type": "text", "text": "Use a cache"}]},
            {"role": "user", "content": [{"type": "text", "text": "Now write documentation"}]},
        )
        self.assertEqual(latest_instruction(messages), "Now write documentation")

    def test_discovery_stream_has_only_selected_session_and_current_instructions(self) -> None:
        records: Final = (record("a", "one", "0.1"), record("b", "two", "0.2"))
        stream: Final = stream_for(records, "one")
        self.assertEqual(len(stream), 1)
        self.assertEqual(stream[0]["request_id"], "a")
        self.assertEqual(stream[0]["instruction"], "Write the guide")
        self.assertNotIn("Earlier context", str(stream))

    def test_classifier_receives_followup_context_and_complete_taxonomy(self) -> None:
        categories: Final = (
            {"id": "docs", "definition": "Documentation", "include": "Guides", "exclude": "Code"},
            {"id": "unknown", "definition": "Unknown", "include": "Missing evidence", "exclude": "Clear evidence"},
        )
        body: Final = classification_input(record("a", "one", "0.1"), categories)
        self.assertIn("Earlier context", str(body["state"]))
        self.assertIn("Write the guide", str(body["state"]))
        self.assertIn("unknown", str(body["questions"]))
        self.assertIn("docs", str(body["questions"]))

    def test_response_extraction_supports_logged_chat_and_messages_shapes(self) -> None:
        self.assertEqual(response_text({"choices": [{"message": {"content": "Draft"}}]}), "Draft")
        self.assertEqual(response_text({"content": [{"type": "text", "text": "Draft"}]}), "Draft")


if __name__ == "__main__":
    unittest.main()
