from __future__ import annotations

from typing import Final

from pydantic import TypeAdapter

SENTIMENT_PROMPT: Final = "Classify the sentiment of this review: 'The battery died after two days.'"
SENTIMENT_LABELS: Final = frozenset({"positive", "negative", "neutral"})
SENTIMENT_OUTPUT_FORMAT: Final[dict[str, object]] = {
    "type": "json_schema",
    "schema": {
        "type": "object",
        "properties": {"sentiment": {"type": "string", "enum": sorted(SENTIMENT_LABELS)}},
        "required": ["sentiment"],
        "additionalProperties": False,
    },
}
_SENTIMENT_JSON: Final = TypeAdapter(dict[str, str])


def assert_sentiment_json(text: str) -> None:
    parsed = _SENTIMENT_JSON.validate_json(text)
    assert set(parsed) == {"sentiment"}, f"output_format schema not enforced, extra or missing keys: {parsed}"
    assert parsed["sentiment"] in SENTIMENT_LABELS, f"sentiment outside the schema enum: {parsed}"
