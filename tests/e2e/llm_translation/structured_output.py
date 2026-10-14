from __future__ import annotations

from typing import Final

from pydantic import BaseModel, ConfigDict, TypeAdapter

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
CONSTRAINED_SENTIMENT_OUTPUT_FORMAT: Final[dict[str, object]] = {
    "type": "json_schema",
    "schema": {
        "type": "object",
        "properties": {
            "sentiment": {"type": "string", "enum": sorted(SENTIMENT_LABELS)},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "reasons": {"type": "array", "items": {"type": "string"}, "maxItems": 3},
        },
        "required": ["sentiment", "confidence", "reasons"],
        "additionalProperties": False,
    },
}
_SENTIMENT_JSON: Final = TypeAdapter(dict[str, str])


class _ConstrainedSentiment(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    sentiment: str
    confidence: float
    reasons: tuple[str, ...]


def assert_sentiment_json(text: str) -> None:
    parsed = _SENTIMENT_JSON.validate_json(text)
    assert set(parsed) == {"sentiment"}, f"output_format schema not enforced, extra or missing keys: {parsed}"
    assert parsed["sentiment"] in SENTIMENT_LABELS, f"sentiment outside the schema enum: {parsed}"


def assert_constrained_sentiment_json(text: str) -> None:
    parsed = _ConstrainedSentiment.model_validate_json(text)
    assert parsed.sentiment in SENTIMENT_LABELS, f"sentiment outside the schema enum: {parsed}"
