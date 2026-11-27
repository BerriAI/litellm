"""Flat per-request pricing: a deployment's ``input_cost_per_query`` replaces token pricing on model calls."""

from typing import Final

from litellm.types.utils import CallTypes

PER_REQUEST_PRICED_CALL_TYPES: Final = frozenset(
    call_type.value
    for call_type in (
        CallTypes.completion,
        CallTypes.acompletion,
        CallTypes.text_completion,
        CallTypes.atext_completion,
        CallTypes.responses,
        CallTypes.aresponses,
        CallTypes.anthropic_messages,
        CallTypes.aanthropic_messages,
        CallTypes.generate_content,
        CallTypes.agenerate_content,
        CallTypes.embedding,
        CallTypes.aembedding,
        CallTypes.image_generation,
        CallTypes.aimage_generation,
        CallTypes.transcription,
        CallTypes.atranscription,
        CallTypes.ocr,
        CallTypes.aocr,
    )
)


def per_request_price(call_type: str, configured_cost_per_request: float | None) -> float | None:
    return configured_cost_per_request if call_type in PER_REQUEST_PRICED_CALL_TYPES else None
