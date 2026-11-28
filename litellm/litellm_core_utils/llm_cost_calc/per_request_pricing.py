"""Flat per-request pricing: ``input_cost_per_query`` set on a call or its deployment replaces usage pricing."""

from collections.abc import Mapping
from typing import Final

from pydantic import TypeAdapter

from litellm.litellm_core_utils import model_cost_table
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
REQUEST_PRICE_FIELD: Final = "input_cost_per_query"
_ENTRY: Final = TypeAdapter(Mapping[str, object])


def _price_in(entry: object) -> float | None:
    if not isinstance(entry, Mapping):
        return None
    price: Final = _ENTRY.validate_python(entry).get(REQUEST_PRICE_FIELD)
    return float(price) if isinstance(price, int | float) and not isinstance(price, bool) else None


def per_request_price(call_type: str, call_params: object, deployment_id: str | None) -> float | None:
    """The flat price of one call: the call's own ``input_cost_per_query``, else the one its router deployment
    registered. The raw registration only carries prices the deployment declared, so a cost-map per-query rate
    (rerank, search, Marengo embeddings) never turns into a flat request price here."""
    if call_type not in PER_REQUEST_PRICED_CALL_TYPES:
        return None
    requested: Final = _price_in(call_params)
    if requested is not None:
        return requested
    return deployment_request_price(deployment_id) if deployment_id else None


def deployment_request_price(deployment_id: str) -> float | None:
    """The flat per-request price a router deployment declared when it registered, if any."""
    return _price_in(model_cost_table.published().get(deployment_id))
