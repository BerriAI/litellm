"""The id a request is filed under in LiteLLM_SpendLogs and in the per-request payload
stores (GCS) the logs viewer reads back through ``/spend/logs/ui/{request_id}``.

Both sides must derive the id the same way or the viewer looks up a payload under a key
it was never stored under, so the derivation lives here rather than in either caller.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

from litellm.constants import EMPTY_MAPPING, INTERNAL_CALL_ORIGIN_METADATA_KEY, NON_INFERENCE_CALL_TYPES
from litellm.types.utils import BACKGROUND_RESPONSE_COST_POLL_CALL_ORIGIN, CallTypes

BATCH_COST_REQUEST_ID_SUFFIX: Final = "_batch_cost"
BACKGROUND_RESPONSE_COST_REQUEST_ID_SUFFIX: Final = "_background_cost"

_RESPONSE_ID_KEYED_CALL_TYPES: Final = frozenset(
    {
        CallTypes.acreate_batch.value,
        CallTypes.aretrieve_batch.value,
        CallTypes.acreate_file.value,
        CallTypes.anthropic_messages.value,
        CallTypes.aanthropic_messages.value,
    }
)
"""Keep batch/file polls idempotent and Messages response-id lookups compatible.
Other calls use a per-call id so reused provider ids cannot drop spend rows."""


def _standard_logging_id(kwargs: Mapping[object, object]) -> str | None:
    match kwargs.get("standard_logging_object"):
        case {"id": str() as standard_logging_id}:
            return standard_logging_id
        case _:
            return None


def get_spend_logs_id(
    call_type: object,
    response_obj: Mapping[object, object],
    kwargs: Mapping[object, object],
    metadata: Mapping[str, object] = EMPTY_MAPPING,
) -> str | None:
    candidate_ids: Final = (
        (
            response_obj.get("id"),
            _standard_logging_id(kwargs),
            kwargs.get("litellm_call_id"),
        )
        if call_type in _RESPONSE_ID_KEYED_CALL_TYPES or call_type in NON_INFERENCE_CALL_TYPES
        else (
            kwargs.get("litellm_call_id"),
            _standard_logging_id(kwargs),
            response_obj.get("id"),
        )
    )
    resolved_id: Final = next(
        (candidate for candidate in candidate_ids if isinstance(candidate, str) and candidate), None
    )
    if resolved_id is not None and call_type == CallTypes.aretrieve_batch.value:
        return f"{resolved_id}{BATCH_COST_REQUEST_ID_SUFFIX}"
    if (
        resolved_id is not None
        and call_type in NON_INFERENCE_CALL_TYPES
        and metadata.get(INTERNAL_CALL_ORIGIN_METADATA_KEY) == BACKGROUND_RESPONSE_COST_POLL_CALL_ORIGIN
    ):
        return f"{resolved_id}{BACKGROUND_RESPONSE_COST_REQUEST_ID_SUFFIX}"
    return resolved_id
