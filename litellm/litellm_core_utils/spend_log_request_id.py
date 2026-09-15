"""The id a request is filed under in LiteLLM_SpendLogs and in the per-request payload
stores (GCS) the logs viewer reads back through ``/spend/logs/ui/{request_id}``.

Both sides must derive the id the same way or the viewer looks up a payload under a key
it was never stored under, so the derivation lives here rather than in either caller.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

from litellm.types.utils import CallTypes

BATCH_COST_REQUEST_ID_SUFFIX: Final = "_batch_cost"

_RESPONSE_ID_KEYED_CALL_TYPES: Final = frozenset(
    {
        CallTypes.acreate_batch.value,
        CallTypes.aretrieve_batch.value,
        CallTypes.acreate_file.value,
    }
)
"""Batch and file rows key off the object's own id so repeated polls of the same
object collapse into one row instead of billing it once per poll. Every other call
type keys off the proxy-generated per-call id: request_id is the LiteLLM_SpendLogs
primary key and the flush inserts with skip_duplicates, so keying off the provider's
response id silently drops every row after the first whenever a provider (commonly a
self-hosted OpenAI-compatible server) reuses completion ids."""


def _standard_logging_id(kwargs: Mapping[str, object]) -> str | None:
    match kwargs.get("standard_logging_object"):
        case {"id": str() as standard_logging_id}:
            return standard_logging_id
        case _:
            return None


def get_spend_logs_id(call_type: str, response_obj: Mapping[str, object], kwargs: Mapping[str, object]) -> str | None:
    candidate_ids: Final = (
        (
            response_obj.get("id"),
            _standard_logging_id(kwargs),
            kwargs.get("litellm_call_id"),
        )
        if call_type in _RESPONSE_ID_KEYED_CALL_TYPES
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
    return resolved_id
