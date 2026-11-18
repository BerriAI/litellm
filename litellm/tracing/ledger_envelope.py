"""Reads the Lens ledger envelope a spend log row carries into the accounting transaction.

The local spend log flush JSON-encodes every mapping on a row before the insert, while the SPEND_LOGS_URL
path passes the mapping itself, so the envelope arrives as either. The envelope is optional: a malformed
one must never roll back the charge, so it is replaced by a cost-only record built from the same row's
committed accounting snapshot.
"""

from collections.abc import Mapping
from datetime import datetime
from math import isfinite
from typing import Final

from pydantic import TypeAdapter, ValidationError

from litellm._logging import verbose_proxy_logger
from litellm.tracing.types import LensExport

_LENS_EXPORT: Final = TypeAdapter(LensExport)
_SCOPE_TEXT: Final = (
    "call_type",
    "api_key",
    "team_id",
    "organization_id",
    "user",
    "end_user",
    "model",
    "model_group",
    "custom_llm_provider",
    "status",
)
_SCOPE_COUNTS: Final = ("prompt_tokens", "completion_tokens")
_NO_CONTENT: Final = {"messages": "", "response": "", "metadata": "", "error_str": ""}


def _millis(value: object) -> int | None:
    if isinstance(value, datetime):
        return int(value.timestamp() * 1000)
    if not isinstance(value, str):
        return None
    try:
        return int(datetime.fromisoformat(value).timestamp() * 1000)
    except ValueError:
        return None


def _spend(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if isfinite(value) else None


def cost_only_export(request_id: str, accounting: Mapping[str, object]) -> LensExport | None:
    """The committed charge alone: no content and nothing outside this row's accounting snapshot."""
    start_ms: Final = _millis(accounting.get("startTime"))
    if start_ms is None or start_ms <= 0:
        return None
    text: Final = {name: value for name in _SCOPE_TEXT if isinstance(value := accounting.get(name), str) and value}
    counts: Final = {
        name: count
        for name in _SCOPE_COUNTS
        if isinstance(count := accounting.get(name), int) and not isinstance(count, bool) and count >= 0
    }
    return LensExport(
        record={
            **text,
            **counts,
            **_NO_CONTENT,
            "request_id": request_id,
            "spend": _spend(accounting.get("spend")),
            "start_time": start_ms,
            "end_time": start_ms,
        },
        payloads=(),
    )


def committed_lens_export(value: object, *, request_id: str, accounting: Mapping[str, object]) -> LensExport | None:
    try:
        return (
            _LENS_EXPORT.validate_json(value)
            if isinstance(value, (str, bytes))
            else _LENS_EXPORT.validate_python(value)
        )
    except ValidationError as error:
        verbose_proxy_logger.warning(
            "Lens ledger envelope for request %s is malformed (%d errors); exporting its committed cost only",
            request_id,
            error.error_count(),
        )
        return cost_only_export(request_id, accounting)
