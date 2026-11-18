"""The Lens ledger record committed beside an accepted accounting delivery.

Amount and request id come from the accounting payload. Correlation ids, redaction and content come from
the same row builder the legacy Lens callback uses, so both producers describe a request identically.
"""

import hashlib
import os
from collections.abc import Mapping, Sequence
from datetime import datetime
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal

from pydantic import TypeAdapter, ValidationError

from litellm.integrations.clickhouse.clickhouse_spend_logger import spend_log_row_from_payload
from litellm.tracing.preview import preview
from litellm.tracing.types import LedgerCapabilities, LensExport, LensPayload, SpendLogPayload

if TYPE_CHECKING:
    from litellm.proxy._types import SpendLogsPayload

LENS_EXPORT_KEY: Final = "lens_export"
DEFAULT_MAX_PAYLOAD_BYTES: Final = 16 * 1024 * 1024
_CONTENT_FIELDS: Final = ("messages", "response")
_REQUEST_ONLY_FIELDS: Final = MappingProxyType({"messages": "", "response": "", "metadata": ""})
_PAYLOAD: Final = TypeAdapter(SpendLogPayload)
_TRACE_ROUTES: Final = ("/v1/traces", "/v1/logs")


def max_payload_bytes(environ: Mapping[str, str] = os.environ) -> int:
    configured: Final = environ.get("LITELLM_LENS_PAYLOAD_MAX_BYTES", "")
    return int(configured) if configured.isdigit() and int(configured) > 0 else DEFAULT_MAX_PAYLOAD_BYTES


def _storable(value: object) -> object:
    return value.replace("\x00", "\ufffd") if isinstance(value, str) else value


def _millis(moment: datetime) -> int:
    return int(moment.timestamp() * 1000)


def _correlated_row(kwargs: Mapping[str, object]) -> Mapping[str, object] | None:
    raw: Final = kwargs.get("standard_logging_object")
    if raw is None:
        return None
    try:
        payload: Final = _PAYLOAD.validate_python(raw)
    except ValidationError:
        return None
    if str(payload.get("call_type") or "").startswith(_TRACE_ROUTES):
        return None
    return dict(spend_log_row_from_payload(payload, kwargs))


def _accounting_row(payload: "SpendLogsPayload") -> Mapping[str, object]:
    return {
        "call_type": payload.get("call_type") or "",
        "api_key": payload.get("api_key") or "",
        "team_id": payload.get("team_id") or "",
        "organization_id": payload.get("organization_id") or "",
        "user": payload.get("user") or "",
        "model": payload.get("model") or "",
        "model_group": payload.get("model_group") or "",
        "custom_llm_provider": payload.get("custom_llm_provider") or "",
        "status": payload.get("status") or "",
        "session_id": payload.get("session_id") or "",
    }


def _spill(
    field: Literal["messages", "response"],
    content: str,
    owner: Mapping[str, object],
    event_time_ms: int,
    capabilities: LedgerCapabilities,
    payload_limit: int,
) -> tuple[str, LensPayload | None]:
    size: Final = len(content.encode())
    if size <= capabilities.inline_bytes:
        return content, None
    if not capabilities.payload_store or size > payload_limit:
        return preview(content, capabilities.inline_bytes), None
    return preview(content, capabilities.preview_bytes), LensPayload(
        field=field,
        team_id=str(owner.get("team_id") or ""),
        api_key=str(owner.get("api_key") or ""),
        event_time_ms=event_time_ms,
        content=content,
        bytes=size,
        sha256=hashlib.sha256(content.encode()).hexdigest(),
    )


def lens_export(
    payload: "SpendLogsPayload",
    kwargs: Mapping[str, object],
    *,
    start_time: datetime,
    end_time: datetime,
    keep_content: bool,
    capabilities: LedgerCapabilities,
    payload_limit: int,
) -> LensExport:
    correlated: Final = _correlated_row(kwargs)
    base: Final = correlated if correlated is not None else _accounting_row(payload)
    event_time_ms: Final = _millis(start_time)
    owner: Final = {
        **{name: _storable(value) for name, value in base.items()},
        "request_id": payload["request_id"],
        "spend": payload["spend"],
        "start_time": event_time_ms,
        "end_time": _millis(end_time),
        **({} if keep_content and correlated is not None else _REQUEST_ONLY_FIELDS),
    }
    spilled: Final = tuple(
        (field, *_spill(field, str(owner.get(field) or ""), owner, event_time_ms, capabilities, payload_limit))
        for field in _CONTENT_FIELDS
    )
    return LensExport(
        record={**owner, **{field: content for field, content, _ in spilled}},
        payloads=tuple(body for _, _, body in spilled if body is not None),
    )
