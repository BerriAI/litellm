"""The caller's Langfuse trace controls, parsed from the live callback kwargs."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

from pydantic import TypeAdapter, ValidationError

from litellm.integrations.otel.model.utils import as_str, as_str_mapping

LANGFUSE_HEADER_PREFIX: Final = "langfuse_"
_ITEMS: Final = TypeAdapter(tuple[object, ...])


@dataclass(frozen=True, slots=True)
class TraceControls:
    """The caller's trace-level Langfuse controls: ``metadata.trace_name`` / ``trace_user_id`` / ``session_id`` /
    ``tags`` on the request (SDK or proxy body), with the proxy's ``langfuse_<control>`` headers winning over the
    body for the scalar ones. Mutation controls (``trace_id``, ``existing_trace_id``, ``update_trace_keys``) are
    deliberately not carried."""

    name: str | None = None
    user_id: str | None = None
    session_id: str | None = None
    tags: tuple[str, ...] = ()
    environment: str | None = None
    release: str | None = None
    version: str | None = None


def caller_trace_controls(kwargs: Mapping[str, object]) -> TraceControls:
    request: Final = as_str_mapping(kwargs.get("litellm_params"))
    if request is None:
        return TraceControls(environment=os.environ.get("LANGFUSE_TRACING_ENVIRONMENT") or None)
    proxy_request: Final = as_str_mapping(request.get("proxy_server_request"))
    headers: Final = as_str_mapping(proxy_request.get("headers")) if proxy_request is not None else None
    bodies: Final = tuple(
        metadata
        for key in ("metadata", "litellm_metadata")
        if (metadata := as_str_mapping(request.get(key))) is not None
    )

    def scalar(control: str, *, body_control: str | None = None) -> str | None:
        from_header: Final = as_str(headers.get(f"{LANGFUSE_HEADER_PREFIX}{control}")) if headers is not None else None
        if from_header:
            return from_header
        return next((value for body in bodies if (value := as_str(body.get(body_control or control)))), None)

    return TraceControls(
        name=scalar("trace_name"),
        user_id=scalar("trace_user_id"),
        session_id=scalar("session_id"),
        tags=next((tags for body in bodies if (tags := _str_items(body.get("tags")))), ()),
        environment=scalar("environment", body_control="langfuse_environment")
        or os.environ.get("LANGFUSE_TRACING_ENVIRONMENT")
        or None,
        release=scalar("release", body_control="trace_release"),
        version=scalar("version", body_control="trace_version"),
    )


def _str_items(value: object) -> tuple[str, ...]:
    try:
        items: Final = _ITEMS.validate_python(value)
    except ValidationError:
        return ()
    return tuple(item for item in items if isinstance(item, str) and item)
