"""The caller's Langfuse trace controls, parsed from the live callback kwargs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final, Literal

from pydantic import TypeAdapter, ValidationError

from litellm.integrations.langfuse.langfuse_otel_attributes import default_langfuse_tags
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


def caller_trace_controls(
    kwargs: Mapping[str, object], *, langfuse_defaults: Literal["request", "completed"] | None = None
) -> TraceControls:
    request: Final = as_str_mapping(kwargs.get("litellm_params")) or {}
    proxy_request: Final = as_str_mapping(request.get("proxy_server_request"))
    headers: Final = as_str_mapping(proxy_request.get("headers")) if proxy_request is not None else None
    user_metadata: Final = as_str_mapping(request.get("metadata")) or {}
    model_metadata: Final = as_str_mapping(request.get("litellm_metadata")) or {}
    bodies: Final = (user_metadata, model_metadata)

    def scalar(control: str) -> str | None:
        from_header: Final = as_str(headers.get(f"{LANGFUSE_HEADER_PREFIX}{control}")) if headers is not None else None
        if from_header:
            return from_header
        return next((value for body in bodies if (value := as_str(body.get(control)))), None)

    caller_tags: Final = next((tags for body in bodies if (tags := _str_items(body.get("tags")))), ())
    defaults: Final = (
        default_langfuse_tags(
            kwargs,
            {**model_metadata, **user_metadata},
            completed=langfuse_defaults == "completed",
        )
        if langfuse_defaults is not None
        else ()
    )
    return TraceControls(
        name=scalar("trace_name"),
        user_id=scalar("trace_user_id"),
        session_id=scalar("session_id"),
        tags=tuple(dict.fromkeys(caller_tags + defaults)) if defaults else caller_tags,
    )


def _str_items(value: object) -> tuple[str, ...]:
    try:
        items: Final = _ITEMS.validate_python(value)
    except ValidationError:
        return ()
    return tuple(item for item in items if isinstance(item, str) and item)
