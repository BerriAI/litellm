"""The caller's Langfuse trace controls, parsed from the live callback kwargs."""

from __future__ import annotations

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
    ``tags`` / ``trace_release`` / ``trace_version`` (else ``version``) on the request (SDK or proxy body), with the
    proxy's ``langfuse_<control>`` headers winning over the body for the scalar ones. ``existing_trace_id`` and
    ``update_trace_keys`` are deliberately not carried, and ``trace_id`` is not an attribute: it picks the trace a
    Langfuse exporter exports the call in (``plumbing.named_trace``)."""

    name: str | None = None
    user_id: str | None = None
    session_id: str | None = None
    tags: tuple[str, ...] = ()
    release: str | None = None
    version: str | None = None


def _header_value(headers: Mapping[str, object], name: str) -> str | None:
    direct: Final = as_str(headers.get(name))
    if direct:
        return direct
    target: Final = name.lower()
    return next(
        (text for key, value in headers.items() if key.lower() == target and (text := as_str(value))),
        None,
    )


def trace_controls_from_headers(
    headers: Mapping[str, object] | None, *, session_id: str | None = None
) -> TraceControls:
    if headers is None:
        return TraceControls(session_id=session_id or None)

    def from_header(control: str) -> str | None:
        return _header_value(headers, f"{LANGFUSE_HEADER_PREFIX}{control}")

    return TraceControls(
        name=from_header("trace_name"),
        user_id=from_header("trace_user_id"),
        session_id=from_header("session_id") or session_id or None,
        release=from_header("trace_release"),
        version=from_header("trace_version") or from_header("version"),
    )


def caller_trace_controls(kwargs: Mapping[str, object]) -> TraceControls:
    request: Final = as_str_mapping(kwargs.get("litellm_params"))
    if request is None:
        return TraceControls()
    proxy_request: Final = as_str_mapping(request.get("proxy_server_request"))
    bodies: Final = tuple(
        metadata
        for key in ("metadata", "litellm_metadata")
        if (metadata := as_str_mapping(request.get(key))) is not None
    )
    sources: Final = (*((proxy_request,) if proxy_request is not None else ()), *bodies)
    header_maps: Final = tuple(
        headers for source in sources if (headers := as_str_mapping(source.get("headers"))) is not None
    )

    def scalar(control: str) -> str | None:
        from_header: Final = next(
            (
                value
                for headers in header_maps
                if (value := _header_value(headers, f"{LANGFUSE_HEADER_PREFIX}{control}"))
            ),
            None,
        )
        if from_header:
            return from_header
        return next((value for body in bodies if (value := as_str(body.get(control)))), None)

    return TraceControls(
        name=scalar("trace_name"),
        user_id=scalar("trace_user_id"),
        session_id=scalar("session_id"),
        tags=next((tags for body in bodies if (tags := _str_items(body.get("tags")))), ()),
        release=scalar("trace_release"),
        version=scalar("trace_version") or scalar("version"),
    )


def _str_items(value: object) -> tuple[str, ...]:
    try:
        items: Final = _ITEMS.validate_python(value)
    except ValidationError:
        return ()
    return tuple(item for item in items if isinstance(item, str) and item)
