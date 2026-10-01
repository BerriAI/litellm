"""
OTLP/HTTP trace export -> `SpanRow`s.

Pure functions, no I/O. Two steps:
1. `decode_otlp()`   protobuf / JSON / gzip `ExportTraceServiceRequest` -> flat spans
2. `normalize()`     framework conventions -> LiteLLM columns (type, agent, input/output,
                     LiteLLM request id). Supported: LangSmith (LangChain, LangGraph,
                     Deep Agents), OTEL GenAI semconv, OpenInference.
"""

import json
from collections.abc import Mapping
from itertools import accumulate
from types import MappingProxyType
from typing import Final

from pydantic import JsonValue, TypeAdapter, ValidationError
from typing_extensions import ReadOnly, TypedDict

from litellm.constants import OTLP_MAX_ATTRIBUTE_VALUE_BYTES, OTLP_MAX_BODY_BYTES
from litellm.rust_bridge.traces import DecodedSpan
from litellm.rust_bridge.traces import decode_otlp as native_decode_otlp
from litellm.tracing.normalizers import select_normalizer
from litellm.tracing.normalizers.base import to_int
from litellm.tracing.types import SpanRow

_MESSAGE_LIST: Final = TypeAdapter(tuple[dict[str, JsonValue], ...])
_MAX_JSON_ESCAPE_BYTES: Final = 6

# attributes whose content we lift into Input/Output and drop from SpanAttributes
_HEAVY_ATTRIBUTES: Final = frozenset(
    {
        "gen_ai.prompt",
        "gen_ai.completion",
        "gen_ai.tool.definitions",
        "gen_ai.input.messages",
        "gen_ai.output.messages",
        "input.value",
        "output.value",
    }
)


class InvalidOTLPPayloadError(ValueError):
    pass


class OTLPPayloadTooLargeError(OverflowError):
    pass


def _truncate(value: str) -> str:
    size = len(value.encode("utf-8"))
    if size <= OTLP_MAX_ATTRIBUTE_VALUE_BYTES:
        return value
    kept = value.encode("utf-8")[:OTLP_MAX_ATTRIBUTE_VALUE_BYTES].decode("utf-8", "ignore")
    return f"{kept}…[truncated {size - OTLP_MAX_ATTRIBUTE_VALUE_BYTES} bytes]"


def _size(value: str) -> int:
    return len(value.encode("utf-8"))


class _ElisionMarker(TypedDict):
    role: ReadOnly[str]
    content: ReadOnly[str]


def _elided(count: int) -> str:
    marker: Final[_ElisionMarker] = {"role": "system", "content": f"…[{count} earlier messages truncated]"}
    return json.dumps(marker)


def _with_content(message: Mapping[str, JsonValue], content: str) -> str:
    return json.dumps(MappingProxyType({**message, "content": content}), default=lambda proxy: proxy.copy())


def _shrunk_message(message: Mapping[str, JsonValue], budget: int) -> str:
    """One message cut to `budget` bytes, as valid JSON.

    Shortens `content` first; if other fields (e.g. huge tool_calls) still don't fit, keeps only role + content.
    """
    content: Final = message.get("content")
    text: Final = content if isinstance(content, str) else json.dumps(content)
    role_only: Final = MappingProxyType({"role": message.get("role", "user")})
    attempts: Final = (
        _cut_content(message, text, budget, 1),
        _cut_content(role_only, text, budget, 1),
        _cut_content(role_only, text, budget, _MAX_JSON_ESCAPE_BYTES),
    )
    return next((attempt for attempt in attempts if _size(attempt) <= budget), attempts[-1])


def _cut_content(message: Mapping[str, JsonValue], text: str, budget: int, escape_factor: int) -> str:
    overhead: Final = _size(_with_content(message, ""))
    room: Final = max(0, budget - overhead - 48) // escape_factor
    kept: Final = text.encode("utf-8")[:room].decode("utf-8", "ignore")
    return _with_content(message, f"{kept}…[truncated {_size(text) - _size(kept)} bytes]")


def _newest_that_fit(encoded: tuple[str, ...], budget: int) -> int:
    """How many trailing messages fit in `budget` bytes (comma separators included), scanning newest first."""
    sizes: Final = tuple(_size(m) + 1 for m in reversed(encoded))
    totals: Final = tuple(accumulate(sizes))
    return next((count for count, total in enumerate(totals) if total > budget), len(totals))


def _truncate_payload(value: str) -> str:
    """Message arrays keep the first message, an elision marker and the newest messages that fit.

    The result is always valid JSON: if even those don't fit, the first and last messages are shortened.
    Anything that isn't a message array is byte-truncated as before.
    """
    if _size(value) <= OTLP_MAX_ATTRIBUTE_VALUE_BYTES or not value.startswith("["):
        return _truncate(value)
    try:
        messages: Final = _MESSAGE_LIST.validate_json(value)
    except ValidationError:
        return _truncate(value)
    if len(messages) < 2:
        return _truncate(value)
    encoded: Final = tuple(json.dumps(m) for m in messages)
    marker_budget: Final = _size(_elided(len(messages))) + 1
    budget: Final = OTLP_MAX_ATTRIBUTE_VALUE_BYTES - 2 - _size(encoded[0]) - 1 - marker_budget
    kept: Final = min(_newest_that_fit(encoded[1:], budget), len(messages) - 2)
    if kept > 0:
        tail: Final = encoded[len(encoded) - kept :]
        return "[" + ", ".join((encoded[0], _elided(len(messages) - 1 - kept), *tail)) + "]"
    half: Final = (OTLP_MAX_ATTRIBUTE_VALUE_BYTES - marker_budget - 4) // 2
    middle: Final = (_elided(len(messages) - 2),) if len(messages) > 2 else ()
    shrunk: Final = (
        "[" + ", ".join((_shrunk_message(messages[0], half), *middle, _shrunk_message(messages[-1], half))) + "]"
    )
    return shrunk if _size(shrunk) <= OTLP_MAX_ATTRIBUTE_VALUE_BYTES else "[" + _elided(len(messages)) + "]"


def decode_otlp(
    body: bytes, content_type: str | None = None, content_encoding: str | None = None
) -> tuple[SpanRow, ...]:
    """Decode an OTLP trace export and normalize every span."""
    try:
        spans: Final = native_decode_otlp(body, content_type, content_encoding, OTLP_MAX_BODY_BYTES)
    except OverflowError as error:
        raise OTLPPayloadTooLargeError(str(error)) from error
    except ValueError as error:
        raise InvalidOTLPPayloadError(str(error)) from error
    return tuple(_span_row(span) for span in spans)


def _exception_message(span: DecodedSpan) -> str:
    """`span.record_exception()` writes an `exception` event; surface it when status.message is empty."""
    for event in span["events"]:
        if event["name"] == "exception":
            attributes = event["attributes"]
            return attributes.get("exception.message") or attributes.get("exception.type", "")
    return ""


def _span_row(span: DecodedSpan) -> SpanRow:
    attributes = span["attributes"]
    resource = span["resource_attributes"]
    row = SpanRow(
        Timestamp=span["start_ns"],
        TraceId=span["trace_id"],
        SpanId=span["span_id"],
        ParentSpanId=span["parent_span_id"],
        TraceState=span["trace_state"],
        SpanName=span["name"],
        SpanKind=span["kind"],
        ServiceName=resource.get("service.name", ""),
        ResourceAttributes=resource,
        ScopeName=span["scope_name"],
        ScopeVersion=span["scope_version"],
        SpanAttributes=attributes,
        Duration=max(span["end_ns"] - span["start_ns"], 0),
        StatusCode=span["status_code"],
        StatusMessage=span["status_message"] or _exception_message(span),
        TeamId="",
        ApiKeyHash="",
        ObservationType="chain",
        AgentName="",
        LiteLLMRequestId="",
        Model="",
        InputTokens=0,
        OutputTokens=0,
        Input="",
        Output="",
    )
    normalize(row, attributes)
    row["SpanAttributes"] = {k: _truncate(v) for k, v in attributes.items() if k not in _HEAVY_ATTRIBUTES}
    row["Input"], row["Output"] = _truncate_payload(row["Input"]), _truncate(row["Output"])
    return row


def _set_tokens(row: SpanRow, attributes: Mapping[str, str]) -> None:
    row["InputTokens"] = to_int(attributes.get("gen_ai.usage.input_tokens"))
    row["OutputTokens"] = to_int(attributes.get("gen_ai.usage.output_tokens"))


def normalize(row: SpanRow, attributes: Mapping[str, str]) -> None:
    select_normalizer(row["ScopeName"], attributes).normalize(row, attributes)
    if not row["InputTokens"] and not row["OutputTokens"]:
        _set_tokens(row, attributes)


def encode_otlp_response(content_type: str | None) -> tuple[bytes, str]:
    """Empty ExportTraceServiceResponse in the caller's encoding."""
    if content_type and "json" in content_type:
        return b"{}", "application/json"
    return b"", "application/x-protobuf"
