import gzip
import json
import zlib
from collections.abc import Mapping
from io import BytesIO
from itertools import accumulate
from types import MappingProxyType
from typing import Final

from pydantic import JsonValue, TypeAdapter, ValidationError
from typing_extensions import ReadOnly, TypedDict

from litellm.constants import OTLP_MAX_ATTRIBUTE_VALUE_BYTES, OTLP_MAX_BODY_BYTES
from litellm.rust_bridge.traces import DecodedSpan
from litellm.rust_bridge.traces import decode_otlp as native_decode_otlp
from litellm.rust_bridge.traces import encode_error as native_encode_error
from litellm.tracing.types import SpanRow

_MESSAGE_LIST: Final = TypeAdapter(tuple[dict[str, JsonValue], ...])
_MAX_JSON_ESCAPE_BYTES: Final = 6


class InvalidOTLPPayloadError(ValueError):
    pass


class OTLPPayloadTooLargeError(OverflowError):
    pass


class OTLPError(TypedDict):
    message: ReadOnly[str]


def _truncate(value: str) -> str:
    encoded: Final = value.encode("utf-8")
    if len(encoded) <= OTLP_MAX_ATTRIBUTE_VALUE_BYTES:
        return value
    kept: Final = encoded[:OTLP_MAX_ATTRIBUTE_VALUE_BYTES].decode("utf-8", "ignore")
    return f"{kept}…[truncated {len(encoded) - len(kept.encode('utf-8'))} bytes]"


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
    payload: Final = _decode_content_encoding(body, content_encoding)
    try:
        spans: Final = native_decode_otlp(payload, content_type)
    except OverflowError as error:
        raise OTLPPayloadTooLargeError(str(error)) from error
    except ValueError as error:
        raise InvalidOTLPPayloadError(str(error)) from error
    return tuple(_span_row(span) for span in spans)


def _decode_content_encoding(body: bytes, content_encoding: str | None) -> bytes:
    if len(body) > OTLP_MAX_BODY_BYTES:
        raise OTLPPayloadTooLargeError(f"OTLP body exceeds {OTLP_MAX_BODY_BYTES} bytes")
    if content_encoding is None or content_encoding.lower() == "identity":
        return body
    if content_encoding.lower() != "gzip":
        raise InvalidOTLPPayloadError("Unsupported OTLP content encoding")
    try:
        with gzip.GzipFile(fileobj=BytesIO(body)) as stream:
            payload: Final = stream.read(OTLP_MAX_BODY_BYTES + 1)
    except (EOFError, OSError, zlib.error) as error:
        raise InvalidOTLPPayloadError("Invalid OTLP gzip body") from error
    if len(payload) > OTLP_MAX_BODY_BYTES:
        raise OTLPPayloadTooLargeError(f"OTLP body exceeds {OTLP_MAX_BODY_BYTES} bytes")
    return payload


def _exception_message(span: DecodedSpan) -> str:
    for event in span["events"]:
        if event["name"] == "exception":
            return event["attributes"].get("exception.message") or event["attributes"].get("exception.type", "")
    return ""


def _span_row(span: DecodedSpan) -> SpanRow:
    attributes: Final = span["attributes"]
    normalized: Final = span["normalized"]
    return SpanRow(
        Timestamp=span["start_ns"],
        TraceId=span["trace_id"],
        SpanId=span["span_id"],
        ParentSpanId=span["parent_span_id"],
        TraceState=span["trace_state"],
        SpanName=span["name"],
        SpanKind=span["kind"],
        ServiceName=span["resource_attributes"].get("service.name", ""),
        ResourceAttributes=span["resource_attributes"],
        ScopeName=span["scope_name"],
        ScopeVersion=span["scope_version"],
        SpanAttributes=MappingProxyType(
            {key: _truncate(value) for key, value in attributes.items() if key not in span["consumed_attributes"]}
        ),
        Duration=span["end_ns"] - span["start_ns"],
        StatusCode=span["status_code"],
        StatusMessage=span["status_message"] or _exception_message(span),
        TeamId="",
        ApiKeyHash="",
        UserId="",
        ObservationType=normalized.observation_type,
        AgentName=normalized.agent_name,
        Framework=normalized.framework,
        Model=normalized.model,
        LiteLLMRequestId=attributes.get("gen_ai.response.id") or normalized.litellm_request_id,
        InputTokens=normalized.input_tokens,
        OutputTokens=normalized.output_tokens,
        Input=_truncate_payload(normalized.input),
        Output=_truncate(normalized.output),
    )


def encode_otlp_response(content_type: str | None, error: str | None = None) -> tuple[bytes, str]:
    media_type: Final = (content_type or "application/x-protobuf").split(";", 1)[0].strip().lower()
    if media_type == "application/json":
        response: Final[OTLPError] = {"message": error or ""}
        return (json.dumps(response).encode() if error else b"{}"), "application/json"
    if error is None:
        return b"", "application/x-protobuf"
    return native_encode_error(error), "application/x-protobuf"
