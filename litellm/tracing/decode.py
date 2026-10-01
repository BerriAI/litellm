"""
OTLP/HTTP trace export -> `SpanRow`s.

Pure functions, no I/O. Two steps:
1. `decode_otlp()`   protobuf / JSON / gzip `ExportTraceServiceRequest` -> flat spans
2. `normalize()`     framework conventions -> LiteLLM columns (type, agent, input/output,
                     LiteLLM request id). Supported: LangSmith (LangChain, LangGraph,
                     Deep Agents), OTEL GenAI semconv, OpenInference.
"""

import gzip
import json
import zlib
from collections.abc import Mapping
from dataclasses import dataclass
from io import BytesIO
from itertools import accumulate
from types import MappingProxyType
from typing import Final

from pydantic import JsonValue, TypeAdapter, ValidationError
from typing_extensions import NotRequired, ReadOnly, TypedDict

from litellm.constants import OTLP_MAX_ATTRIBUTE_VALUE_BYTES, OTLP_MAX_BODY_BYTES
from litellm.rust_bridge.traces import DecodedSpan
from litellm.rust_bridge.traces import decode_otlp as native_decode_otlp
from litellm.rust_bridge.traces import encode_error as native_encode_error
from litellm.tracing.normalizers.messages import content_text
from litellm.tracing.types import SpanRow, SpanType

_FRAMEWORK_SUFFIXES: Final = (
    ".wrap_model_call",
    ".wrap_tool_call",
    ".before_agent",
    ".after_agent",
    ".before_model",
    ".after_model",
)
_LLM_OPERATIONS: Final = frozenset({"chat", "text_completion", "generate_content"})
_LC_ROLES: Final = MappingProxyType({"human": "user", "ai": "assistant", "system": "system", "tool": "tool"})
_OPENINFERENCE_TYPES: Final[Mapping[str, SpanType]] = MappingProxyType({"AGENT": "agent", "LLM": "llm", "TOOL": "tool"})


_JSON: Final[TypeAdapter[JsonValue]] = TypeAdapter(JsonValue)
_MESSAGE_LIST: Final = TypeAdapter(tuple[dict[str, JsonValue], ...])
_MAX_JSON_ESCAPE_BYTES: Final = 6
_MAX_TOKENS: Final = (1 << 32) - 1


class InvalidOTLPPayloadError(ValueError):
    pass


class OTLPPayloadTooLargeError(OverflowError):
    pass


class MessageExtras(TypedDict):
    tool_calls: ReadOnly[NotRequired[JsonValue]]
    name: ReadOnly[NotRequired[str]]


class NormalizedMessage(MessageExtras):
    role: ReadOnly[str]
    content: ReadOnly[str]


class OTLPError(TypedDict):
    message: ReadOnly[str]


@dataclass(frozen=True, slots=True)
class NormalizedSpan:
    kind: SpanType
    agent: str = ""
    model: str = ""
    request_id: str = ""
    input: str = ""
    output: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    consumed: frozenset[str] = frozenset()


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
    normalized: Final = normalize(span)
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
            {key: _truncate(value) for key, value in attributes.items() if key not in normalized.consumed}
        ),
        Duration=span["end_ns"] - span["start_ns"],
        StatusCode=span["status_code"],
        StatusMessage=span["status_message"] or _exception_message(span),
        TeamId="",
        ApiKeyHash="",
        ObservationType=normalized.kind,
        AgentName=normalized.agent,
        Model=normalized.model,
        LiteLLMRequestId=attributes.get("gen_ai.response.id") or normalized.request_id,
        InputTokens=normalized.input_tokens,
        OutputTokens=normalized.output_tokens,
        Input=_truncate_payload(normalized.input),
        Output=_truncate(normalized.output),
    )


def _loads(value: str) -> JsonValue:
    if len(value.encode("utf-8")) > OTLP_MAX_BODY_BYTES:
        return None
    try:
        return _JSON.validate_json(value)
    except ValidationError:
        return None


def _text(value: JsonValue) -> str:
    return value if isinstance(value, str) else ""


def _message(value: JsonValue) -> NormalizedMessage | None:
    if not isinstance(value, dict):
        return None
    kwargs: Final = value.get("kwargs", value)
    if not isinstance(kwargs, dict):
        return None
    kind: Final = _text(kwargs.get("type")) or _text(kwargs.get("role"))
    if not kind:
        return None
    calls: Final = kwargs.get("tool_calls")
    if calls is not None and (not isinstance(calls, list) or not all(isinstance(call, dict) for call in calls)):
        return None
    role: Final = _LC_ROLES.get(kind, kind)
    content: Final = kwargs.get("content", "")
    name: Final = kwargs.get("name")
    tool_calls: Final = MessageExtras(tool_calls=calls) if calls else MessageExtras()
    tool_name: Final = MessageExtras(name=name) if role == "tool" and isinstance(name, str) else MessageExtras()
    message: Final[NormalizedMessage] = {
        "role": role,
        "content": content_text(content),
        **tool_calls,
        **tool_name,
    }
    return message


def _messages(value: JsonValue, raw: str) -> str:
    if not isinstance(value, list):
        return raw
    messages: Final = tuple(_message(item) for item in value)
    return json.dumps(messages) if all(message is not None for message in messages) else raw


def _langsmith_type(span: DecodedSpan) -> SpanType:
    attributes: Final = span["attributes"]
    kind: Final = attributes.get("langsmith.span.kind", "chain")
    if kind in ("llm", "tool"):
        return "llm" if kind == "llm" else "tool"
    if not span["parent_span_id"] or span["name"] == attributes.get("langsmith.metadata.lc_agent_name"):
        return "agent"
    return "framework" if span["name"].endswith(_FRAMEWORK_SUFFIXES) else "chain"


def _langsmith_io(kind: SpanType, attributes: Mapping[str, str]) -> tuple[str, str, str]:
    raw_prompt: Final = attributes.get("gen_ai.prompt", "")
    raw_completion: Final = attributes.get("gen_ai.completion", "")
    prompt: Final = _loads(raw_prompt)
    completion: Final = _loads(raw_completion)
    messages: Final = prompt.get("messages") if isinstance(prompt, dict) else None
    if kind == "llm":
        batch: Final = (
            messages[0] if isinstance(messages, list) and messages and isinstance(messages[0], list) else messages
        )
        generations: Final = completion.get("generations") if isinstance(completion, dict) else None
        first: Final = generations[0] if isinstance(generations, list) and generations else None
        item: Final = first[0] if isinstance(first, list) and first else first
        message: Final = item.get("message") if isinstance(item, dict) else None
        parsed: Final = _message(message)
        kwargs: Final = message.get("kwargs", message) if isinstance(message, dict) else None
        metadata: Final = kwargs.get("response_metadata") if isinstance(kwargs, dict) else None
        request_id: Final = _text(metadata.get("id")) if isinstance(metadata, dict) else ""
        return _messages(batch, raw_prompt), json.dumps(parsed) if parsed is not None else raw_completion, request_id
    if kind == "tool":
        output: Final = completion.get("output", completion) if isinstance(completion, dict) else completion
        update: Final = output.get("update") if isinstance(output, dict) else None
        updates: Final = update.get("messages") if isinstance(update, dict) else None
        final: Final = updates[-1] if isinstance(updates, list) and updates else output
        content: Final = final.get("content", final) if isinstance(final, dict) else final
        return (
            raw_prompt,
            (content if isinstance(content, str) else json.dumps(content)) if content is not None else raw_completion,
            "",
        )
    if kind == "agent":
        outputs: Final = completion.get("messages") if isinstance(completion, dict) else None
        last: Final = _message(outputs[-1]) if isinstance(outputs, list) and outputs else None
        return _messages(messages, raw_prompt), json.dumps(last) if last is not None else raw_completion, ""
    return raw_prompt, raw_completion, ""


def _to_int(value: str | None) -> int:
    try:
        number: Final = int(value) if value else 0
    except ValueError:
        return 0
    if not 0 <= number <= _MAX_TOKENS:
        raise InvalidOTLPPayloadError("OTLP token count is outside the storage range")
    return number


def normalize(span: DecodedSpan) -> NormalizedSpan:
    attributes: Final = span["attributes"]
    fallback: Final[SpanType] = "agent" if not span["parent_span_id"] else "chain"
    input_tokens: Final = _to_int(attributes.get("gen_ai.usage.input_tokens"))
    output_tokens: Final = _to_int(attributes.get("gen_ai.usage.output_tokens"))
    if span["scope_name"] == "langsmith" or "langsmith.span.kind" in attributes:
        kind: Final = _langsmith_type(span)
        prompt, completion, request_id = _langsmith_io(kind, attributes)
        return NormalizedSpan(
            kind,
            attributes.get("langsmith.metadata.lc_agent_name", ""),
            attributes.get("gen_ai.request.model", ""),
            request_id,
            prompt,
            completion,
            input_tokens,
            output_tokens,
            frozenset({"gen_ai.prompt", "gen_ai.completion"}),
        )
    if "openinference.span.kind" in attributes:
        return NormalizedSpan(
            _OPENINFERENCE_TYPES.get(attributes["openinference.span.kind"].upper(), fallback),
            attributes.get("agent.name", ""),
            attributes.get("llm.model_name", ""),
            "",
            attributes.get("input.value", ""),
            attributes.get("output.value", ""),
            _to_int(attributes.get("llm.token_count.prompt"))
            if "llm.token_count.prompt" in attributes
            else input_tokens,
            _to_int(attributes.get("llm.token_count.completion"))
            if "llm.token_count.completion" in attributes
            else output_tokens,
            frozenset({"input.value", "output.value"}),
        )
    operation: Final = attributes.get("gen_ai.operation.name", "")
    genai_kind: Final[SpanType] = (
        "llm"
        if operation in _LLM_OPERATIONS
        else "tool"
        if operation == "execute_tool"
        else "agent"
        if operation == "invoke_agent"
        else fallback
    )
    input_key: Final = (
        "gen_ai.input.messages" if attributes.get("gen_ai.input.messages") else "gen_ai.tool.call.arguments"
    )
    output_key: Final = (
        "gen_ai.output.messages" if attributes.get("gen_ai.output.messages") else "gen_ai.tool.call.result"
    )
    return NormalizedSpan(
        genai_kind,
        attributes.get("gen_ai.agent.name", ""),
        attributes.get("gen_ai.request.model") or attributes.get("gen_ai.response.model", ""),
        "",
        attributes.get(input_key, ""),
        attributes.get(output_key, ""),
        input_tokens,
        output_tokens,
        frozenset({input_key, output_key}),
    )


def encode_otlp_response(content_type: str | None, error: str | None = None) -> tuple[bytes, str]:
    media_type: Final = (content_type or "application/x-protobuf").split(";", 1)[0].strip().lower()
    if media_type == "application/json":
        response: Final[OTLPError] = {"message": error or ""}
        return (json.dumps(response).encode() if error else b"{}"), "application/json"
    if error is None:
        return b"", "application/x-protobuf"
    return native_encode_error(error), "application/x-protobuf"
