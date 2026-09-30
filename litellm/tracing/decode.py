"""
OTLP/HTTP trace export -> `SpanRow`s.

Pure functions, no I/O. Two steps:
1. `decode_otlp()`   protobuf / JSON / gzip `ExportTraceServiceRequest` -> flat spans
2. `normalize()`     framework conventions -> LiteLLM columns (type, agent, input/output,
                     LiteLLM request id). Supported: LangSmith (LangChain, LangGraph,
                     Deep Agents), OTEL GenAI semconv, OpenInference.
"""

import json
from collections.abc import Callable, Mapping
from typing import Any, Final
from types import MappingProxyType
from typing_extensions import ReadOnly, TypedDict

from litellm.constants import OTLP_MAX_ATTRIBUTE_VALUE_BYTES, OTLP_MAX_BODY_BYTES
from litellm.rust_bridge.traces import DecodedSpan
from litellm.rust_bridge.traces import decode_otlp as native_decode_otlp
from litellm.tracing.types import SpanRow, SpanType

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
# LangChain / Deep Agents middleware wrappers: real spans, but noise in the UI
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


class InvalidOTLPPayloadError(ValueError):
    pass


class OTLPPayloadTooLargeError(OverflowError):
    pass


# ---------------------------------------------------------------- decode


def _truncate(value: str) -> str:
    size = len(value.encode("utf-8"))
    if size <= OTLP_MAX_ATTRIBUTE_VALUE_BYTES:
        return value
    kept = value.encode("utf-8")[:OTLP_MAX_ATTRIBUTE_VALUE_BYTES].decode("utf-8", "ignore")
    return f"{kept}…[truncated {size - OTLP_MAX_ATTRIBUTE_VALUE_BYTES} bytes]"


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
        SpanAttributes=span["attributes"],
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
    normalized: Final = normalize(row, attributes)
    result: Final[SpanRow] = {
        **normalized,
        "SpanAttributes": MappingProxyType(
            {k: _truncate(v) for k, v in attributes.items() if k not in _HEAVY_ATTRIBUTES}
        ),
        "Input": _truncate(normalized["Input"]),
        "Output": _truncate(normalized["Output"]),
    }
    return result


# ---------------------------------------------------------------- normalize


def _loads(value: str) -> object:
    try:
        return json.loads(value)
    except (ValueError, TypeError):
        return None


class ToolCall(TypedDict):
    name: ReadOnly[object]
    args: ReadOnly[object]


def _lc_message(message: Mapping[str, Any]) -> dict[str, Any]:
    kwargs: Final = message.get("kwargs", message)
    role: Final = _LC_ROLES.get(
        kwargs.get("type") or kwargs.get("role"), kwargs.get("role") or kwargs.get("type") or ""
    )
    content: Final = kwargs.get("content", "")
    base: Final = (("role", role), ("content", content if isinstance(content, str) else json.dumps(content)))
    calls: Final = (
        (("tool_calls", tuple(ToolCall(name=t.get("name"), args=t.get("args")) for t in kwargs["tool_calls"])),)
        if kwargs.get("tool_calls")
        else ()
    )
    tool: Final = (("name", kwargs["name"]),) if role == "tool" and kwargs.get("name") else ()
    return dict((*base, *calls, *tool))  # mutable-ok: json.dumps requires a dict for the normalized message object


def _langsmith_type(row: SpanRow, attributes: Mapping[str, str]) -> SpanType:
    kind = attributes.get("langsmith.span.kind", "chain")
    name = row["SpanName"]
    if not row["ParentSpanId"] or name == attributes.get("langsmith.metadata.lc_agent_name"):
        return "agent"
    if kind in ("llm", "tool"):
        return kind
    if name.endswith(_FRAMEWORK_SUFFIXES):
        return "framework"
    return "chain"


def _langsmith_io(row: SpanRow, attributes: Mapping[str, str]) -> tuple[str, str, str]:
    prompt: Final = _loads(attributes.get("gen_ai.prompt", ""))
    completion: Final = _loads(attributes.get("gen_ai.completion", ""))
    prompt_payload: Final = prompt if isinstance(prompt, dict) else MappingProxyType({})
    if row["ObservationType"] == "llm" and isinstance(completion, dict):
        messages: Final = prompt_payload.get("messages") or ((),)
        batch: Final = messages[0] if messages and isinstance(messages[0], list) else messages
        input_text: Final = (
            json.dumps(tuple(_lc_message(m) for m in batch if isinstance(m, dict)))
            if isinstance(batch, (tuple, list))
            else ""
        )
        generations: Final = completion.get("generations")
        first: Final = generations[0] if isinstance(generations, list) and generations else None
        item: Final = first[0] if isinstance(first, list) and first else None
        message: Final = item.get("message") if isinstance(item, dict) else None
        generation: Final = message.get("kwargs") if isinstance(message, dict) else None
        if isinstance(generation, dict):
            metadata: Final = generation.get("response_metadata")
            return (
                input_text,
                json.dumps(_lc_message(MappingProxyType({"kwargs": generation}))),
                metadata.get("id", "") if isinstance(metadata, dict) else "",
            )
        return input_text, attributes.get("gen_ai.completion", ""), row["LiteLLMRequestId"]
    if row["ObservationType"] == "tool":
        output: Final = completion.get("output", completion) if isinstance(completion, dict) else completion
        update_messages: Final = (
            (output.get("update") or MappingProxyType({})).get("messages") or ()
            if isinstance(output, dict) and "update" in output
            else ()
        )
        latest: Final = update_messages[-1] if update_messages else output
        content: Final = latest.get("content", latest) if isinstance(latest, dict) else latest
        return (
            attributes.get("gen_ai.prompt", ""),
            content if isinstance(content, str) else json.dumps(content),
            row["LiteLLMRequestId"],
        )
    if row["ObservationType"] == "agent":
        input_messages: Final = prompt.get("messages") if isinstance(prompt, dict) else None
        output_messages: Final = completion.get("messages") if isinstance(completion, dict) else None
        input_text: Final = (
            json.dumps(tuple(_lc_message(m) for m in input_messages if isinstance(m, dict)))
            if input_messages
            else attributes.get("gen_ai.prompt", "")
        )
        output_text: Final = (
            json.dumps(_lc_message(output_messages[-1]))
            if output_messages and isinstance(output_messages[-1], dict)
            else attributes.get("gen_ai.completion", "")
        )
        return input_text, output_text, row["LiteLLMRequestId"]
    return attributes.get("gen_ai.prompt", ""), attributes.get("gen_ai.completion", ""), row["LiteLLMRequestId"]


def normalize_langsmith(row: SpanRow, attributes: Mapping[str, str]) -> SpanRow:
    named: Final[SpanRow] = {
        **row,
        "ObservationType": _langsmith_type(row, attributes),
        "AgentName": attributes.get("langsmith.metadata.lc_agent_name", ""),
        "Model": attributes.get("gen_ai.request.model", ""),
    }
    input_text, output_text, request_id = _langsmith_io(named, attributes)
    result: Final[SpanRow] = {**named, "Input": input_text, "Output": output_text, "LiteLLMRequestId": request_id}
    return result


def normalize_genai(row: SpanRow, attributes: Mapping[str, str]) -> SpanRow:
    operation: Final = attributes.get("gen_ai.operation.name", "")
    kind: Final[SpanType] = (
        "agent"
        if operation == "invoke_agent" or not row["ParentSpanId"]
        else "llm"
        if operation in _LLM_OPERATIONS
        else "tool"
        if operation == "execute_tool"
        else row["ObservationType"]
    )
    result: Final[SpanRow] = {
        **row,
        "ObservationType": kind,
        "AgentName": attributes.get("gen_ai.agent.name", ""),
        "Model": attributes.get("gen_ai.request.model") or attributes.get("gen_ai.response.model", ""),
        "LiteLLMRequestId": attributes.get("gen_ai.response.id", ""),
        "Input": attributes.get("gen_ai.input.messages") or attributes.get("gen_ai.tool.call.arguments", ""),
        "Output": attributes.get("gen_ai.output.messages") or attributes.get("gen_ai.tool.call.result", ""),
    }
    return result


def normalize_openinference(row: SpanRow, attributes: Mapping[str, str]) -> SpanRow:
    kind: Final = attributes.get("openinference.span.kind", "").upper()
    result: Final[SpanRow] = {
        **row,
        "ObservationType": _OPENINFERENCE_TYPES.get(kind, "agent" if not row["ParentSpanId"] else "chain"),
        "AgentName": attributes.get("agent.name", ""),
        "Model": attributes.get("llm.model_name", ""),
        "Input": attributes.get("input.value", ""),
        "Output": attributes.get("output.value", ""),
        "InputTokens": _to_int(attributes.get("llm.token_count.prompt")),
        "OutputTokens": _to_int(attributes.get("llm.token_count.completion")),
    }
    return result


def _to_int(value: str | None) -> int:
    try:
        return int(value) if value else 0
    except ValueError:
        return 0


def select_normalizer(
    scope_name: str, attributes: Mapping[str, str]
) -> Callable[[SpanRow, Mapping[str, str]], SpanRow]:
    if scope_name == "langsmith" or "langsmith.span.kind" in attributes:
        return normalize_langsmith
    if "openinference.span.kind" in attributes:
        return normalize_openinference
    return normalize_genai


def normalize(row: SpanRow, attributes: Mapping[str, str]) -> SpanRow:
    normalized: Final = select_normalizer(row["ScopeName"], attributes)(row, attributes)
    if normalized["InputTokens"] or normalized["OutputTokens"]:
        return normalized
    result: Final[SpanRow] = {
        **normalized,
        "InputTokens": _to_int(attributes.get("gen_ai.usage.input_tokens")),
        "OutputTokens": _to_int(attributes.get("gen_ai.usage.output_tokens")),
    }
    return result


def encode_otlp_response(content_type: str | None) -> tuple[bytes, str]:
    """Empty ExportTraceServiceResponse in the caller's encoding."""
    if content_type and "json" in content_type:
        return b"{}", "application/json"
    return b"", "application/x-protobuf"
