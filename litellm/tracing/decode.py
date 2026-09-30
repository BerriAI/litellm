"""
OTLP/HTTP trace export -> `SpanRow`s.

Pure functions, no I/O. Two steps:
1. `decode_otlp()`   protobuf / JSON / gzip `ExportTraceServiceRequest` -> flat spans
2. `normalize()`     framework conventions -> LiteLLM columns (type, agent, input/output,
                     LiteLLM request id). Supported: LangSmith (LangChain, LangGraph,
                     Deep Agents), OTEL GenAI semconv, OpenInference.
"""

import base64
import gzip
import json
from collections.abc import Callable, Mapping
from typing import Any, Final

from google.protobuf.json_format import Parse  # type: ignore[import-untyped]
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (  # type: ignore[import-untyped]
    ExportTraceServiceRequest,
)
from opentelemetry.proto.common.v1.common_pb2 import AnyValue  # type: ignore[import-untyped]
from opentelemetry.proto.trace.v1.trace_pb2 import Span as OtlpSpan  # type: ignore[import-untyped]
from opentelemetry.proto.trace.v1.trace_pb2 import Status  # type: ignore[import-untyped]

from litellm.constants import OTLP_MAX_ATTRIBUTE_VALUE_BYTES
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
_LC_ROLES: Final = {"human": "user", "ai": "assistant", "system": "system", "tool": "tool"}
_OPENINFERENCE_TYPES: Final[dict[str, SpanType]] = {"AGENT": "agent", "LLM": "llm", "TOOL": "tool"}


# ---------------------------------------------------------------- decode


def _any_value(value: AnyValue) -> str:
    field = value.WhichOneof("value")
    if field is None:
        return ""
    if field == "bytes_value":
        return value.bytes_value.decode("utf-8", "replace")
    if field == "array_value":
        return json.dumps([_any_value(v) for v in value.array_value.values])
    if field == "kvlist_value":
        return json.dumps({kv.key: _any_value(kv.value) for kv in value.kvlist_value.values})
    if field == "bool_value":
        return "true" if value.bool_value else "false"
    return str(getattr(value, field))


def _truncate(value: str) -> str:
    size = len(value.encode("utf-8"))
    if size <= OTLP_MAX_ATTRIBUTE_VALUE_BYTES:
        return value
    kept = value.encode("utf-8")[:OTLP_MAX_ATTRIBUTE_VALUE_BYTES].decode("utf-8", "ignore")
    return f"{kept}…[truncated {size - OTLP_MAX_ATTRIBUTE_VALUE_BYTES} bytes]"


def _attributes(key_values) -> dict[str, str]:
    return {kv.key: _any_value(kv.value) for kv in key_values}


def _parse_request(body: bytes, content_type: str | None, content_encoding: str | None) -> ExportTraceServiceRequest:
    if content_encoding == "gzip" or body[:2] == b"\x1f\x8b":
        body = gzip.decompress(body)
    request = ExportTraceServiceRequest()
    if content_type and "json" in content_type:
        Parse(body.decode("utf-8"), request, ignore_unknown_fields=True)
    else:
        request.ParseFromString(body)
    return request


def decode_otlp(body: bytes, content_type: str | None = None, content_encoding: str | None = None) -> list[SpanRow]:
    """Decode an OTLP trace export and normalize every span."""
    request = _parse_request(body, content_type, content_encoding)
    rows: list[SpanRow] = []
    for resource_spans in request.resource_spans:
        resource = _attributes(resource_spans.resource.attributes)
        for scope_spans in resource_spans.scope_spans:
            for span in scope_spans.spans:
                rows.append(_span_row(span, resource, scope_spans.scope.name, scope_spans.scope.version))
    return rows


def _exception_message(span: OtlpSpan) -> str:
    """`span.record_exception()` writes an `exception` event; surface it when status.message is empty."""
    for event in span.events:
        if event.name == "exception":
            attributes = _attributes(event.attributes)
            return attributes.get("exception.message") or attributes.get("exception.type", "")
    return ""


def _span_row(span: OtlpSpan, resource: dict[str, str], scope_name: str, scope_version: str) -> SpanRow:
    attributes = _attributes(span.attributes)
    row = SpanRow(
        Timestamp=span.start_time_unix_nano,
        TraceId=span.trace_id.hex(),
        SpanId=span.span_id.hex(),
        ParentSpanId=span.parent_span_id.hex(),
        TraceState=span.trace_state,
        SpanName=span.name,
        SpanKind=OtlpSpan.SpanKind.Name(span.kind),
        ServiceName=resource.get("service.name", ""),
        ResourceAttributes=resource,
        ScopeName=scope_name,
        ScopeVersion=scope_version,
        SpanAttributes={},
        Duration=max(span.end_time_unix_nano - span.start_time_unix_nano, 0),
        StatusCode=Status.StatusCode.Name(span.status.code),
        StatusMessage=span.status.message or _exception_message(span),
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
    row["Input"], row["Output"] = _truncate(row["Input"]), _truncate(row["Output"])
    return row


# ---------------------------------------------------------------- normalize


def _loads(value: str) -> Any:
    try:
        return json.loads(value)
    except (ValueError, TypeError):
        return None


def _lc_message(message: Mapping[str, Any]) -> dict[str, Any]:
    """LangChain serialized message (or plain {role, content}) -> {role, content, tool_calls?}."""
    kwargs = message.get("kwargs", message)
    role = _LC_ROLES.get(kwargs.get("type") or kwargs.get("role"), kwargs.get("role") or kwargs.get("type") or "")
    content = kwargs.get("content", "")
    out: dict[str, Any] = {"role": role, "content": content if isinstance(content, str) else json.dumps(content)}
    if kwargs.get("tool_calls"):
        out["tool_calls"] = [{"name": t.get("name"), "args": t.get("args")} for t in kwargs["tool_calls"]]
    if role == "tool" and kwargs.get("name"):
        out["name"] = kwargs["name"]
    return out


def _langsmith_type(row: SpanRow, attributes: Mapping[str, str]) -> SpanType:
    kind = attributes.get("langsmith.span.kind", "chain")
    name = row["SpanName"]
    if not row["ParentSpanId"] or name == attributes.get("langsmith.metadata.lc_agent_name"):
        return "agent"
    if kind in ("llm", "tool"):
        return kind  # type: ignore[return-value]
    if name.endswith(_FRAMEWORK_SUFFIXES):
        return "framework"
    return "chain"


def _langsmith_io(row: SpanRow, attributes: Mapping[str, str]) -> None:
    prompt = _loads(attributes.get("gen_ai.prompt", ""))
    completion = _loads(attributes.get("gen_ai.completion", ""))
    if row["ObservationType"] == "llm" and isinstance(completion, dict):
        messages = (prompt or {}).get("messages") or [[]]
        batch = messages[0] if messages and isinstance(messages[0], list) else messages
        row["Input"] = json.dumps([_lc_message(m) for m in batch])
        generation = completion["generations"][0][0]["message"]["kwargs"]
        row["Output"] = json.dumps(_lc_message({"kwargs": generation}))
        row["LiteLLMRequestId"] = (generation.get("response_metadata") or {}).get("id") or ""
        return
    if row["ObservationType"] == "tool":
        output = (completion or {}).get("output", completion) if isinstance(completion, dict) else completion
        if isinstance(output, dict) and "update" in output:  # LangGraph Command, e.g. Deep Agents `task`
            update_messages = (output.get("update") or {}).get("messages") or []
            output = update_messages[-1] if update_messages else output
        if isinstance(output, dict):
            output = output.get("content", output)
        row["Input"] = attributes.get("gen_ai.prompt", "")
        row["Output"] = output if isinstance(output, str) else json.dumps(output)
        return
    if row["ObservationType"] == "agent":
        input_messages = (prompt or {}).get("messages") if isinstance(prompt, dict) else None
        output_messages = (completion or {}).get("messages") if isinstance(completion, dict) else None
        # agents built with @traceable take arbitrary args, not a message list: keep the raw payload then
        row["Input"] = (
            json.dumps([_lc_message(m) for m in input_messages if isinstance(m, dict)])
            if input_messages
            else attributes.get("gen_ai.prompt", "")
        )
        row["Output"] = (
            json.dumps(_lc_message(output_messages[-1]))
            if output_messages and isinstance(output_messages[-1], dict)
            else attributes.get("gen_ai.completion", "")
        )
        return
    row["Input"] = attributes.get("gen_ai.prompt", "")
    row["Output"] = attributes.get("gen_ai.completion", "")


def normalize_langsmith(row: SpanRow, attributes: Mapping[str, str]) -> None:
    row["ObservationType"] = _langsmith_type(row, attributes)
    row["AgentName"] = attributes.get("langsmith.metadata.lc_agent_name", "")
    row["Model"] = attributes.get("gen_ai.request.model", "")
    _langsmith_io(row, attributes)


def normalize_genai(row: SpanRow, attributes: Mapping[str, str]) -> None:
    operation = attributes.get("gen_ai.operation.name", "")
    if operation == "invoke_agent" or not row["ParentSpanId"]:
        row["ObservationType"] = "agent"
    elif operation in _LLM_OPERATIONS:
        row["ObservationType"] = "llm"
    elif operation == "execute_tool":
        row["ObservationType"] = "tool"
    row["AgentName"] = attributes.get("gen_ai.agent.name", "")
    row["Model"] = attributes.get("gen_ai.request.model") or attributes.get("gen_ai.response.model", "")
    row["LiteLLMRequestId"] = attributes.get("gen_ai.response.id", "")
    row["Input"] = attributes.get("gen_ai.input.messages") or attributes.get("gen_ai.tool.call.arguments", "")
    row["Output"] = attributes.get("gen_ai.output.messages") or attributes.get("gen_ai.tool.call.result", "")


def normalize_openinference(row: SpanRow, attributes: Mapping[str, str]) -> None:
    kind = attributes.get("openinference.span.kind", "").upper()
    row["ObservationType"] = _OPENINFERENCE_TYPES.get(kind, "agent" if not row["ParentSpanId"] else "chain")
    row["AgentName"] = attributes.get("agent.name", "")
    row["Model"] = attributes.get("llm.model_name", "")
    row["Input"] = attributes.get("input.value", "")
    row["Output"] = attributes.get("output.value", "")
    row["InputTokens"] = _to_int(attributes.get("llm.token_count.prompt"))
    row["OutputTokens"] = _to_int(attributes.get("llm.token_count.completion"))


def _set_tokens(row: SpanRow, attributes: Mapping[str, str]) -> None:
    row["InputTokens"] = _to_int(attributes.get("gen_ai.usage.input_tokens"))
    row["OutputTokens"] = _to_int(attributes.get("gen_ai.usage.output_tokens"))


def _to_int(value: str | None) -> int:
    try:
        return int(value) if value else 0
    except ValueError:
        return 0


def select_normalizer(scope_name: str, attributes: Mapping[str, str]) -> Callable[[SpanRow, Mapping[str, str]], None]:
    if scope_name == "langsmith" or "langsmith.span.kind" in attributes:
        return normalize_langsmith
    if "openinference.span.kind" in attributes:
        return normalize_openinference
    return normalize_genai


def normalize(row: SpanRow, attributes: Mapping[str, str]) -> None:
    select_normalizer(row["ScopeName"], attributes)(row, attributes)
    if not row["InputTokens"] and not row["OutputTokens"]:
        _set_tokens(row, attributes)


def encode_otlp_response(content_type: str | None) -> tuple[bytes, str]:
    """Empty ExportTraceServiceResponse in the caller's encoding."""
    if content_type and "json" in content_type:
        return b"{}", "application/json"
    return b"", "application/x-protobuf"


def hex_id(b64: str) -> str:
    """base64 (OTLP/JSON id encoding) -> hex."""
    return base64.b64decode(b64).hex() if b64 else ""
