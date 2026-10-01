"""
OTLP/HTTP trace export -> `SpanRow`s.

Pure functions, no I/O. Two steps:
1. `decode_otlp()`   protobuf / JSON / gzip `ExportTraceServiceRequest` -> flat spans
2. `normalize()`     framework conventions -> LiteLLM columns (type, agent, input/output,
                     LiteLLM request id). Supported: LangSmith (LangChain, LangGraph,
                     Deep Agents), OTEL GenAI semconv, OpenInference.
"""

from collections.abc import Mapping
from typing import Final

from litellm.constants import OTLP_MAX_ATTRIBUTE_VALUE_BYTES, OTLP_MAX_BODY_BYTES
from litellm.rust_bridge.traces import DecodedSpan
from litellm.rust_bridge.traces import decode_otlp as native_decode_otlp
from litellm.tracing.normalizers import select_normalizer
from litellm.tracing.normalizers.base import to_int
from litellm.tracing.types import SpanRow

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
    row["SpanAttributes"] = {  # mutable-ok: the Rust JSON bridge requires a plain dict for span attributes
        k: _truncate(v) for k, v in attributes.items() if k not in _HEAVY_ATTRIBUTES
    }
    row["Input"], row["Output"] = _truncate(row["Input"]), _truncate(row["Output"])
    return row


# ---------------------------------------------------------------- normalize


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
