"""
Tests for OTLP decode + normalization (litellm/tracing/decode.py).

The fixture is a trimmed real export from a Deep Agents run (LangSmith OTEL mode):
deep_research_agent -> task (tool) -> researcher (subagent) -> search_docs (tool).
"""

import gzip
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.path.abspath("../../.."))

import pytest
from google.protobuf.json_format import Parse
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
from opentelemetry.proto.common.v1.common_pb2 import AnyValue, KeyValue
from opentelemetry.proto.trace.v1.trace_pb2 import ResourceSpans, ScopeSpans, Span, Status

from litellm.tracing import decode
from litellm.tracing.decode import decode_otlp, encode_otlp_response

pytestmark = pytest.mark.requires_rust_extension

FIXTURE = Path(__file__).parent / "fixtures" / "langsmith_deep_agent_export.json"
TRACE_ID = "4bad42b84e9de3ba46fc870185f8f023"


def _fixture_json() -> bytes:
    return FIXTURE.read_bytes()


def _fixture_protobuf() -> bytes:
    request = ExportTraceServiceRequest()
    Parse(_fixture_json().decode(), request)
    return request.SerializeToString()


@pytest.fixture
def rows_by_name() -> dict:
    rows = decode_otlp(_fixture_json(), "application/json")
    return {r["SpanName"]: r for r in rows}


def _kv(key: str, value: str | int) -> KeyValue:
    if isinstance(value, int):
        return KeyValue(key=key, value=AnyValue(int_value=value))
    return KeyValue(key=key, value=AnyValue(string_value=value))


def _export(*spans: Span, service: str = "svc", scope: str = "test") -> bytes:
    resource_spans = ResourceSpans(scope_spans=[ScopeSpans(spans=list(spans))])
    resource_spans.resource.attributes.append(_kv("service.name", service))
    resource_spans.scope_spans[0].scope.name = scope
    return ExportTraceServiceRequest(resource_spans=[resource_spans]).SerializeToString()


def _span(name: str, span_id: bytes, parent: bytes = b"", **attributes: str | int) -> Span:
    return Span(
        trace_id=bytes.fromhex(TRACE_ID),
        span_id=span_id,
        parent_span_id=parent,
        name=name,
        start_time_unix_nano=1_000,
        end_time_unix_nano=5_000,
        attributes=[_kv(k.replace("__", "."), v) for k, v in attributes.items()],
    )


# ---------------------------------------------------------------- LangSmith / Deep Agents fixture


def test_classifies_every_langsmith_span(rows_by_name):
    assert {name: r["ObservationType"] for name, r in rows_by_name.items()} == {
        "deep_research_agent": "agent",
        "ChatOpenAI": "llm",
        "FilesystemMiddleware.wrap_model_call": "framework",
        "task": "tool",
        "researcher": "agent",
        "search_docs": "tool",
    }


def test_agent_name_is_the_enclosing_agent(rows_by_name):
    assert rows_by_name["task"]["AgentName"] == "deep_research_agent"
    assert rows_by_name["ChatOpenAI"]["AgentName"] == "deep_research_agent"
    assert rows_by_name["researcher"]["AgentName"] == "researcher"
    assert rows_by_name["search_docs"]["AgentName"] == "researcher"


def test_subagent_is_nested_under_task_tool(rows_by_name):
    assert rows_by_name["researcher"]["ParentSpanId"] == rows_by_name["task"]["SpanId"]
    assert rows_by_name["deep_research_agent"]["ParentSpanId"] == ""


def test_llm_span_carries_litellm_request_id_model_and_tokens(rows_by_name):
    llm = rows_by_name["ChatOpenAI"]
    assert llm["LiteLLMRequestId"] == "chatcmpl-4077bb36-9380-4a3b-9481-245700cef09a"
    assert llm["Model"] == "claude-sonnet-4-5"
    assert (llm["InputTokens"], llm["OutputTokens"]) == (3332, 467)


def test_llm_input_output_are_normalized_messages(rows_by_name):
    llm = rows_by_name["ChatOpenAI"]
    messages = json.loads(llm["Input"])
    assert [m["role"] for m in messages][:2] == ["system", "user"]
    assert "research lead" in messages[0]["content"]
    output = json.loads(llm["Output"])
    assert output["role"] == "assistant"
    assert output["tool_calls"][0]["name"]


def test_task_tool_output_is_subagent_final_message_text(rows_by_name):
    task = rows_by_name["task"]
    assert json.loads(task["Input"])["subagent_type"] == "researcher"
    assert task["Output"].startswith("Based on my research")
    assert not task["Output"].startswith("{")


def test_agent_input_output(rows_by_name):
    root = rows_by_name["deep_research_agent"]
    assert json.loads(root["Input"]) == [
        {"role": "user", "content": "Should we store OTEL agent spans in ClickHouse or Postgres at 50k spans/sec?"}
    ]
    assert json.loads(root["Output"])["role"] == "assistant"


def test_plain_tool_input_output(rows_by_name):
    tool = rows_by_name["search_docs"]
    assert json.loads(tool["Input"]) == {"query": "ClickHouse Postgres OpenTelemetry OTEL spans performance comparison"}
    assert tool["Output"].startswith("ClickHouse ingests")


def test_heavy_attributes_are_lifted_out_of_span_attributes(rows_by_name):
    for row in rows_by_name.values():
        assert not set(row["SpanAttributes"]) & decode._HEAVY_ATTRIBUTES
    assert rows_by_name["ChatOpenAI"]["SpanAttributes"]["langsmith.span.kind"] == "llm"


def test_ids_are_hex_and_resource_is_kept(rows_by_name):
    root = rows_by_name["deep_research_agent"]
    assert root["TraceId"] == TRACE_ID
    assert root["SpanId"] == "5e79f3b5b504985e"
    assert root["ServiceName"] == "agent-demo"
    assert root["ScopeName"] == "langsmith"
    assert root["SpanKind"] == "SPAN_KIND_INTERNAL"
    assert root["StatusCode"] == "STATUS_CODE_OK"
    assert root["Duration"] > 0


def test_protobuf_and_json_decode_identically():
    from_json = decode_otlp(_fixture_json(), "application/json")
    from_protobuf = decode_otlp(_fixture_protobuf(), "application/x-protobuf")
    assert from_json == from_protobuf
    assert len(from_json) == 6


def test_content_type_defaults_to_protobuf():
    assert len(decode_otlp(_fixture_protobuf(), None)) == 6


@pytest.mark.parametrize("content_encoding", ["gzip", None])
def test_gzip_body_by_header_or_magic_bytes(content_encoding):
    rows = decode_otlp(gzip.compress(_fixture_protobuf()), "application/x-protobuf", content_encoding)
    assert len(rows) == 6


def test_long_values_are_truncated_with_marker():
    with patch.object(decode, "OTLP_MAX_ATTRIBUTE_VALUE_BYTES", 100):
        rows = {r["SpanName"]: r for r in decode_otlp(_fixture_json(), "application/json")}
    task = rows["task"]
    assert "…[truncated " in task["Input"]
    assert task["Input"].encode().startswith(task["Input"].split("…")[0].encode())
    assert len(task["Input"].split("…")[0].encode()) <= 100


# ---------------------------------------------------------------- status / exceptions


def test_exception_event_fills_status_message():
    span = _span("get_customer_plan", b"\x01" * 8, b"\x02" * 8)
    span.status.CopyFrom(Status(code=Status.STATUS_CODE_ERROR))
    event = span.events.add()
    event.name = "exception"
    event.attributes.extend(
        [_kv("exception.type", "KeyError"), _kv("exception.message", "customer acme-404 not found")]
    )
    (row,) = decode_otlp(_export(span))
    assert row["StatusCode"] == "STATUS_CODE_ERROR"
    assert row["StatusMessage"] == "customer acme-404 not found"


def test_status_message_wins_over_exception_event():
    span = _span("tool", b"\x01" * 8, b"\x02" * 8)
    span.status.CopyFrom(Status(code=Status.STATUS_CODE_ERROR, message="boom"))
    event = span.events.add()
    event.name = "exception"
    event.attributes.append(_kv("exception.message", "other"))
    (row,) = decode_otlp(_export(span))
    assert row["StatusMessage"] == "boom"


# ---------------------------------------------------------------- GenAI semconv / OpenInference


def test_genai_semconv_spans():
    root = _span(
        "invoke_agent planner", b"\x01" * 8, gen_ai__operation__name="invoke_agent", gen_ai__agent__name="planner"
    )
    chat = _span(
        "chat gpt-4o",
        b"\x02" * 8,
        b"\x01" * 8,
        gen_ai__operation__name="chat",
        gen_ai__agent__name="planner",
        gen_ai__request__model="gpt-4o",
        gen_ai__response__id="chatcmpl-abc",
        gen_ai__usage__input_tokens=12,
        gen_ai__usage__output_tokens=3,
        gen_ai__input__messages='[{"role":"user","content":"hi"}]',
        gen_ai__output__messages='[{"role":"assistant","content":"hello"}]',
    )
    tool = _span(
        "execute_tool search",
        b"\x03" * 8,
        b"\x01" * 8,
        gen_ai__operation__name="execute_tool",
        gen_ai__tool__call__arguments='{"q":"x"}',
        gen_ai__tool__call__result="found",
    )
    rows = {r["SpanName"]: r for r in decode_otlp(_export(root, chat, tool))}
    assert rows["invoke_agent planner"]["ObservationType"] == "agent"
    assert rows["invoke_agent planner"]["AgentName"] == "planner"
    llm = rows["chat gpt-4o"]
    assert (llm["ObservationType"], llm["Model"], llm["LiteLLMRequestId"]) == ("llm", "gpt-4o", "chatcmpl-abc")
    assert (llm["InputTokens"], llm["OutputTokens"]) == (12, 3)
    assert json.loads(llm["Input"])[0]["content"] == "hi"
    assert "gen_ai.input.messages" not in llm["SpanAttributes"]
    assert (rows["execute_tool search"]["ObservationType"], rows["execute_tool search"]["Output"]) == ("tool", "found")


def test_openinference_spans():
    root = _span("agent", b"\x01" * 8, openinference__span__kind="AGENT", agent__name="writer", input__value="task")
    llm = _span(
        "llm",
        b"\x02" * 8,
        b"\x01" * 8,
        openinference__span__kind="LLM",
        llm__model_name="claude-sonnet-4-5",
        llm__token_count__prompt=40,
        llm__token_count__completion=8,
        input__value="prompt",
        output__value="answer",
    )
    chain = _span("retriever", b"\x03" * 8, b"\x01" * 8, openinference__span__kind="RETRIEVER")
    rows = {r["SpanName"]: r for r in decode_otlp(_export(root, llm, chain))}
    assert (rows["agent"]["ObservationType"], rows["agent"]["AgentName"], rows["agent"]["Input"]) == (
        "agent",
        "writer",
        "task",
    )
    assert rows["llm"]["ObservationType"] == "llm"
    assert (rows["llm"]["Model"], rows["llm"]["InputTokens"], rows["llm"]["OutputTokens"]) == (
        "claude-sonnet-4-5",
        40,
        8,
    )
    assert (rows["llm"]["Input"], rows["llm"]["Output"]) == ("prompt", "answer")
    assert "input.value" not in rows["llm"]["SpanAttributes"]
    assert rows["retriever"]["ObservationType"] == "chain"


def test_non_string_attribute_values_are_stringified():
    span = _span("root", b"\x01" * 8)
    span.attributes.extend(
        [
            KeyValue(key="flag", value=AnyValue(bool_value=True)),
            KeyValue(key="ratio", value=AnyValue(double_value=0.5)),
            KeyValue(key="raw", value=AnyValue(bytes_value=b"abc")),
        ]
    )
    array = KeyValue(key="list")
    array.value.array_value.values.extend([AnyValue(string_value="a"), AnyValue(int_value=1)])
    span.attributes.append(array)
    (row,) = decode_otlp(_export(span))
    assert row["SpanAttributes"]["flag"] == "true"
    assert row["SpanAttributes"]["ratio"] == "0.5"
    assert row["SpanAttributes"]["raw"] == "abc"
    assert json.loads(row["SpanAttributes"]["list"]) == ["a", "1"]


# ---------------------------------------------------------------- helpers


def test_encode_otlp_response_matches_request_encoding():
    assert encode_otlp_response("application/json") == (b"{}", "application/json")
    assert encode_otlp_response("application/x-protobuf") == (b"", "application/x-protobuf")
    assert encode_otlp_response(None) == (b"", "application/x-protobuf")
