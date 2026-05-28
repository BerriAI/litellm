"""Regression coverage for LIT-3201: OpenTelemetry spans for MCP operations.

Exercises ``litellm.proxy._experimental.mcp_server.mcp_tracing.mcp_otel_span``
against an in-process OTEL ``InMemorySpanExporter`` so each test asserts on
the real exporter output rather than on a mock. Spans are wired in via a
``SimpleSpanProcessor`` attached to a stand-alone ``TracerProvider``, and
the proxy's ``open_telemetry_logger`` global is monkey-patched to expose
that provider's tracer.
"""

from __future__ import annotations

import asyncio
import os
import sys

import pytest

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../..")),
)

from opentelemetry import trace as otel_trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.trace import StatusCode

from litellm.proxy import proxy_server
from litellm.proxy._experimental.mcp_server.mcp_tracing import (
    MCP_OTEL_TRACER_NAME,
    mcp_otel_span,
)


class _FakeOtelLogger:
    """Just enough surface area for ``mcp_otel_span`` (only ``tracer``)."""

    def __init__(self, tracer):
        self.tracer = tracer


@pytest.fixture
def exporter(monkeypatch):
    """Wire InMemorySpanExporter into a fresh TracerProvider and expose its
    tracer via the proxy ``open_telemetry_logger`` global."""
    exp = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exp))
    tracer = provider.get_tracer(MCP_OTEL_TRACER_NAME)
    monkeypatch.setattr(
        proxy_server, "open_telemetry_logger", _FakeOtelLogger(tracer), raising=False
    )
    yield exp


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _finished(exporter):
    return list(exporter.get_finished_spans())


# -- success matrix --


@pytest.mark.parametrize(
    "operation, kwargs, expected_name, expected_attrs",
    [
        (
            "tool.call",
            {
                "server_name": "github",
                "tool_name": "create_issue",
                "arguments": {"title": "x", "body": "y"},
            },
            "mcp.tool.call",
            {
                "mcp.operation": "tool.call",
                "mcp.server.name": "github",
                "mcp.tool.name": "create_issue",
                "mcp.arguments_count": 2,
            },
        ),
        (
            "get_prompt",
            {
                "server_name": "tools",
                "prompt_name": "summarize",
                "arguments": {"x": 1},
            },
            "mcp.get_prompt",
            {
                "mcp.operation": "get_prompt",
                "mcp.server.name": "tools",
                "mcp.prompt.name": "summarize",
                "mcp.arguments_count": 1,
            },
        ),
        (
            "read_resource",
            {"resource_uri": "file:///etc/hostname"},
            "mcp.read_resource",
            {
                "mcp.operation": "read_resource",
                "mcp.resource.uri": "file:///etc/hostname",
            },
        ),
        (
            "list_tools",
            {"extra_attributes": {"mcp.servers_filter": ""}},
            "mcp.list_tools",
            {"mcp.operation": "list_tools", "mcp.servers_filter": ""},
        ),
        (
            "list_prompts",
            {"extra_attributes": {"mcp.servers_filter": "github,slack"}},
            "mcp.list_prompts",
            {"mcp.operation": "list_prompts", "mcp.servers_filter": "github,slack"},
        ),
        (
            "list_resources",
            {"extra_attributes": {"mcp.servers_filter": ""}},
            "mcp.list_resources",
            {"mcp.operation": "list_resources"},
        ),
        (
            "list_resource_templates",
            {"extra_attributes": {"mcp.servers_filter": ""}},
            "mcp.list_resource_templates",
            {"mcp.operation": "list_resource_templates"},
        ),
    ],
    ids=[
        "tool.call",
        "get_prompt",
        "read_resource",
        "list_tools",
        "list_prompts_with_filter",
        "list_resources",
        "list_resource_templates",
    ],
)
def test_mcp_span_success_path_records_attributes(
    exporter, operation, kwargs, expected_name, expected_attrs
):
    async def run():
        async with mcp_otel_span(operation, **kwargs) as span:
            assert span is not None, "span must be live when OTEL is configured"

    _run(run())
    spans = _finished(exporter)
    assert len(spans) == 1
    span = spans[0]
    assert span.name == expected_name
    for k, v in expected_attrs.items():
        assert span.attributes[k] == v, f"{k}: {span.attributes.get(k)!r} != {v!r}"
    assert span.status.status_code == StatusCode.OK
    assert not span.events


# -- failure path --


def test_mcp_span_failure_records_exception_and_reraises(exporter):
    class _Boom(RuntimeError):
        pass

    async def run():
        async with mcp_otel_span("tool.call", tool_name="t", server_name="s"):
            raise _Boom("kaboom")

    with pytest.raises(_Boom):
        _run(run())

    spans = _finished(exporter)
    assert len(spans) == 1
    span = spans[0]
    assert span.status.status_code == StatusCode.ERROR
    assert span.attributes["mcp.error.type"] == "_Boom"
    assert "exception" in [e.name for e in span.events]


# -- parent span inheritance --


def test_mcp_span_inherits_parent_when_one_is_active(exporter):
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    otel_trace.set_tracer_provider(provider)
    parent_tracer = provider.get_tracer("test.parent")
    proxy_server.open_telemetry_logger = _FakeOtelLogger(
        provider.get_tracer(MCP_OTEL_TRACER_NAME)
    )

    async def run():
        with parent_tracer.start_as_current_span("parent.op") as parent:
            parent_ctx = parent.get_span_context()
            async with mcp_otel_span("tool.call", tool_name="t", server_name="s"):
                pass
        return parent_ctx

    parent_ctx = _run(run())

    finished = _finished(exporter)
    mcp_span = next(s for s in finished if s.name == "mcp.tool.call")
    parent_span = next(s for s in finished if s.name == "parent.op")
    assert mcp_span.context.trace_id == parent_ctx.trace_id
    assert mcp_span.parent is not None
    assert mcp_span.parent.span_id == parent_span.context.span_id


# -- no-op paths --


def test_mcp_span_no_op_when_otel_logger_missing(monkeypatch):
    exp = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exp))
    monkeypatch.setattr(proxy_server, "open_telemetry_logger", None, raising=False)

    async def run():
        async with mcp_otel_span("tool.call", tool_name="x") as s:
            assert s is None

    _run(run())
    assert _finished(exp) == []


def test_mcp_span_no_op_when_logger_has_no_tracer(monkeypatch):
    exp = InMemorySpanExporter()

    class _NoTracer:
        pass

    monkeypatch.setattr(
        proxy_server, "open_telemetry_logger", _NoTracer(), raising=False
    )

    async def run():
        async with mcp_otel_span("list_tools") as s:
            assert s is None

    _run(run())
    assert _finished(exp) == []


# -- secrets safety --


def test_mcp_span_records_arguments_count_not_values(exporter):
    async def run():
        async with mcp_otel_span(
            "tool.call",
            tool_name="t",
            server_name="s",
            arguments={"secret": "s3kret-token", "x": 1},
        ):
            pass

    _run(run())
    spans = _finished(exporter)
    assert len(spans) == 1
    span = spans[0]
    assert span.attributes["mcp.arguments_count"] == 2
    for k, v in span.attributes.items():
        assert "s3kret-token" not in str(v), (k, v)
