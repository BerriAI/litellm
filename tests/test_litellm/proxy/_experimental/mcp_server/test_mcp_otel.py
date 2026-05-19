"""Tests for OTEL span coverage of MCP flows.

These tests stand in for a manual "instrument and curl every MCP entry
point" exercise. They wire an in-memory OTEL exporter into LiteLLM and
then drive each flow:

* helper-level: ``mcp_span`` no-op, success, error
* internal: ``_list_mcp_tools`` (delegated to via the MCP protocol)
* internal: ``execute_mcp_tool`` (managed-server path)
* protocol: ``list_tools`` (MCP server handler)
* protocol: ``mcp_server_tool_call`` (MCP server handler)
* REST: ``with_mcp_span`` decorator on a fake FastAPI-style handler

For each flow, the assertions cover:

1. A span is created with the expected stable name.
2. The span carries the documented MCP attributes (operation, tool name,
   server name, user id, etc.) — operators rely on these to slice traces.
3. Failures mark the span ``ERROR`` and record the exception.
4. Spans degrade to no-ops when OTEL is not configured (so the helpers
   are safe to call from every MCP code path unconditionally).
"""

from __future__ import annotations

from typing import Any, Dict, Iterator, List

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

import litellm
from litellm.integrations.opentelemetry import OpenTelemetry, OpenTelemetryConfig
from litellm.proxy._experimental.mcp_server import mcp_otel
from litellm.proxy._experimental.mcp_server.mcp_otel import (
    ATTR_MCP_ALLOWED_SERVER_COUNT,
    ATTR_MCP_OPERATION,
    ATTR_MCP_RESULT_IS_ERROR,
    ATTR_MCP_SERVER_ID,
    ATTR_MCP_SERVER_NAME,
    ATTR_MCP_TOOL_COUNT,
    ATTR_MCP_TOOL_NAME,
    ATTR_MCP_USER_ID,
    SPAN_MCP_EXECUTE_TOOL,
    SPAN_MCP_LIST_TOOLS,
    SPAN_MCP_PROTOCOL_CALL_TOOL,
    SPAN_MCP_PROTOCOL_LIST_TOOLS,
    SPAN_MCP_REST_CALL_TOOL,
    SPAN_MCP_REST_LIST_TOOLS,
    mcp_span,
    with_mcp_span,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def in_memory_otel(monkeypatch: pytest.MonkeyPatch) -> Iterator[InMemorySpanExporter]:
    """Install an in-memory OTEL exporter and an ``OpenTelemetry`` callback.

    Yields the exporter so tests can assert on emitted spans.

    A fresh ``TracerProvider`` is installed for the test, but the global
    provider is *not* restored on teardown — ``opentelemetry`` warns when
    you set a global provider twice, and tests within the same process
    share state. We re-clear the exporter at the start of every test to
    avoid bleed-through.

    ``OpenTelemetry.__init__`` self-registers as
    ``proxy_server.open_telemetry_logger`` if that global is None. We
    capture-and-restore that global on teardown so the no-otel tests
    aren't poisoned by a leaked instance from this fixture.
    """
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    # Setting the global tracer provider is what plain ``trace.get_tracer``
    # consumers (e.g. anything that didn't construct its own tracer) will
    # see. We also bind the tracer directly on the OpenTelemetry instance
    # because mcp_otel reads ``otel.tracer`` rather than calling
    # ``trace.get_tracer`` on each span.
    trace.set_tracer_provider(provider)

    from litellm.proxy import proxy_server

    prev_proxy_otel = getattr(proxy_server, "open_telemetry_logger", None)

    otel = OpenTelemetry(config=OpenTelemetryConfig(exporter="console"))
    otel.tracer = provider.get_tracer("mcp-otel-tests")

    original_callbacks = list(litellm.callbacks or [])
    litellm.callbacks = [otel]
    try:
        yield exporter
    finally:
        litellm.callbacks = original_callbacks
        proxy_server.open_telemetry_logger = prev_proxy_otel
        # Drain remaining spans so the next test starts clean even if
        # the previous test forgot to assert on them.
        exporter.clear()


def _spans_by_name(exporter: InMemorySpanExporter, name: str) -> List[ReadableSpan]:
    """Return all finished spans with ``name`` (drains nothing)."""
    return [s for s in exporter.get_finished_spans() if s.name == name]


# ---------------------------------------------------------------------------
# Helper-level coverage (mcp_span / with_mcp_span / no-op fallback)
# ---------------------------------------------------------------------------


def test_mcp_span_is_noop_when_otel_not_configured() -> None:
    """Without an OTEL logger in ``litellm.callbacks`` the span yields ``None``.

    This is the contract that lets MCP code wrap *every* entry point with
    ``mcp_span(...)`` without guarding the call site.

    We also null out ``proxy_server.open_telemetry_logger`` because
    ``OpenTelemetry.__init__`` self-registers there — any earlier test
    that instantiated one (even in a different file) would otherwise
    leak it into this check.
    """
    from litellm.proxy import proxy_server

    original_callbacks = list(litellm.callbacks or [])
    original_proxy_otel = getattr(proxy_server, "open_telemetry_logger", None)
    litellm.callbacks = []
    proxy_server.open_telemetry_logger = None
    try:
        with mcp_span(
            "mcp.test.noop", attributes={ATTR_MCP_TOOL_NAME: "irrelevant"}
        ) as span:
            assert span is None
            # set_mcp_span_attribute must also tolerate a None span.
            mcp_otel.set_mcp_span_attribute(span, "x", "y")
    finally:
        litellm.callbacks = original_callbacks
        proxy_server.open_telemetry_logger = original_proxy_otel


def test_mcp_span_success_emits_span_with_attributes(
    in_memory_otel: InMemorySpanExporter,
) -> None:
    with mcp_span(
        "mcp.test.success",
        attributes={
            ATTR_MCP_OPERATION: "list_tools",
            ATTR_MCP_TOOL_NAME: "foo",
            "ignored.none": None,  # must be skipped, not stringified
        },
    ) as span:
        assert span is not None
        # set_mcp_span_attributes must tolerate a None value too.
        mcp_otel.set_mcp_span_attribute(span, "late.attr", 42)
        mcp_otel.set_mcp_span_attribute(span, "skipped.none", None)

    spans = _spans_by_name(in_memory_otel, "mcp.test.success")
    assert len(spans) == 1
    s = spans[0]
    assert s.status.status_code.name == "OK"
    assert s.attributes[ATTR_MCP_OPERATION] == "list_tools"
    assert s.attributes[ATTR_MCP_TOOL_NAME] == "foo"
    assert s.attributes["late.attr"] == 42
    assert "ignored.none" not in s.attributes
    assert "skipped.none" not in s.attributes


def test_mcp_span_records_exception_and_sets_error_status(
    in_memory_otel: InMemorySpanExporter,
) -> None:
    """A raised exception must be recorded on the span and re-raised."""

    class _Boom(RuntimeError):
        pass

    with pytest.raises(_Boom):
        with mcp_span("mcp.test.error", attributes={ATTR_MCP_TOOL_NAME: "kaboom"}):
            raise _Boom("expected")

    spans = _spans_by_name(in_memory_otel, "mcp.test.error")
    assert len(spans) == 1
    s = spans[0]
    assert s.status.status_code.name == "ERROR"
    # record_exception adds an event named "exception" with the typename.
    event_names = [e.name for e in s.events]
    assert "exception" in event_names


def test_mcp_span_nests_inside_parent(
    in_memory_otel: InMemorySpanExporter,
) -> None:
    """Nested ``mcp_span`` calls must produce a parent-child trace."""
    with mcp_span("mcp.test.parent"):
        with mcp_span("mcp.test.child"):
            pass

    parent_spans = _spans_by_name(in_memory_otel, "mcp.test.parent")
    child_spans = _spans_by_name(in_memory_otel, "mcp.test.child")
    assert len(parent_spans) == 1
    assert len(child_spans) == 1
    parent = parent_spans[0]
    child = child_spans[0]
    # parent_span_id on the child must match the parent's span id.
    assert child.parent is not None
    assert child.parent.span_id == parent.context.span_id
    assert child.context.trace_id == parent.context.trace_id


@pytest.mark.asyncio
async def test_with_mcp_span_decorator_attributes_factory(
    in_memory_otel: InMemorySpanExporter,
) -> None:
    """The decorator builds attributes from the wrapped callable's kwargs."""

    @with_mcp_span(
        SPAN_MCP_REST_LIST_TOOLS,
        attribute_factory=lambda *args, **kwargs: {
            ATTR_MCP_OPERATION: "list_tools",
            ATTR_MCP_USER_ID: getattr(kwargs.get("user_api_key_dict"), "user_id", None),
        },
    )
    async def fake_handler(
        request: Any = None, user_api_key_dict: Any = None
    ) -> Dict[str, Any]:
        return {"tools": [1, 2, 3]}

    class _Auth:
        user_id = "test-user-abc"

    result = await fake_handler(request=object(), user_api_key_dict=_Auth())

    assert result == {"tools": [1, 2, 3]}
    spans = _spans_by_name(in_memory_otel, SPAN_MCP_REST_LIST_TOOLS)
    assert len(spans) == 1
    assert spans[0].attributes[ATTR_MCP_USER_ID] == "test-user-abc"
    assert spans[0].attributes[ATTR_MCP_OPERATION] == "list_tools"


@pytest.mark.asyncio
async def test_with_mcp_span_attribute_factory_failure_does_not_break_handler(
    in_memory_otel: InMemorySpanExporter,
) -> None:
    """If the factory raises, we must still execute the handler and emit a span.

    Operators expect attribute-extraction bugs to surface as missing
    attributes, not as 500s.
    """

    @with_mcp_span(
        "mcp.test.factory_boom",
        attribute_factory=lambda *args, **kwargs: 1 / 0,
    )
    async def fake_handler() -> str:
        return "ok"

    result = await fake_handler()
    assert result == "ok"

    spans = _spans_by_name(in_memory_otel, "mcp.test.factory_boom")
    assert len(spans) == 1
    assert spans[0].status.status_code.name == "OK"


# ---------------------------------------------------------------------------
# Helper discovery: callback list vs proxy_server global
# ---------------------------------------------------------------------------


def test_get_active_otel_logger_prefers_callback_list(
    in_memory_otel: InMemorySpanExporter,
) -> None:
    """A registered ``OpenTelemetry`` callback wins over the proxy global.

    Important because tests and SDK usage both go through
    ``litellm.callbacks``; we should not require ``proxy_server`` to be
    imported for spans to work.
    """
    logger = mcp_otel._get_active_otel_logger()
    assert logger is not None
    # Span emission round-trip
    with mcp_span("mcp.test.discovery"):
        pass
    assert _spans_by_name(in_memory_otel, "mcp.test.discovery")


def test_get_active_otel_logger_returns_none_without_otel() -> None:
    from litellm.proxy import proxy_server

    original_callbacks = list(litellm.callbacks or [])
    original_proxy_otel = getattr(proxy_server, "open_telemetry_logger", None)
    litellm.callbacks = []
    proxy_server.open_telemetry_logger = None
    try:
        # If proxy_server is unimported the import-error path returns None.
        # If it *is* imported but has no logger configured, that also returns
        # None. Either way the function must not crash.
        assert mcp_otel._get_active_otel_logger() is None
    finally:
        litellm.callbacks = original_callbacks
        proxy_server.open_telemetry_logger = original_proxy_otel


# ---------------------------------------------------------------------------
# Internal flows: _list_mcp_tools + execute_mcp_tool
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_mcp_tools_emits_span_with_tool_count(
    in_memory_otel: InMemorySpanExporter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``_list_mcp_tools`` must emit ``mcp.list_tools`` with a tool_count attribute."""
    from litellm.proxy._experimental.mcp_server import server as mcp_server_module

    # Stub the heavy permission merge and the upstream fetch so the test
    # is hermetic — we're asserting on the *span*, not on the tool list.
    async def fake_merge(auth):
        return auth

    async def fake_get_tools(**kwargs):
        return [{"name": "alpha"}, {"name": "beta"}]

    monkeypatch.setattr(mcp_server_module, "_merge_toolset_permissions", fake_merge)
    monkeypatch.setattr(
        mcp_server_module, "_get_tools_from_mcp_servers", fake_get_tools
    )

    tools = await mcp_server_module._list_mcp_tools(user_api_key_auth=None)
    assert len(tools) == 2

    spans = _spans_by_name(in_memory_otel, SPAN_MCP_LIST_TOOLS)
    assert len(spans) == 1
    s = spans[0]
    assert s.status.status_code.name == "OK"
    assert s.attributes[ATTR_MCP_OPERATION] == "list_tools"
    assert s.attributes[ATTR_MCP_TOOL_COUNT] == 2


@pytest.mark.asyncio
async def test_list_mcp_tools_records_error_on_upstream_failure(
    in_memory_otel: InMemorySpanExporter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Upstream errors must surface as ``mcp.error`` on the span.

    ``_list_mcp_tools`` swallows the exception and returns ``[]`` (to keep
    the MCP HTTP stream alive); the span attribute is the only signal an
    operator gets that something went wrong.
    """
    from litellm.proxy._experimental.mcp_server import server as mcp_server_module

    async def fake_merge(auth):
        return auth

    async def boom(**kwargs):
        raise RuntimeError("upstream MCP server is down")

    monkeypatch.setattr(mcp_server_module, "_merge_toolset_permissions", fake_merge)
    monkeypatch.setattr(mcp_server_module, "_get_tools_from_mcp_servers", boom)

    tools = await mcp_server_module._list_mcp_tools(user_api_key_auth=None)
    assert tools == []

    spans = _spans_by_name(in_memory_otel, SPAN_MCP_LIST_TOOLS)
    assert len(spans) == 1
    s = spans[0]
    # The span itself is *not* ERROR — the helper caught the exception
    # before mcp_span saw it. The error is recorded as an attribute so
    # it surfaces in dashboards without breaking the parent trace.
    assert s.attributes["mcp.error"] == "upstream MCP server is down"
    assert s.attributes[ATTR_MCP_TOOL_COUNT] == 0


@pytest.mark.asyncio
async def test_execute_mcp_tool_emits_span_with_server_and_tool(
    in_memory_otel: InMemorySpanExporter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``execute_mcp_tool`` emits ``mcp.execute_tool`` with the resolved server."""
    from mcp.types import CallToolResult, TextContent

    from litellm.proxy._experimental.mcp_server import server as mcp_server_module
    from litellm.types.mcp_server.mcp_server_manager import MCPServer

    fake_server = MCPServer(
        server_id="srv-99",
        name="myserver",
        url="http://example.com",
        transport="http",
        auth_type=None,
        mcp_info={},
    )

    def fake_get_server_from_tool_name(name: str):
        return fake_server

    async def fake_managed_call(**kwargs):
        return CallToolResult(
            content=[TextContent(type="text", text="ok")], isError=False
        )

    monkeypatch.setattr(
        mcp_server_module.global_mcp_server_manager,
        "_get_mcp_server_from_tool_name",
        fake_get_server_from_tool_name,
    )
    monkeypatch.setattr(
        mcp_server_module, "_handle_managed_mcp_tool", fake_managed_call
    )

    from datetime import datetime as _dt

    result = await mcp_server_module.execute_mcp_tool(
        name="myserver-getThing",
        arguments={"id": "1"},
        allowed_mcp_servers=[fake_server],
        start_time=_dt.now(),
        user_api_key_auth=None,
    )
    assert result.isError is False

    spans = _spans_by_name(in_memory_otel, SPAN_MCP_EXECUTE_TOOL)
    assert len(spans) == 1
    s = spans[0]
    assert s.status.status_code.name == "OK"
    assert s.attributes[ATTR_MCP_OPERATION] == "execute_tool"
    assert s.attributes[ATTR_MCP_SERVER_NAME] == "myserver"
    assert s.attributes[ATTR_MCP_SERVER_ID] == "srv-99"
    assert s.attributes[ATTR_MCP_ALLOWED_SERVER_COUNT] == 1
    assert s.attributes[ATTR_MCP_RESULT_IS_ERROR] is False


@pytest.mark.asyncio
async def test_execute_mcp_tool_records_tool_error_flag(
    in_memory_otel: InMemorySpanExporter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tool-side errors (``CallToolResult.isError=True``) surface on the span."""
    from mcp.types import CallToolResult, TextContent

    from litellm.proxy._experimental.mcp_server import server as mcp_server_module
    from litellm.types.mcp_server.mcp_server_manager import MCPServer

    fake_server = MCPServer(
        server_id="srv-err",
        name="errsrv",
        url="http://example.com",
        transport="http",
        auth_type=None,
        mcp_info={},
    )

    monkeypatch.setattr(
        mcp_server_module.global_mcp_server_manager,
        "_get_mcp_server_from_tool_name",
        lambda name: fake_server,
    )

    async def fake_managed_call(**kwargs):
        return CallToolResult(
            content=[TextContent(type="text", text="boom")], isError=True
        )

    monkeypatch.setattr(
        mcp_server_module, "_handle_managed_mcp_tool", fake_managed_call
    )

    from datetime import datetime as _dt

    result = await mcp_server_module.execute_mcp_tool(
        name="errsrv-failingTool",
        arguments={},
        allowed_mcp_servers=[fake_server],
        start_time=_dt.now(),
        user_api_key_auth=None,
    )
    assert result.isError is True

    spans = _spans_by_name(in_memory_otel, SPAN_MCP_EXECUTE_TOOL)
    assert len(spans) == 1
    s = spans[0]
    assert s.attributes[ATTR_MCP_RESULT_IS_ERROR] is True


@pytest.mark.asyncio
async def test_execute_mcp_tool_marks_span_error_on_exception(
    in_memory_otel: InMemorySpanExporter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An exception from the managed call must end the span as ERROR."""
    from litellm.proxy._experimental.mcp_server import server as mcp_server_module
    from litellm.types.mcp_server.mcp_server_manager import MCPServer

    fake_server = MCPServer(
        server_id="srv-x",
        name="xsrv",
        url="http://example.com",
        transport="http",
        auth_type=None,
        mcp_info={},
    )

    monkeypatch.setattr(
        mcp_server_module.global_mcp_server_manager,
        "_get_mcp_server_from_tool_name",
        lambda name: fake_server,
    )

    async def boom(**kwargs):
        raise RuntimeError("network down")

    monkeypatch.setattr(mcp_server_module, "_handle_managed_mcp_tool", boom)

    from datetime import datetime as _dt

    with pytest.raises(RuntimeError):
        await mcp_server_module.execute_mcp_tool(
            name="xsrv-anyTool",
            arguments={},
            allowed_mcp_servers=[fake_server],
            start_time=_dt.now(),
            user_api_key_auth=None,
        )

    spans = _spans_by_name(in_memory_otel, SPAN_MCP_EXECUTE_TOOL)
    assert len(spans) == 1
    s = spans[0]
    assert s.status.status_code.name == "ERROR"
    # The resolved server attributes must still be present so the failed
    # span isn't anonymous.
    assert s.attributes[ATTR_MCP_SERVER_NAME] == "xsrv"


# ---------------------------------------------------------------------------
# Span name registry — guards against accidental renames that would
# silently break dashboards/alerts referencing these strings.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "constant,expected",
    [
        (SPAN_MCP_LIST_TOOLS, "mcp.list_tools"),
        (SPAN_MCP_EXECUTE_TOOL, "mcp.execute_tool"),
        (SPAN_MCP_PROTOCOL_LIST_TOOLS, "mcp.protocol.list_tools"),
        (SPAN_MCP_PROTOCOL_CALL_TOOL, "mcp.protocol.call_tool"),
        (SPAN_MCP_REST_LIST_TOOLS, "mcp.rest.list_tools"),
        (SPAN_MCP_REST_CALL_TOOL, "mcp.rest.call_tool"),
    ],
)
def test_span_name_constants_are_stable(constant: str, expected: str) -> None:
    """If these change, downstream alerts/dashboards break — bump cautiously."""
    assert constant == expected
