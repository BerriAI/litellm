"""The MCP ``mcp-session-id`` is captured for tool-call logging so the otel span
can carry ``mcp.session.id``. Guards the header read against casing and absence."""

from typing import Final
from unittest.mock import AsyncMock

import pytest

from litellm.proxy.mcp_server import operations
from litellm.proxy.mcp_server import server as mcp_server
from litellm.proxy.mcp_server.server import _mcp_session_id_from_headers


def test_reads_session_id_case_insensitively():
    # Clients send varied casing (``Mcp-Session-Id``, ``mcp-session-id``); all resolve.
    assert _mcp_session_id_from_headers({"mcp-session-id": "s1"}) == "s1"
    assert _mcp_session_id_from_headers({"Mcp-Session-Id": "s2"}) == "s2"
    assert _mcp_session_id_from_headers({"MCP-SESSION-ID": "s3"}) == "s3"


def test_stateless_call_has_no_session_id():
    # No header (stateless request) and an empty value both yield None, not "".
    assert _mcp_session_id_from_headers({"authorization": "Bearer x"}) is None
    assert _mcp_session_id_from_headers({"mcp-session-id": ""}) is None
    assert _mcp_session_id_from_headers(None) is None
    assert _mcp_session_id_from_headers({}) is None


@pytest.mark.asyncio
async def test_list_mcp_tools_threads_caller_session_without_inventing_from_trace_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: Final[list[dict[str, object]]] = []

    def fake_function_setup(*_args: object, **kwargs: object) -> tuple[None, None]:
        captured.append(dict(kwargs))
        return None, None

    monkeypatch.setattr(operations, "function_setup", fake_function_setup)
    monkeypatch.setattr(operations, "_get_allowed_mcp_servers", AsyncMock(return_value=[]))

    await operations._get_tools_from_mcp_servers(
        user_api_key_auth=None,
        mcp_auth_header=None,
        mcp_servers=None,
        raw_headers={"x-litellm-session-id": "sess-list-1", "langfuse_session_id": "lf-list-1"},
        log_list_tools_to_spendlogs=True,
    )
    await operations._get_tools_from_mcp_servers(
        user_api_key_auth=None,
        mcp_auth_header=None,
        mcp_servers=None,
        raw_headers={},
        litellm_trace_id="generated-trace-only",
        log_list_tools_to_spendlogs=True,
    )

    with_session, trace_only = captured
    assert with_session["litellm_session_id"] == "sess-list-1"
    with_meta: Final = with_session["metadata"]
    assert isinstance(with_meta, dict)
    assert with_meta["session_id"] == "sess-list-1"
    assert with_meta["trace_id"] == "sess-list-1"
    assert isinstance(with_meta["headers"], dict)
    assert with_meta["headers"]["langfuse_session_id"] == "lf-list-1"

    assert "litellm_session_id" not in trace_only
    trace_only_meta: Final = trace_only["metadata"]
    assert isinstance(trace_only_meta, dict)
    assert "session_id" not in trace_only_meta


@pytest.mark.asyncio
async def test_keyless_mcp_tool_call_threads_sanitized_headers_and_caller_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: Final[dict[str, object]] = {}

    async def fake_call_mcp_tool(**kwargs: object) -> object:
        captured.update(kwargs)
        return mcp_server.CallToolResult(content=[mcp_server.TextContent(type="text", text="ok")])

    monkeypatch.setattr(operations, "_dispatch_virtual_mcp_tool", AsyncMock(return_value=None))
    monkeypatch.setattr(operations, "call_mcp_tool", fake_call_mcp_tool)

    context: Final = operations.OperationContext(
        _caller=None,
        raw_headers={
            "x-litellm-session-id": "keyless-sess-1",
            "langfuse_session_id": "lf-keyless-1",
            "x-litellm-api-key": "sk-secret",
        },
    )
    await operations._execute_mcp_server_tool_call(
        context,
        mcp_server.CallToolRequestParams(name="echo", arguments={"text": "hi"}),
    )

    assert captured["litellm_session_id"] == "keyless-sess-1"
    metadata: Final = captured["metadata"]
    assert isinstance(metadata, dict)
    assert metadata["session_id"] == "keyless-sess-1"
    assert metadata["trace_id"] == "keyless-sess-1"
    headers: Final = metadata["headers"]
    assert isinstance(headers, dict)
    assert headers["langfuse_session_id"] == "lf-keyless-1"
    assert "x-litellm-api-key" not in headers


def test_publish_transport_span_anchors_and_stamps_keyless_mcp_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("opentelemetry")
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    from opentelemetry.trace import SpanKind

    from litellm.integrations.otel.logger import build_otel_v2_logger
    from litellm.integrations.otel.model.config import OpenTelemetryV2Config, is_otel_v2_enabled
    from litellm.integrations.otel.plumbing import context as otel_context
    from litellm.integrations.otel.plumbing import providers
    from litellm.proxy import proxy_server

    monkeypatch.setenv("LITELLM_OTEL_V2", "true")
    is_otel_v2_enabled.cache_clear()
    cfg: Final = OpenTelemetryV2Config(exporter="in_memory", mapper_names=["genai", "langfuse"])
    exporter: Final = InMemorySpanExporter()
    provider: Final = providers.build_tracer_provider(cfg, exporter=exporter)
    logger: Final = build_otel_v2_logger(config=cfg, tracer_provider=provider)
    monkeypatch.setattr(proxy_server, "open_telemetry_logger", logger)
    otel_context._request_root_span.set(None)

    scope: Final[dict[str, object]] = {"type": "http", "method": "POST", "path": "/echo/mcp", "headers": []}
    try:
        with provider.get_tracer("server").start_as_current_span(
            "POST /{mcp_server_name}/mcp", kind=SpanKind.SERVER
        ) as server_span:
            mcp_server._otel_publish_transport_span_on_scope(
                scope,  # pyright: ignore[reportArgumentType]  # minimal test ASGI scope
                {"x-litellm-session-id": "keyless-init-1"},
            )
            assert scope[mcp_server._MCP_TRANSPORT_SPAN_SCOPE_KEY] is server_span
    finally:
        otel_context._request_root_span.set(None)
        is_otel_v2_enabled.cache_clear()

    (finished,) = exporter.get_finished_spans()
    assert dict(finished.attributes or {})["session.id"] == "keyless-init-1"
