import sys
import types
import inspect
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException
import importlib

from litellm.responses.mcp.litellm_proxy_mcp_handler import (
    LiteLLM_Proxy_MCP_Handler,
)
from litellm.responses.utils import INVALID_MCP_CLIENT_IP_SENTINEL
from typing import Any, cast
from litellm.types.llms.openai import ResponseAPIUsage, ResponsesAPIResponse
from litellm.types.utils import ModelResponse
from litellm.types.responses.main import OutputFunctionToolCall


class _DummyMCPResult:
    def __init__(self):
        self.content = []


def _setup_mcp_call_environment(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    """Patch MCP globals so _execute_tool_calls can run in tests."""
    proxy_module = types.SimpleNamespace(proxy_logging_obj=object())
    monkeypatch.setitem(sys.modules, "litellm.proxy.proxy_server", proxy_module)

    fake_manager = types.SimpleNamespace(
        call_tool=AsyncMock(return_value=_DummyMCPResult()),
        # Newer logging path calls this to enrich spend logs metadata
        _get_mcp_server_from_tool_name=MagicMock(return_value=None),
    )
    monkeypatch.setattr(
        "litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager",
        fake_manager,
    )
    return fake_manager.call_tool


def _setup_proxy_logging(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    """Patch proxy_logging_obj so failure hook can be asserted."""
    proxy_logging_obj = MagicMock()
    proxy_logging_obj.post_call_failure_hook = AsyncMock()
    proxy_module = types.SimpleNamespace(proxy_logging_obj=proxy_logging_obj)
    monkeypatch.setitem(sys.modules, "litellm.proxy.proxy_server", proxy_module)
    return proxy_logging_obj.post_call_failure_hook


def test_deduplicate_mcp_tools_single_allowed_server():
    tools = [{"name": "search"}, {"name": "search"}]  # duplicate on purpose

    deduped, server_map = LiteLLM_Proxy_MCP_Handler._deduplicate_mcp_tools(
        tools,
        ["everything"],
    )

    assert len(deduped) == 1
    assert server_map == {"search": "everything"}


@pytest.mark.parametrize(
    "tool_name,expected_server",
    [
        ("alpha-tool", "alpha"),
        ("beta-another_tool", "beta"),
    ],
)
def test_deduplicate_mcp_tools_prefixed_names(tool_name, expected_server):
    tools = [{"name": tool_name}]

    _, server_map = LiteLLM_Proxy_MCP_Handler._deduplicate_mcp_tools(
        tools,
        ["alpha", "beta"],
    )

    assert server_map[tool_name] == expected_server


def test_extract_tool_calls_from_chat_response_handles_tool_calls():
    response = ModelResponse(
        id="resp-1",
        choices=[
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call-123",
                            "type": "function",
                            "function": {"name": "foo", "arguments": "{}"},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        model="gpt",
        created=0,
        object="chat.completion",
    )

    tool_calls = LiteLLM_Proxy_MCP_Handler._extract_tool_calls_from_chat_response(
        response
    )

    assert len(tool_calls) == 1
    assert tool_calls[0]["function"]["name"] == "foo"


def test_create_follow_up_messages_for_chat_appends_tool_results():
    original_messages = [{"role": "user", "content": "hi"}]
    response = ModelResponse(
        id="resp-2",
        choices=[
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call-abc",
                            "type": "function",
                            "function": {"name": "foo", "arguments": "{}"},
                        }
                    ],
                },
            }
        ],
        model="gpt",
        created=0,
        object="chat.completion",
    )
    tool_results = [
        {
            "tool_call_id": "call-abc",
            "name": "foo",
            "result": "done",
        }
    ]

    follow_up = LiteLLM_Proxy_MCP_Handler._create_follow_up_messages_for_chat(
        original_messages,
        response,
        tool_results,
    )

    assert follow_up[0]["role"] == "user"
    assert follow_up[-1]["role"] == "tool"
    assert follow_up[-1]["name"] == "foo"
    assert follow_up[-1]["content"] == "done"


def test_transform_mcp_tools_to_openai_uses_chat_format(monkeypatch):
    captured = {}

    def fake_transform_chat(tool):
        captured.setdefault("chat", []).append(tool)
        return {"chat": True}

    def fake_transform_responses(tool):
        captured.setdefault("responses", []).append(tool)
        return {"responses": True}

    monkeypatch.setattr(
        "litellm.experimental_mcp_client.tools.transform_mcp_tool_to_openai_tool",
        fake_transform_chat,
    )
    monkeypatch.setattr(
        "litellm.experimental_mcp_client.tools.transform_mcp_tool_to_openai_responses_api_tool",
        fake_transform_responses,
    )

    chat_tools = LiteLLM_Proxy_MCP_Handler._transform_mcp_tools_to_openai(
        ["tool"], target_format="chat"
    )
    resp_tools = LiteLLM_Proxy_MCP_Handler._transform_mcp_tools_to_openai(["tool"])

    assert chat_tools == [{"chat": True}]
    assert resp_tools == [{"responses": True}]
    assert captured["chat"] == ["tool"]
    assert captured["responses"] == ["tool"]


def test_create_follow_up_input_handles_response_function_tool_call():
    response = types.SimpleNamespace(
        output=[
            OutputFunctionToolCall(
                id="id",
                type="function_call",
                call_id="call-1",
                name="foo",
                arguments="{}",
                status="completed",
            )
        ]
    )

    follow_up = LiteLLM_Proxy_MCP_Handler._create_follow_up_input(
        response=cast(Any, response),
        tool_results=[],
        original_input=None,
    )

    assert follow_up == [
        {
            "type": "function_call",
            "call_id": "call-1",
            "name": "foo",
            "arguments": "{}",
        }
    ]


def test_parse_mcp_tools_recognizes_lazymcp_urls():
    tools, other_tools = LiteLLM_Proxy_MCP_Handler._parse_mcp_tools(
        [
            {"type": "mcp", "server_url": "https://host.example/lazymcp"},
            {"type": "mcp", "server_url": "https://host.example/lazymcp/github"},
            {"type": "mcp", "server_url": "https://host.example/mcp/github"},
        ]
    )

    assert other_tools == []
    assert [tool["server_url"] for tool in tools] == [
        "litellm_proxy/lazymcp",
        "litellm_proxy/lazymcp/github",
        "litellm_proxy/mcp/github",
    ]


def test_should_use_litellm_mcp_gateway_callable_as_static_method():
    assert LiteLLM_Proxy_MCP_Handler._should_use_litellm_mcp_gateway(
        [{"type": "mcp", "server_url": "litellm_proxy/lazymcp/github"}]
    )


def test_decode_lazymcp_tool_server_map_value_handles_invalid_payloads():
    assert LiteLLM_Proxy_MCP_Handler._decode_lazymcp_tool_server_map_value(None) is None
    assert (
        LiteLLM_Proxy_MCP_Handler._decode_lazymcp_tool_server_map_value("not-lazymcp")
        is None
    )
    assert LiteLLM_Proxy_MCP_Handler._decode_lazymcp_tool_server_map_value(
        "lazymcp:not-json"
    ) == {"mcp_servers": [], "toolset_id": None}
    assert LiteLLM_Proxy_MCP_Handler._decode_lazymcp_tool_server_map_value(
        "lazymcp:[]"
    ) == {"mcp_servers": [], "toolset_id": None}
    assert LiteLLM_Proxy_MCP_Handler._decode_lazymcp_tool_server_map_value(
        'lazymcp:{"mcp_servers":"github"}'
    ) == {"mcp_servers": []}
    assert LiteLLM_Proxy_MCP_Handler._decode_lazymcp_tool_server_map_value(
        LiteLLM_Proxy_MCP_Handler._encode_lazymcp_tool_server_map_value(None, "toolset")
    ) == {"mcp_servers": None, "toolset_id": "toolset"}


def test_should_use_litellm_mcp_gateway_matches_proxy_urls():
    assert LiteLLM_Proxy_MCP_Handler._should_use_litellm_mcp_gateway(
        [{"type": "mcp", "server_url": "https://proxy.example/mcp/github"}]
    )
    assert LiteLLM_Proxy_MCP_Handler._should_use_litellm_mcp_gateway(
        [{"type": "mcp", "server_url": "https://proxy.example/lazymcp/github"}]
    )
    assert not LiteLLM_Proxy_MCP_Handler._should_use_litellm_mcp_gateway(
        [{"type": "function", "server_url": "https://proxy.example/lazymcp/github"}]
    )


def test_get_requested_mcp_servers_handles_lazymcp_variants():
    servers, use_lazymcp = LiteLLM_Proxy_MCP_Handler._get_requested_mcp_servers(
        [
            {"type": "mcp", "server_url": "litellm_proxy/mcp/github"},
            {"type": "mcp", "server_url": "litellm_proxy/lazymcp/slack"},
            {"type": "mcp", "server_url": "litellm_proxy/lazymcp"},
        ]
    )

    assert servers == ["github", "slack"]
    assert use_lazymcp is True


@pytest.mark.asyncio
async def test_resolve_lazymcp_scope_handles_server_toolset_and_errors(monkeypatch):
    server_manager = types.SimpleNamespace(
        get_mcp_server_by_name=MagicMock(side_effect=[object(), None, None, None]),
        get_toolset_by_name_cached=AsyncMock(
            side_effect=[
                types.SimpleNamespace(toolset_id="toolset-1"),
                RuntimeError("db"),
            ]
        ),
    )
    proxy_module = types.SimpleNamespace(
        prisma_client=object(),
        _is_mcp_access_group_cached=AsyncMock(return_value=False),
    )
    monkeypatch.setitem(sys.modules, "litellm.proxy.proxy_server", proxy_module)

    assert await LiteLLM_Proxy_MCP_Handler._resolve_lazymcp_scope(
        ["github"], server_manager
    ) == (["github"], None)
    assert await LiteLLM_Proxy_MCP_Handler._resolve_lazymcp_scope(
        ["toolset"], server_manager
    ) == (None, "toolset-1")
    assert await LiteLLM_Proxy_MCP_Handler._resolve_lazymcp_scope(
        ["broken"], server_manager
    ) == ([], None)


@pytest.mark.asyncio
async def test_resolve_lazymcp_scope_keeps_access_group(monkeypatch):
    server_manager = types.SimpleNamespace(
        get_mcp_server_by_name=MagicMock(return_value=None),
        get_toolset_by_name_cached=AsyncMock(return_value=None),
    )
    proxy_module = types.SimpleNamespace(
        prisma_client=object(),
        _is_mcp_access_group_cached=AsyncMock(return_value=True),
    )
    monkeypatch.setitem(sys.modules, "litellm.proxy.proxy_server", proxy_module)

    assert await LiteLLM_Proxy_MCP_Handler._resolve_lazymcp_scope(
        ["dev_group"], server_manager
    ) == (["dev_group"], None)
    server_manager.get_toolset_by_name_cached.assert_not_awaited()


@pytest.mark.asyncio
async def test_lazymcp_catalog_uses_verified_client_ip(monkeypatch):
    captured = {}

    async def fake_get_lazymcp_catalog(**kwargs):
        captured.update(kwargs)
        return {"description": "ok"}

    monkeypatch.setattr(
        "litellm.proxy._experimental.mcp_server.server._get_lazymcp_catalog",
        fake_get_lazymcp_catalog,
    )

    await LiteLLM_Proxy_MCP_Handler._get_lazymcp_gateway_tools(
        user_api_key_auth=None,
        effective_filter=["github"],
        active_toolset_id=None,
        mcp_auth_header=None,
        mcp_server_auth_headers=None,
        client_ip="10.0.0.8",
    )

    assert captured["client_ip"] == "10.0.0.8"


@pytest.mark.asyncio
async def test_lazymcp_catalog_uses_fail_closed_client_ip(monkeypatch):
    from litellm.responses.utils import ResponsesAPIRequestUtils

    captured = {}

    async def fake_get_lazymcp_catalog(**kwargs):
        captured.update(kwargs)
        return {"description": "ok"}

    monkeypatch.setattr(
        "litellm.proxy._experimental.mcp_server.server._get_lazymcp_catalog",
        fake_get_lazymcp_catalog,
    )

    await LiteLLM_Proxy_MCP_Handler._get_lazymcp_gateway_tools(
        user_api_key_auth=None,
        effective_filter=None,
        active_toolset_id=None,
        mcp_auth_header=None,
        mcp_server_auth_headers=None,
        client_ip=ResponsesAPIRequestUtils.get_verified_mcp_client_ip(None),
    )

    assert captured["client_ip"] == INVALID_MCP_CLIENT_IP_SENTINEL


@pytest.mark.asyncio
async def test_lazymcp_catalog_rejects_unauthorized_toolset(monkeypatch):
    get_catalog_mock = AsyncMock(return_value={"description": "blocked"})
    apply_scope_mock = AsyncMock(
        side_effect=HTTPException(status_code=403, detail="forbidden")
    )
    monkeypatch.setattr(
        "litellm.proxy._experimental.mcp_server.server._get_lazymcp_catalog",
        get_catalog_mock,
    )
    monkeypatch.setattr(
        "litellm.proxy._experimental.mcp_server.server._apply_toolset_scope",
        apply_scope_mock,
    )

    with pytest.raises(HTTPException) as exc_info:
        await LiteLLM_Proxy_MCP_Handler._get_lazymcp_gateway_tools(
            user_api_key_auth=types.SimpleNamespace(api_key="sk-test"),
            effective_filter=None,
            active_toolset_id="toolset-blocked",
            mcp_auth_header=None,
            mcp_server_auth_headers=None,
            client_ip="10.0.0.8",
        )

    assert exc_info.value.status_code == 403
    apply_scope_mock.assert_awaited_once()
    get_catalog_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_lazymcp_catalog_allowed_toolset_uses_scoped_auth(monkeypatch):
    user_auth = types.SimpleNamespace(api_key="sk-test")
    scoped_auth = types.SimpleNamespace(api_key="sk-scoped")
    get_catalog_mock = AsyncMock(return_value={"description": "ok"})
    apply_scope_mock = AsyncMock(return_value=scoped_auth)
    monkeypatch.setattr(
        "litellm.proxy._experimental.mcp_server.server._get_lazymcp_catalog",
        get_catalog_mock,
    )
    monkeypatch.setattr(
        "litellm.proxy._experimental.mcp_server.server._apply_toolset_scope",
        apply_scope_mock,
    )

    await LiteLLM_Proxy_MCP_Handler._get_lazymcp_gateway_tools(
        user_api_key_auth=user_auth,
        effective_filter=None,
        active_toolset_id="toolset-allowed",
        mcp_auth_header=None,
        mcp_server_auth_headers=None,
        client_ip="10.0.0.8",
    )

    apply_scope_mock.assert_awaited_once_with(user_auth, "toolset-allowed")
    assert get_catalog_mock.await_args is not None
    assert get_catalog_mock.await_args.kwargs["user_api_key_auth"] is scoped_auth


@pytest.mark.asyncio
async def test_execute_tool_calls_strips_server_prefix(monkeypatch):
    call_tool_mock = _setup_mcp_call_environment(monkeypatch)
    tool_name = "deepwiki-read_wiki_structure"
    tool_calls = [
        {
            "id": "call-1",
            "function": {"name": tool_name, "arguments": "{}"},
        }
    ]

    await LiteLLM_Proxy_MCP_Handler._execute_tool_calls(
        tool_server_map={tool_name: "deepwiki"},
        tool_calls=tool_calls,
        user_api_key_auth=None,
    )

    assert call_tool_mock.await_count == 1
    assert call_tool_mock.await_args is not None
    assert call_tool_mock.await_args.kwargs["name"] == "read_wiki_structure"


@pytest.mark.asyncio
async def test_execute_tool_calls_keeps_tool_name_without_prefix(monkeypatch):
    call_tool_mock = _setup_mcp_call_environment(monkeypatch)
    tool_name = "read_wiki_structure"
    tool_calls = [
        {
            "id": "call-2",
            "function": {"name": tool_name, "arguments": "{}"},
        }
    ]

    await LiteLLM_Proxy_MCP_Handler._execute_tool_calls(
        tool_server_map={tool_name: "deepwiki"},
        tool_calls=tool_calls,
        user_api_key_auth=None,
    )

    assert call_tool_mock.await_count == 1
    assert call_tool_mock.await_args is not None
    assert call_tool_mock.await_args.kwargs["name"] == tool_name


@pytest.mark.asyncio
async def test_execute_tool_calls_keeps_tool_name_when_equal_to_server(monkeypatch):
    call_tool_mock = _setup_mcp_call_environment(monkeypatch)
    tool_name = "echo"
    tool_calls = [
        {
            "id": "call-3",
            "function": {"name": tool_name, "arguments": "{}"},
        }
    ]

    await LiteLLM_Proxy_MCP_Handler._execute_tool_calls(
        tool_server_map={tool_name: "echo"},
        tool_calls=tool_calls,
        user_api_key_auth=None,
    )

    assert call_tool_mock.await_count == 1
    assert call_tool_mock.await_args is not None
    assert call_tool_mock.await_args.kwargs["name"] == tool_name


@pytest.mark.asyncio
async def test_execute_tool_calls_does_not_hijack_standard_mcp_name_collision(
    monkeypatch,
):
    call_tool_mock = _setup_mcp_call_environment(monkeypatch)
    tool_name = "mcp_call"
    tool_calls = [
        {
            "id": "call-standard-mcp",
            "function": {"name": tool_name, "arguments": "{}"},
        }
    ]

    await LiteLLM_Proxy_MCP_Handler._execute_tool_calls(
        tool_server_map={tool_name: "standard-server"},
        tool_calls=tool_calls,
        user_api_key_auth=None,
    )

    assert call_tool_mock.await_count == 1
    assert call_tool_mock.await_args is not None
    assert call_tool_mock.await_args.kwargs["server_name"] == "standard-server"
    assert call_tool_mock.await_args.kwargs["name"] == tool_name


@pytest.mark.asyncio
async def test_execute_tool_calls_passes_lazymcp_route_scope(monkeypatch):
    proxy_module = types.SimpleNamespace(proxy_logging_obj=object())
    monkeypatch.setitem(sys.modules, "litellm.proxy.proxy_server", proxy_module)

    captured = {}

    def fake_set_auth_context(**kwargs):
        captured.update(kwargs)

    async def fake_lazymcp_tool_call(_name, _arguments):
        return _DummyMCPResult()

    monkeypatch.setattr(
        "litellm.proxy._experimental.mcp_server.server.set_auth_context",
        fake_set_auth_context,
    )
    monkeypatch.setattr(
        "litellm.proxy._experimental.mcp_server.server.lazymcp_tool_call",
        fake_lazymcp_tool_call,
    )
    tool_server_map_value = (
        LiteLLM_Proxy_MCP_Handler._encode_lazymcp_tool_server_map_value(
            ["github"], None
        )
    )

    await LiteLLM_Proxy_MCP_Handler._execute_tool_calls(
        tool_server_map={"mcp_call": tool_server_map_value},
        tool_calls=[
            {
                "id": "call-lazy",
                "function": {
                    "name": "mcp_call",
                    "arguments": '{"server":"github","tool":"search","arguments":{}}',
                },
            }
        ],
        user_api_key_auth=None,
    )

    assert captured["mcp_servers"] == ["github"]


@pytest.mark.asyncio
async def test_execute_tool_calls_preserves_empty_lazymcp_scope(monkeypatch):
    proxy_module = types.SimpleNamespace(proxy_logging_obj=object())
    monkeypatch.setitem(sys.modules, "litellm.proxy.proxy_server", proxy_module)

    captured = {}

    def fake_set_auth_context(**kwargs):
        captured.update(kwargs)

    async def fake_lazymcp_tool_call(_name, _arguments):
        return _DummyMCPResult()

    monkeypatch.setattr(
        "litellm.proxy._experimental.mcp_server.server.set_auth_context",
        fake_set_auth_context,
    )
    monkeypatch.setattr(
        "litellm.proxy._experimental.mcp_server.server.lazymcp_tool_call",
        fake_lazymcp_tool_call,
    )
    tool_server_map_value = (
        LiteLLM_Proxy_MCP_Handler._encode_lazymcp_tool_server_map_value([], None)
    )

    await LiteLLM_Proxy_MCP_Handler._execute_tool_calls(
        tool_server_map={"mcp_call": tool_server_map_value},
        tool_calls=[
            {
                "id": "call-lazy-empty-scope",
                "function": {
                    "name": "mcp_call",
                    "arguments": '{"server":"github","tool":"search","arguments":{}}',
                },
            }
        ],
        user_api_key_auth=None,
    )

    assert captured["mcp_servers"] == []


@pytest.mark.asyncio
async def test_execute_tool_calls_passes_lazymcp_client_ip_and_scoped_permissions(
    monkeypatch,
):
    proxy_module = types.SimpleNamespace(proxy_logging_obj=object())
    monkeypatch.setitem(sys.modules, "litellm.proxy.proxy_server", proxy_module)

    captured = {}

    def fake_set_auth_context(**kwargs):
        captured.update(kwargs)

    async def fake_apply_toolset_scope(user_api_key_auth, toolset_id):
        captured["toolset_scope"] = {
            "user_api_key_auth": user_api_key_auth,
            "toolset_id": toolset_id,
        }
        return user_api_key_auth

    async def fake_lazymcp_tool_call(_name, _arguments):
        return _DummyMCPResult()

    monkeypatch.setattr(
        "litellm.proxy._experimental.mcp_server.server.set_auth_context",
        fake_set_auth_context,
    )
    monkeypatch.setattr(
        "litellm.proxy._experimental.mcp_server.server._apply_toolset_scope",
        fake_apply_toolset_scope,
    )
    monkeypatch.setattr(
        "litellm.proxy._experimental.mcp_server.server.lazymcp_tool_call",
        fake_lazymcp_tool_call,
    )

    user_auth = types.SimpleNamespace(api_key="sk-test")
    tool_server_map_value = (
        LiteLLM_Proxy_MCP_Handler._encode_lazymcp_tool_server_map_value(
            ["github"], "toolset-123"
        )
    )

    await LiteLLM_Proxy_MCP_Handler._execute_tool_calls(
        tool_server_map={"mcp_call": tool_server_map_value},
        tool_calls=[
            {
                "id": "call-lazy",
                "function": {
                    "name": "mcp_call",
                    "arguments": '{"server":"github","tool":"search","arguments":{}}',
                },
            }
        ],
        user_api_key_auth=user_auth,
        client_ip="10.0.0.8",
    )

    assert captured["client_ip"] == "10.0.0.8"
    assert captured["toolset_scope"] == {
        "user_api_key_auth": user_auth,
        "toolset_id": "toolset-123",
    }


@pytest.mark.asyncio
async def test_execute_tool_calls_rejects_unauthorized_lazymcp_toolset(monkeypatch):
    proxy_module = types.SimpleNamespace(proxy_logging_obj=object())
    monkeypatch.setitem(sys.modules, "litellm.proxy.proxy_server", proxy_module)

    async def fake_lazymcp_tool_call(_name, _arguments):
        return _DummyMCPResult()

    apply_scope_mock = AsyncMock(
        side_effect=HTTPException(status_code=403, detail="forbidden")
    )
    lazymcp_tool_call_mock = AsyncMock(side_effect=fake_lazymcp_tool_call)
    monkeypatch.setattr(
        "litellm.proxy._experimental.mcp_server.server._apply_toolset_scope",
        apply_scope_mock,
    )
    monkeypatch.setattr(
        "litellm.proxy._experimental.mcp_server.server.lazymcp_tool_call",
        lazymcp_tool_call_mock,
    )
    tool_server_map_value = (
        LiteLLM_Proxy_MCP_Handler._encode_lazymcp_tool_server_map_value(
            None, "toolset-blocked"
        )
    )

    results = await LiteLLM_Proxy_MCP_Handler._execute_tool_calls(
        tool_server_map={"mcp_call": tool_server_map_value},
        tool_calls=[
            {
                "id": "call-lazy-blocked",
                "function": {"name": "mcp_call", "arguments": "{}"},
            }
        ],
        user_api_key_auth=types.SimpleNamespace(api_key="sk-test"),
    )

    assert results[0]["tool_call_id"] == "call-lazy-blocked"
    assert "forbidden" in results[0]["result"]
    apply_scope_mock.assert_awaited_once()
    lazymcp_tool_call_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_execute_tool_calls_allowed_lazymcp_toolset_uses_scoped_auth(
    monkeypatch,
):
    proxy_module = types.SimpleNamespace(proxy_logging_obj=object())
    monkeypatch.setitem(sys.modules, "litellm.proxy.proxy_server", proxy_module)

    user_auth = types.SimpleNamespace(api_key="sk-test")
    scoped_auth = types.SimpleNamespace(api_key="sk-scoped")
    captured = {}

    def fake_set_auth_context(**kwargs):
        captured.update(kwargs)

    async def fake_lazymcp_tool_call(_name, _arguments):
        return _DummyMCPResult()

    apply_scope_mock = AsyncMock(return_value=scoped_auth)
    monkeypatch.setattr(
        "litellm.proxy._experimental.mcp_server.server._apply_toolset_scope",
        apply_scope_mock,
    )
    monkeypatch.setattr(
        "litellm.proxy._experimental.mcp_server.server.set_auth_context",
        fake_set_auth_context,
    )
    monkeypatch.setattr(
        "litellm.proxy._experimental.mcp_server.server.lazymcp_tool_call",
        fake_lazymcp_tool_call,
    )
    tool_server_map_value = (
        LiteLLM_Proxy_MCP_Handler._encode_lazymcp_tool_server_map_value(
            None, "toolset-allowed"
        )
    )

    await LiteLLM_Proxy_MCP_Handler._execute_tool_calls(
        tool_server_map={"mcp_call": tool_server_map_value},
        tool_calls=[
            {
                "id": "call-lazy-allowed",
                "function": {"name": "mcp_call", "arguments": "{}"},
            }
        ],
        user_api_key_auth=user_auth,
    )

    apply_scope_mock.assert_awaited_once_with(user_auth, "toolset-allowed")
    assert captured["user_api_key_auth"] is scoped_auth


@pytest.mark.asyncio
async def test_execute_tool_calls_ignores_spoofed_lazymcp_forwarded_header(
    monkeypatch,
):
    proxy_module = types.SimpleNamespace(proxy_logging_obj=object())
    monkeypatch.setitem(sys.modules, "litellm.proxy.proxy_server", proxy_module)

    captured = {}

    def fake_set_auth_context(**kwargs):
        captured.update(kwargs)

    async def fake_lazymcp_tool_call(_name, _arguments):
        return _DummyMCPResult()

    monkeypatch.setattr(
        "litellm.proxy._experimental.mcp_server.server.set_auth_context",
        fake_set_auth_context,
    )
    monkeypatch.setattr(
        "litellm.proxy._experimental.mcp_server.server.lazymcp_tool_call",
        fake_lazymcp_tool_call,
    )
    tool_server_map_value = (
        LiteLLM_Proxy_MCP_Handler._encode_lazymcp_tool_server_map_value(
            ["internal"], None
        )
    )

    await LiteLLM_Proxy_MCP_Handler._execute_tool_calls(
        tool_server_map={"mcp_call": tool_server_map_value},
        tool_calls=[
            {
                "id": "call-lazy",
                "function": {
                    "name": "mcp_call",
                    "arguments": '{"server":"internal","tool":"search","arguments":{}}',
                },
            }
        ],
        user_api_key_auth=None,
        raw_headers={"x-forwarded-for": "10.0.0.1"},
        client_ip="203.0.113.9",
    )

    assert captured["client_ip"] == "203.0.113.9"


@pytest.mark.asyncio
async def test_execute_tool_calls_passes_lazymcp_toolset_scope(monkeypatch):
    proxy_module = types.SimpleNamespace(proxy_logging_obj=object())
    monkeypatch.setitem(sys.modules, "litellm.proxy.proxy_server", proxy_module)

    captured = {}

    def fake_set_auth_context(**kwargs):
        captured.update(kwargs)

    async def fake_lazymcp_tool_call(_name, _arguments):
        from litellm.proxy._experimental.mcp_server.server import _mcp_active_toolset_id

        captured["active_toolset"] = _mcp_active_toolset_id.get()
        return _DummyMCPResult()

    monkeypatch.setattr(
        "litellm.proxy._experimental.mcp_server.server.set_auth_context",
        fake_set_auth_context,
    )
    monkeypatch.setattr(
        "litellm.proxy._experimental.mcp_server.server.lazymcp_tool_call",
        fake_lazymcp_tool_call,
    )
    tool_server_map_value = (
        LiteLLM_Proxy_MCP_Handler._encode_lazymcp_tool_server_map_value(
            None, "toolset-123"
        )
    )

    await LiteLLM_Proxy_MCP_Handler._execute_tool_calls(
        tool_server_map={"mcp_status": tool_server_map_value},
        tool_calls=[
            {
                "id": "call-lazy-status",
                "function": {"name": "mcp_status", "arguments": "{}"},
            }
        ],
        user_api_key_auth=None,
    )

    assert captured["mcp_servers"] is None
    assert captured["active_toolset"] == "toolset-123"


@pytest.mark.asyncio
async def test_execute_tool_calls_logs_failure_via_post_call_failure_hook(monkeypatch):
    """
    Regression test for ae4d92ad...:
    Ensure responses-side MCP tool execution logs failures via proxy_logging_obj.post_call_failure_hook.
    """
    post_call_failure_hook = _setup_proxy_logging(monkeypatch)

    fake_manager = types.SimpleNamespace(
        call_tool=AsyncMock(side_effect=HTTPException(status_code=500, detail="boom"))
    )
    monkeypatch.setattr(
        "litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager",
        fake_manager,
    )

    tool_name = "deepwiki-read_wiki_structure"
    tool_calls = [
        {"id": "call-err", "function": {"name": tool_name, "arguments": "{}"}}
    ]

    user_auth = types.SimpleNamespace(api_key="test_key", user_id="test_user")

    results = await LiteLLM_Proxy_MCP_Handler._execute_tool_calls(
        tool_server_map={tool_name: "deepwiki"},
        tool_calls=tool_calls,
        user_api_key_auth=user_auth,
        litellm_call_id="cid",
        litellm_trace_id="tid",
    )

    assert len(results) == 1
    assert results[0]["tool_call_id"] == "call-err"
    assert results[0]["name"] == tool_name

    post_call_failure_hook.assert_awaited_once()
    assert post_call_failure_hook.await_args is not None
    assert (
        post_call_failure_hook.await_args.kwargs.get("route")
        == "/responses/mcp/call_tool"
    )


@pytest.mark.asyncio
async def test_execute_tool_calls_passes_litellm_call_id_and_trace_id_to_function_setup(
    monkeypatch,
):
    """
    Regression test for ae4d92ad...:
    Ensure litellm_call_id / litellm_trace_id are forwarded into function_setup kwargs.
    """
    _setup_proxy_logging(monkeypatch)
    call_tool_mock = _setup_mcp_call_environment(monkeypatch)

    captured = {}

    def fake_function_setup(*_args, **kwargs):
        captured.update(kwargs)
        return None, None

    # NOTE: Don't patch via dotted string path here because `litellm.responses`
    # is a function attribute on the `litellm` package (shadowing the submodule),
    # which breaks monkeypatch's importpath resolution.
    handler_module = importlib.import_module(
        "litellm.responses.mcp.litellm_proxy_mcp_handler"
    )
    monkeypatch.setattr(handler_module, "function_setup", fake_function_setup)

    tool_name = "deepwiki-read_wiki_structure"
    tool_calls = [{"id": "call-1", "function": {"name": tool_name, "arguments": "{}"}}]

    await LiteLLM_Proxy_MCP_Handler._execute_tool_calls(
        tool_server_map={tool_name: "deepwiki"},
        tool_calls=tool_calls,
        user_api_key_auth=None,
        litellm_call_id="cid",
        litellm_trace_id="tid",
    )

    # Ensure the tool call was attempted (sanity)
    assert call_tool_mock.await_count == 1

    assert captured.get("litellm_call_id") == "cid"
    assert captured.get("litellm_trace_id") == "tid"


@pytest.mark.asyncio
async def test_get_mcp_tools_from_manager_enables_list_tools_logging(monkeypatch):
    """
    Regression test for 872e5b98...:
    Ensure responses-side tool discovery enables list-tools SpendLogs logging flags.
    """
    mock_get_tools = AsyncMock(return_value=[])
    monkeypatch.setattr(
        "litellm.proxy._experimental.mcp_server.server._get_tools_from_mcp_servers",
        mock_get_tools,
    )

    # Patch manager methods used by _get_mcp_tools_from_manager to avoid needing full UserAPIKeyAuth fields.
    fake_manager = types.SimpleNamespace(
        get_allowed_mcp_servers=AsyncMock(return_value=[]),
        get_mcp_servers_from_ids=MagicMock(return_value=[]),
        get_mcp_server_by_name=MagicMock(return_value=None),
    )
    monkeypatch.setattr(
        "litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager",
        fake_manager,
    )

    user_auth = types.SimpleNamespace(api_key="test_key", user_id="test_user")
    tools, _server_names = await LiteLLM_Proxy_MCP_Handler._get_mcp_tools_from_manager(
        user_api_key_auth=user_auth,
        mcp_tools_with_litellm_proxy=[
            {"type": "mcp", "server_url": "litellm_proxy/mcp/deepwiki"}
        ],
    )

    assert tools == []
    assert mock_get_tools.await_count == 1
    assert mock_get_tools.await_args is not None
    assert mock_get_tools.await_args.kwargs["log_list_tools_to_spendlogs"] is True
    assert mock_get_tools.await_args.kwargs["list_tools_log_source"] == "responses"


@pytest.mark.asyncio
async def test_standard_mcp_preserves_missing_client_ip_behavior(monkeypatch):
    captured = {}

    async def fake_standard_tools(**kwargs):
        captured.update(kwargs)
        return [], []

    monkeypatch.setattr(
        LiteLLM_Proxy_MCP_Handler,
        "_get_standard_mcp_tools",
        fake_standard_tools,
    )
    fake_manager = types.SimpleNamespace(get_mcp_server_by_name=MagicMock())
    monkeypatch.setattr(
        "litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager",
        fake_manager,
    )

    await LiteLLM_Proxy_MCP_Handler._get_mcp_tools_from_manager(
        user_api_key_auth=None,
        mcp_tools_with_litellm_proxy=[
            {"type": "mcp", "server_url": "litellm_proxy/mcp/standard"}
        ],
        client_ip=INVALID_MCP_CLIENT_IP_SENTINEL,
    )

    assert captured["client_ip"] is None


@pytest.mark.asyncio
async def test_standard_mcp_keeps_verified_client_ip(monkeypatch):
    captured = {}

    async def fake_standard_tools(**kwargs):
        captured.update(kwargs)
        return [], []

    monkeypatch.setattr(
        LiteLLM_Proxy_MCP_Handler,
        "_get_standard_mcp_tools",
        fake_standard_tools,
    )
    fake_manager = types.SimpleNamespace(get_mcp_server_by_name=MagicMock())
    monkeypatch.setattr(
        "litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager",
        fake_manager,
    )

    await LiteLLM_Proxy_MCP_Handler._get_mcp_tools_from_manager(
        user_api_key_auth=None,
        mcp_tools_with_litellm_proxy=[
            {"type": "mcp", "server_url": "litellm_proxy/mcp/standard"}
        ],
        client_ip="10.0.0.7",
    )

    assert captured["client_ip"] == "10.0.0.7"


@pytest.mark.asyncio
async def test_responses_non_streaming_auto_execution_passes_verified_client_ip(
    monkeypatch,
):
    from litellm.responses import main as responses_main

    tools = [{"type": "mcp", "server_url": "litellm_proxy/lazymcp/internal"}]
    captured_execute_kwargs = {}
    process_calls = []

    monkeypatch.setattr(
        LiteLLM_Proxy_MCP_Handler,
        "_parse_mcp_tools",
        staticmethod(lambda _tools: (tools, [])),
    )

    async def fake_process(**kwargs):
        process_calls.append(kwargs)
        return ([], {"mcp_call": 'lazymcp:{"mcp_servers":["internal"]}'})

    monkeypatch.setattr(
        LiteLLM_Proxy_MCP_Handler,
        "_process_mcp_tools_without_openai_transform",
        fake_process,
    )
    monkeypatch.setattr(
        LiteLLM_Proxy_MCP_Handler,
        "_transform_mcp_tools_to_openai",
        staticmethod(lambda *_args, **_kwargs: []),
    )
    monkeypatch.setattr(
        LiteLLM_Proxy_MCP_Handler,
        "_should_auto_execute_tools",
        staticmethod(lambda **_kwargs: True),
    )
    monkeypatch.setattr(
        LiteLLM_Proxy_MCP_Handler,
        "_extract_tool_calls_from_response",
        staticmethod(
            lambda **_kwargs: [
                {
                    "id": "call-1",
                    "function": {
                        "name": "mcp_call",
                        "arguments": '{"server":"internal","tool":"search","arguments":{}}',
                    },
                }
            ]
        ),
    )

    async def fake_execute(**kwargs):
        captured_execute_kwargs.update(kwargs)
        return [{"tool_call_id": "call-1", "result": "executed"}]

    monkeypatch.setattr(
        LiteLLM_Proxy_MCP_Handler,
        "_execute_tool_calls",
        fake_execute,
    )
    monkeypatch.setattr(
        responses_main.ResponsesAPIRequestUtils,
        "extract_mcp_headers_from_request",
        staticmethod(lambda **_kwargs: (None, None, None, None)),
    )
    monkeypatch.setattr(
        responses_main,
        "aresponses",
        AsyncMock(
            side_effect=[
                ResponsesAPIResponse(
                    id="resp-1",
                    model="test-model",
                    created_at=123,
                    output=[],
                    usage=ResponseAPIUsage(
                        input_tokens=1, output_tokens=1, total_tokens=2
                    ),
                ),
                ResponsesAPIResponse(
                    id="resp-2",
                    model="test-model",
                    created_at=124,
                    output=[],
                    usage=ResponseAPIUsage(
                        input_tokens=1, output_tokens=1, total_tokens=2
                    ),
                ),
            ]
        ),
    )
    monkeypatch.setattr(
        LiteLLM_Proxy_MCP_Handler,
        "_make_follow_up_call",
        AsyncMock(
            return_value=ResponsesAPIResponse(
                id="resp-2",
                model="test-model",
                created_at=124,
                output=[],
                usage=ResponseAPIUsage(input_tokens=1, output_tokens=1, total_tokens=2),
            )
        ),
    )

    await responses_main.aresponses_api_with_mcp(
        model="test-model",
        input="hello",
        tools=tools,
        secret_fields={"mcp_client_ip": "10.0.0.7"},
    )

    assert captured_execute_kwargs["client_ip"] == "10.0.0.7"
    assert [call["client_ip"] for call in process_calls] == [
        "10.0.0.7",
        "10.0.0.7",
    ]


def test_chat_streaming_iterator_execution_threads_client_ip():
    from litellm.responses.mcp import chat_completions_handler

    source = inspect.getsource(chat_completions_handler.acompletion_with_mcp)

    assert "client_ip=client_ip" in source
    assert "self.client_ip = client_ip" in source
    assert "client_ip=self.client_ip" in source
