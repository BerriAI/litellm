import json
import subprocess
import sys
import textwrap
import types
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException
import importlib

from litellm.proxy._experimental.mcp_server.faults.list_outcomes import AggregateToolListing
from litellm.responses.mcp.litellm_proxy_mcp_handler import (
    LiteLLM_Proxy_MCP_Handler,
)
from litellm.constants import MCP_VIRTUAL_TOOL_SEARCH_SERVER_NAME
from litellm.responses.mcp.tool_search_bridge import virtual_tool_server_map
from typing import Any, cast
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
        get_registry=MagicMock(return_value={}),
        call_tool=AsyncMock(return_value=_DummyMCPResult()),
        # Newer logging path calls this to enrich spend logs metadata
        _get_mcp_server_from_tool_name=MagicMock(return_value=None),
        get_mcp_server_by_name=MagicMock(return_value=None),
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
async def test_execute_tool_calls_strips_prefix_when_alias_differs_from_server_name(
    monkeypatch,
):
    call_tool_mock = _setup_mcp_call_environment(monkeypatch)
    fake_server = types.SimpleNamespace(
        alias="my_deepwiki",
        server_name="deepwiki_test",
        server_id="test-server-id",
        short_prefix=None,
        mcp_info=None,
        tool_name_to_display_name=None,
    )
    from litellm.proxy._experimental.mcp_server import mcp_server_manager as _msm

    _msm.global_mcp_server_manager._get_mcp_server_from_tool_name = MagicMock(
        return_value=fake_server
    )

    tool_name = "my_deepwiki-read_wiki_structure"
    tool_calls = [
        {
            "id": "call-4",
            "function": {"name": tool_name, "arguments": "{}"},
        }
    ]

    await LiteLLM_Proxy_MCP_Handler._execute_tool_calls(
        tool_server_map={tool_name: "deepwiki_test"},
        tool_calls=tool_calls,
        user_api_key_auth=None,
    )

    assert call_tool_mock.await_count == 1
    assert call_tool_mock.await_args is not None
    assert call_tool_mock.await_args.kwargs["name"] == "read_wiki_structure"


@pytest.mark.asyncio
async def test_execute_tool_calls_reverse_maps_display_name(monkeypatch):
    call_tool_mock = _setup_mcp_call_environment(monkeypatch)
    colliding_server = types.SimpleNamespace(
        alias=None,
        server_name="other_mcp",
        server_id="other-server-id",
        short_prefix=None,
        mcp_info=None,
        tool_name_to_display_name={"search": "search_docs"},
    )
    fake_server = types.SimpleNamespace(
        alias=None,
        server_name="deepwiki_mcp",
        server_id="test-server-id",
        short_prefix=None,
        mcp_info=None,
        tool_name_to_display_name={"read_wiki_structure": "browse_repo_docs"},
    )
    from litellm.proxy._experimental.mcp_server import mcp_server_manager as _msm

    _msm.global_mcp_server_manager._get_mcp_server_from_tool_name = MagicMock(return_value=colliding_server)
    _msm.global_mcp_server_manager.get_mcp_server_by_name = MagicMock(return_value=fake_server)

    tool_name = "browse_repo_docs"
    tool_calls = [
        {
            "id": "call-5",
            "function": {"name": tool_name, "arguments": "{}"},
        }
    ]

    await LiteLLM_Proxy_MCP_Handler._execute_tool_calls(
        tool_server_map={tool_name: "deepwiki_mcp"},
        tool_calls=tool_calls,
        user_api_key_auth=None,
    )

    assert call_tool_mock.await_count == 1
    assert call_tool_mock.await_args is not None
    assert call_tool_mock.await_args.kwargs["name"] == "read_wiki_structure"


@pytest.mark.asyncio
async def test_execute_tool_calls_logs_failure_via_post_call_failure_hook(monkeypatch):
    """
    Regression test for ae4d92ad...:
    Ensure responses-side MCP tool execution logs failures via proxy_logging_obj.post_call_failure_hook.
    """
    post_call_failure_hook = _setup_proxy_logging(monkeypatch)

    fake_manager = types.SimpleNamespace(
        get_registry=MagicMock(return_value={}),
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
async def test_execute_tool_calls_threads_logging_obj_into_call_tool(monkeypatch):
    """The Responses-API MCP path must hand the request's litellm_logging_obj to
    global_mcp_server_manager.call_tool, otherwise pre_call_tool_check /
    _create_during_hook_task get None and no guardrail evaluation is bridged onto
    the request logger, so MCP tool calls made through the Responses API report zero
    guardrail evaluations in the monitor. Drop the litellm_logging_obj kwarg on the
    call_tool invocation and this fails."""
    _setup_proxy_logging(monkeypatch)
    call_tool_mock = _setup_mcp_call_environment(monkeypatch)

    sentinel_logging_obj = MagicMock()
    sentinel_logging_obj.async_post_mcp_tool_call_hook = AsyncMock()
    sentinel_logging_obj.async_success_handler = AsyncMock()

    handler_module = importlib.import_module("litellm.responses.mcp.litellm_proxy_mcp_handler")
    monkeypatch.setattr(
        handler_module,
        "function_setup",
        lambda *_args, **_kwargs: (sentinel_logging_obj, None),
    )

    tool_name = "deepwiki-read_wiki_structure"
    tool_calls = [{"id": "call-1", "function": {"name": tool_name, "arguments": "{}"}}]

    await LiteLLM_Proxy_MCP_Handler._execute_tool_calls(
        tool_server_map={tool_name: "deepwiki"},
        tool_calls=tool_calls,
        user_api_key_auth=None,
    )

    assert call_tool_mock.await_count == 1
    assert call_tool_mock.await_args is not None
    assert call_tool_mock.await_args.kwargs["litellm_logging_obj"] is sentinel_logging_obj


@pytest.mark.asyncio
async def test_get_mcp_tools_from_manager_enables_list_tools_logging(monkeypatch):
    """
    Regression test for 872e5b98...:
    Ensure responses-side tool discovery enables list-tools SpendLogs logging flags.
    """
    mock_get_tools = AsyncMock(return_value=AggregateToolListing(tools=[], outcomes={}))
    monkeypatch.setattr(
        "litellm.proxy._experimental.mcp_server.server._get_tools_from_mcp_servers",
        mock_get_tools,
    )

    # Patch manager methods used by _get_mcp_tools_from_manager to avoid needing full UserAPIKeyAuth fields.
    fake_manager = types.SimpleNamespace(
        get_registry=MagicMock(return_value={}),
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


def test_get_parent_request_tags_from_metadata():
    tags = LiteLLM_Proxy_MCP_Handler._get_parent_request_tags(
        {"metadata": {"tags": ["team-a", "prod"]}}
    )
    assert tags == ["team-a", "prod"]


def test_get_parent_request_tags_from_nested_litellm_params():
    tags = LiteLLM_Proxy_MCP_Handler._get_parent_request_tags(
        {
            "metadata": {"tags": ["top-level"]},
            "litellm_params": {
                "metadata": {"tags": ["nested"]},
                "proxy_server_request": {"headers": {"user-agent": "client/1.0"}},
            },
        }
    )
    assert tags == ["nested", "User-Agent: client", "User-Agent: client/1.0"]


@pytest.mark.asyncio
async def test_get_mcp_tools_from_manager_forwards_request_tags(monkeypatch):
    mock_get_tools = AsyncMock(return_value=AggregateToolListing(tools=[], outcomes={}))
    monkeypatch.setattr(
        "litellm.proxy._experimental.mcp_server.server._get_tools_from_mcp_servers",
        mock_get_tools,
    )
    fake_manager = types.SimpleNamespace(
        get_registry=MagicMock(return_value={}),
        get_allowed_mcp_servers=AsyncMock(return_value=[]),
        get_mcp_servers_from_ids=MagicMock(return_value=[]),
        get_mcp_server_by_name=MagicMock(return_value=None),
    )
    monkeypatch.setattr(
        "litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager",
        fake_manager,
    )

    await LiteLLM_Proxy_MCP_Handler._get_mcp_tools_from_manager(
        user_api_key_auth=types.SimpleNamespace(api_key="k", user_id="u"),
        mcp_tools_with_litellm_proxy=[
            {"type": "mcp", "server_url": "litellm_proxy/mcp/deepwiki"}
        ],
        request_tags=["team-a"],
    )

    assert mock_get_tools.await_args.kwargs["request_tags"] == ["team-a"]


@pytest.mark.asyncio
async def test_execute_tool_calls_exposes_sanitized_client_headers_to_logging(monkeypatch):
    """The Responses API MCP bridge used to log an empty header dict, hiding the caller's
    headers from logging callbacks and hooks."""
    _setup_proxy_logging(monkeypatch)
    _setup_mcp_call_environment(monkeypatch)

    captured = {}

    def fake_function_setup(*_args, **kwargs):
        captured.update(kwargs)
        return None, None

    handler_module = importlib.import_module(
        "litellm.responses.mcp.litellm_proxy_mcp_handler"
    )
    monkeypatch.setattr(handler_module, "function_setup", fake_function_setup)

    tool_name = "deepwiki-read_wiki_structure"
    await LiteLLM_Proxy_MCP_Handler._execute_tool_calls(
        tool_server_map={tool_name: "deepwiki"},
        tool_calls=[{"id": "call-1", "function": {"name": tool_name, "arguments": "{}"}}],
        user_api_key_auth=None,
        raw_headers={"x-nuid": "nuid-1", "x-litellm-api-key": "sk-proxy", "cookie": "s=1"},
    )

    expected = {"x-nuid": "nuid-1", "cookie": "***REDACTED***"}
    assert captured["metadata"]["headers"] == expected
    assert captured["proxy_server_request"]["headers"] == expected


@pytest.mark.asyncio
async def test_execute_tool_calls_propagates_request_tags_to_function_setup(monkeypatch):
    _setup_proxy_logging(monkeypatch)
    _setup_mcp_call_environment(monkeypatch)
    captured = {}

    def fake_function_setup(*_args, **kwargs):
        captured.update(kwargs)
        return None, None

    handler_module = importlib.import_module(
        "litellm.responses.mcp.litellm_proxy_mcp_handler"
    )
    monkeypatch.setattr(handler_module, "function_setup", fake_function_setup)

    tool_name = "deepwiki-read_wiki_structure"
    await LiteLLM_Proxy_MCP_Handler._execute_tool_calls(
        tool_server_map={tool_name: "deepwiki"},
        tool_calls=[{"id": "call-1", "function": {"name": tool_name, "arguments": "{}"}}],
        user_api_key_auth=None,
        request_tags=["team-a", "prod"],
    )

    assert captured["metadata"]["tags"] == ["team-a", "prod"]


def test_completion_with_function_tools_works_without_fastapi_installed():
    script = textwrap.dedent(
        """
        import sys

        class _FastapiBlocker:
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "fastapi" or fullname.startswith("fastapi."):
                    raise ModuleNotFoundError("No module named 'fastapi'")
                return None

        sys.meta_path.insert(0, _FastapiBlocker())

        import litellm

        response = litellm.completion(
            model="openai/gpt-5.5",
            messages=[{"role": "user", "content": "What is the weather in SF?"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get the current weather for a location",
                        "parameters": {
                            "type": "object",
                            "properties": {"location": {"type": "string"}},
                            "required": ["location"],
                        },
                    },
                }
            ],
            mock_response="sunny",
        )
        assert response.choices[0].message.content == "sunny"
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr


def test_extract_tool_call_details_reads_anthropic_tool_use_input():
    """
    Regression test (LIT-4517): an Anthropic tool_use block carries its arguments
    under `input`, not `arguments`.

    Given: A tool_use content block as /v1/messages returns it
    When:  The shared extractor reads it
    Then:  The arguments come back, so the MCP tool is called with them

    Reading only `arguments` fails silently rather than loudly: _parse_tool_arguments
    turns the resulting None into {}, so the tool still executes, just with every
    argument dropped.
    """
    tool_use_block = {
        "type": "tool_use",
        "id": "toolu_01ABC",
        "name": "read_wiki_structure",
        "input": {"repoName": "BerriAI/litellm"},
    }

    name, arguments, call_id = LiteLLM_Proxy_MCP_Handler._extract_tool_call_details(tool_use_block)

    assert name == "read_wiki_structure"
    assert call_id == "toolu_01ABC"
    assert arguments == {"repoName": "BerriAI/litellm"}
    assert LiteLLM_Proxy_MCP_Handler._parse_tool_arguments(arguments) == {"repoName": "BerriAI/litellm"}


def test_extract_tool_call_details_still_prefers_openai_arguments():
    """The OpenAI chat shape must keep winning; `input` is only the fallback."""
    openai_tool_call = {
        "id": "call_123",
        "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
    }

    name, arguments, call_id = LiteLLM_Proxy_MCP_Handler._extract_tool_call_details(openai_tool_call)

    assert name == "get_weather"
    assert call_id == "call_123"
    assert arguments == '{"city": "Paris"}'
# mcp_tool_search virtual tools on the Responses API type: "mcp" path


def _mcp_tool_config(server: str = "deepwiki", allowed_tools=None):
    config = {
        "type": "mcp",
        "server_label": "x",
        "server_url": f"litellm_proxy/mcp/{server}",
        "require_approval": "never",
    }
    if allowed_tools is not None:
        config["allowed_tools"] = allowed_tools
    return config


def _auth(tool_search_enabled=None):
    """Real UserAPIKeyAuth; tool_search_enabled=None means no object_permission at all."""
    from litellm.models.object_permission import LiteLLM_ObjectPermissionTable
    from litellm.proxy._types import UserAPIKeyAuth

    if tool_search_enabled is None:
        return UserAPIKeyAuth(api_key="k", user_id="u")
    return UserAPIKeyAuth(
        api_key="k",
        user_id="u",
        object_permission=LiteLLM_ObjectPermissionTable(
            object_permission_id="test",
            mcp_tool_search_enabled=tool_search_enabled,
        ),
    )


def _setup_async_proxy_logging(monkeypatch: pytest.MonkeyPatch) -> None:
    """proxy_logging_obj with awaitable hooks, needed once a tool actually dispatches."""
    proxy_logging_obj = MagicMock()
    proxy_logging_obj.post_call_failure_hook = AsyncMock()
    proxy_logging_obj.post_mcp_call_hook = AsyncMock(side_effect=lambda **kwargs: kwargs["response"])
    monkeypatch.setitem(
        sys.modules,
        "litellm.proxy.proxy_server",
        types.SimpleNamespace(proxy_logging_obj=proxy_logging_obj),
    )


def _text_result(text: str, is_error: bool = False):
    from mcp.types import CallToolResult, TextContent

    return CallToolResult(content=[TextContent(type="text", text=text)], isError=is_error)


def _patch_catalog_listing(monkeypatch):
    """Patch the manager listing seam and return the mock, so a test can assert it was never used."""
    from mcp.types import Tool

    mock_get_tools = AsyncMock(
        return_value=AggregateToolListing(
            tools=[
                Tool(
                    name="deepwiki-read_wiki_structure",
                    description="read the wiki structure",
                    inputSchema={"type": "object", "properties": {}},
                )
            ],
            outcomes={},
        )
    )
    monkeypatch.setattr(
        "litellm.proxy._experimental.mcp_server.server._get_tools_from_mcp_servers",
        mock_get_tools,
    )
    monkeypatch.setattr(
        "litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager",
        types.SimpleNamespace(
            get_registry=MagicMock(return_value={}),
            get_allowed_mcp_servers=AsyncMock(return_value=[]),
            get_mcp_servers_from_ids=MagicMock(return_value=[]),
            get_mcp_server_by_name=MagicMock(return_value=None),
        ),
    )
    return mock_get_tools


@pytest.mark.asyncio
async def test_process_mcp_tools_returns_only_virtual_tools_when_flag_enabled(monkeypatch):
    """With mcp_tool_search_enabled the catalog must never be listed; the model
    sees only mcp_tool_search + mcp_tool_call."""
    mock_get_tools = _patch_catalog_listing(monkeypatch)

    openai_tools, tool_server_map = await LiteLLM_Proxy_MCP_Handler._process_mcp_tools_to_openai_format(
        user_api_key_auth=_auth(True),
        mcp_tools_with_litellm_proxy=[_mcp_tool_config()],
    )

    assert [tool["name"] for tool in openai_tools] == ["mcp_tool_search", "mcp_tool_call"]
    assert all(tool["type"] == "function" for tool in openai_tools)
    assert tool_server_map == {
        "mcp_tool_search": MCP_VIRTUAL_TOOL_SEARCH_SERVER_NAME,
        "mcp_tool_call": MCP_VIRTUAL_TOOL_SEARCH_SERVER_NAME,
    }
    assert mock_get_tools.await_count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tool_search_enabled,require_approval",
    [
        (False, "never"),
        (None, "never"),
        (True, None),
    ],
    ids=["flag_false", "no_object_permission", "flag_on_but_approval_required"],
)
async def test_process_mcp_tools_injects_catalog_unchanged(monkeypatch, tool_search_enabled, require_approval):
    """Regression: the catalog path is untouched for keys without the flag, and
    the virtual tools are skipped when they could not be auto-executed."""
    mock_get_tools = _patch_catalog_listing(monkeypatch)
    tool_config = _mcp_tool_config()
    if require_approval is None:
        tool_config.pop("require_approval")

    openai_tools, tool_server_map = await LiteLLM_Proxy_MCP_Handler._process_mcp_tools_to_openai_format(
        user_api_key_auth=_auth(tool_search_enabled),
        mcp_tools_with_litellm_proxy=[tool_config],
    )

    assert [tool["name"] for tool in openai_tools] == ["deepwiki-read_wiki_structure"]
    assert tool_server_map == {"deepwiki-read_wiki_structure": "deepwiki"}
    assert mock_get_tools.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "allowed_tools,expected_names",
    [
        (["read_wiki_structure"], ["deepwiki-read_wiki_structure"]),
        (["something_else"], []),
    ],
    ids=["keeps_allowed", "filters_everything_out"],
)
async def test_execute_tool_calls_scopes_mcp_tool_search(monkeypatch, allowed_tools, expected_names):
    """mcp_tool_search is scoped to the request's server aliases, and its results
    are filtered to allowed_tools; filtering everything out is an empty list, not an error."""
    _setup_proxy_logging(monkeypatch)
    _setup_mcp_call_environment(monkeypatch)

    search_mock = AsyncMock(
        return_value=_text_result(
            json.dumps(
                [
                    {"name": "deepwiki-read_wiki_structure", "description": "d", "inputSchema": {}},
                    {"name": "deepwiki-ask_question", "description": "d", "inputSchema": {}},
                ]
            )
        )
    )
    monkeypatch.setattr(
        "litellm.proxy._experimental.mcp_server.tool_search.handle_mcp_tool_search",
        search_mock,
    )

    results = await LiteLLM_Proxy_MCP_Handler._execute_tool_calls(
        tool_server_map=virtual_tool_server_map(),
        tool_calls=[
            {
                "id": "call-search",
                "function": {"name": "mcp_tool_search", "arguments": json.dumps({"query": "wiki", "top_k": 3})},
            }
        ],
        user_api_key_auth=_auth(True),
        virtual_tool_scope=LiteLLM_Proxy_MCP_Handler._build_virtual_tool_scope(
            [_mcp_tool_config(allowed_tools=allowed_tools)]
        ),
    )

    assert search_mock.await_args.kwargs["mcp_servers"] == ["deepwiki"]
    assert search_mock.await_args.kwargs["query"] == "wiki"
    assert search_mock.await_args.kwargs["top_k"] == 3
    assert [entry["name"] for entry in json.loads(results[0]["result"])] == expected_names


@pytest.mark.asyncio
async def test_execute_tool_calls_mcp_tool_call_dispatches_inner_tool_with_auth_headers(monkeypatch):
    """mcp_tool_call unwraps to the real tool, forwards the per-request auth
    headers and spend-logs the inner tool name rather than mcp_tool_call."""
    _setup_mcp_call_environment(monkeypatch)
    _setup_async_proxy_logging(monkeypatch)

    call_mock = AsyncMock(return_value=_text_result("wiki structure"))
    monkeypatch.setattr(
        "litellm.proxy._experimental.mcp_server.tool_search.handle_mcp_tool_call",
        call_mock,
    )

    captured = {}

    def fake_function_setup(*_args, **kwargs):
        captured.update(kwargs)
        return None, None

    handler_module = importlib.import_module("litellm.responses.mcp.litellm_proxy_mcp_handler")
    monkeypatch.setattr(handler_module, "function_setup", fake_function_setup)

    results = await LiteLLM_Proxy_MCP_Handler._execute_tool_calls(
        tool_server_map=virtual_tool_server_map(),
        tool_calls=[
            {
                "id": "call-inner",
                "function": {
                    "name": "mcp_tool_call",
                    "arguments": json.dumps(
                        {"tool_name": "deepwiki-read_wiki_structure", "arguments": {"repo": "litellm"}}
                    ),
                },
            }
        ],
        user_api_key_auth=_auth(True),
        mcp_auth_header="legacy-token",
        mcp_server_auth_headers={"deepwiki": {"authorization": "Bearer per-server"}},
        oauth2_headers={"x-oauth": "1"},
        raw_headers={"x-raw": "1"},
        virtual_tool_scope=LiteLLM_Proxy_MCP_Handler._build_virtual_tool_scope([_mcp_tool_config()]),
    )

    call_kwargs = call_mock.await_args.kwargs
    assert call_kwargs["tool_name"] == "deepwiki-read_wiki_structure"
    assert call_kwargs["arguments"] == {"repo": "litellm"}
    assert call_kwargs["mcp_servers"] == ["deepwiki"]
    assert call_kwargs["mcp_auth_header"] == "legacy-token"
    assert call_kwargs["mcp_server_auth_headers"] == {"deepwiki": {"authorization": "Bearer per-server"}}
    assert call_kwargs["oauth2_headers"] == {"x-oauth": "1"}
    assert call_kwargs["raw_headers"] == {"x-raw": "1"}

    assert captured["model"] == "MCP: deepwiki-read_wiki_structure"
    assert captured["metadata"]["tool_name"] == "read_wiki_structure"
    assert captured["input"][0]["content"]["arguments"] == {"repo": "litellm"}
    assert results[0]["result"] == "wiki structure"
    assert results[0]["display_name"] == "deepwiki-read_wiki_structure"


@pytest.mark.asyncio
async def test_execute_tool_calls_mcp_tool_call_rejects_tool_outside_allowed_tools(monkeypatch):
    _setup_proxy_logging(monkeypatch)
    _setup_mcp_call_environment(monkeypatch)
    call_mock = AsyncMock(return_value=_text_result("should not run"))
    monkeypatch.setattr(
        "litellm.proxy._experimental.mcp_server.tool_search.handle_mcp_tool_call",
        call_mock,
    )

    results = await LiteLLM_Proxy_MCP_Handler._execute_tool_calls(
        tool_server_map=virtual_tool_server_map(),
        tool_calls=[
            {
                "id": "call-inner",
                "function": {
                    "name": "mcp_tool_call",
                    "arguments": json.dumps({"tool_name": "deepwiki-delete_everything", "arguments": {}}),
                },
            }
        ],
        user_api_key_auth=_auth(True),
        virtual_tool_scope=LiteLLM_Proxy_MCP_Handler._build_virtual_tool_scope(
            [_mcp_tool_config(allowed_tools=["read_wiki_structure"])]
        ),
    )

    assert call_mock.await_count == 0
    assert "not in allowed_tools" in results[0]["result"]


@pytest.mark.asyncio
async def test_execute_tool_calls_ignores_virtual_names_without_scope(monkeypatch):
    """A real upstream tool literally named mcp_tool_search must still dispatch
    normally when the virtual tools are not in play."""
    call_tool_mock = _setup_mcp_call_environment(monkeypatch)
    search_mock = AsyncMock(return_value=_text_result("[]"))
    monkeypatch.setattr(
        "litellm.proxy._experimental.mcp_server.tool_search.handle_mcp_tool_search",
        search_mock,
    )

    await LiteLLM_Proxy_MCP_Handler._execute_tool_calls(
        tool_server_map={"mcp_tool_search": "deepwiki"},
        tool_calls=[{"id": "c", "function": {"name": "mcp_tool_search", "arguments": "{}"}}],
        user_api_key_auth=None,
    )

    assert search_mock.await_count == 0
    assert call_tool_mock.await_count == 1
