import json
from collections.abc import Awaitable, Callable
from typing import Final, NamedTuple
from unittest.mock import AsyncMock

import httpx
import pytest
import respx
from mcp.types import Tool

import litellm
from litellm.caching.caching import DualCache
from litellm.exceptions import GuardrailRaisedException
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.proxy import proxy_server
from litellm.proxy._experimental.mcp_server import mcp_server_manager, server, tool_registry
from litellm.proxy._experimental.mcp_server.faults.list_outcomes import AggregateToolListing
from litellm.proxy._types import LiteLLM_ObjectPermissionTable, UserAPIKeyAuth
from litellm.proxy.utils import ProxyLogging
from litellm.types.mcp_server.mcp_server_manager import MCPServer

_UPSTREAM: Final = "https://upstream.example/v1"
_TOOL_NAME: Final = "ledger-commit_write"


def _ledger_tools() -> list[dict[str, str]]:
    return [{"type": "mcp", "server_url": "litellm_proxy/ledger", "require_approval": "never"}]


def _ledger_caller() -> UserAPIKeyAuth:
    return UserAPIKeyAuth(
        object_permission=LiteLLM_ObjectPermissionTable(object_permission_id="test", mcp_servers=["ledger"])
    )


class _LedgerGateway(NamedTuple):
    tool: AsyncMock
    list_tools: AsyncMock


@pytest.fixture
def ledger_gateway(monkeypatch: pytest.MonkeyPatch) -> _LedgerGateway:
    """Serve one auto-approved MCP tool through the real MCP manager and tool registry."""
    manager: Final = mcp_server_manager.MCPServerManager()
    manager.registry = {
        "ledger": MCPServer(
            server_id="ledger",
            name="ledger",
            server_name="ledger",
            transport="http",
            url="https://ledger.example/mcp",
            # a spec_path server runs its tools from the local tool registry instead of over the network
            spec_path="ledger.json",
            auth_type="none",
        )
    }
    manager.tool_name_to_mcp_server_name_mapping = {_TOOL_NAME: "ledger"}
    tool: Final = AsyncMock(return_value={"written": True})
    registry: Final = tool_registry.MCPToolRegistry()
    registry.register_tool(_TOOL_NAME, "Write one ledger entry", {"type": "object"}, tool)
    list_tools: Final = AsyncMock(
        return_value=AggregateToolListing(tools=[Tool(name=_TOOL_NAME, inputSchema={"type": "object"})], outcomes={})
    )
    monkeypatch.setattr(tool_registry, "global_mcp_tool_registry", registry)
    monkeypatch.setattr(mcp_server_manager, "global_mcp_server_manager", manager)
    monkeypatch.setattr(proxy_server, "proxy_logging_obj", ProxyLogging(user_api_key_cache=DualCache()))
    monkeypatch.setattr(server, "_get_tools_from_mcp_servers", list_tools)
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    return _LedgerGateway(tool=tool, list_tools=list_tools)


def _is_follow_up(request: httpx.Request) -> bool:
    body: Final = json.loads(request.content)
    messages: Final = body.get("messages") or []
    items: Final = body["input"] if isinstance(body.get("input"), list) else []
    return any(message.get("role") == "tool" for message in messages) or any(
        item.get("type") == "function_call_output" for item in items
    )


def _chat_completion(message: dict[str, object], finish_reason: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 0,
            "model": "gpt-4.1-mini",
            "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    )


def _responses_api_response(output_item: dict[str, object]) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "resp_1",
            "object": "response",
            "created_at": 0,
            "status": "completed",
            "model": "gpt-4.1-mini",
            "output": [output_item],
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
            "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        },
    )


def _tool_call(responses_api: bool) -> httpx.Response:
    arguments: Final = json.dumps({"entry": "alpha"})
    if responses_api:
        return _responses_api_response(
            {"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": _TOOL_NAME, "arguments": arguments}
        )
    return _chat_completion(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": _TOOL_NAME, "arguments": arguments}}
            ],
        },
        "tool_calls",
    )


def _final_answer(responses_api: bool) -> httpx.Response:
    if responses_api:
        return _responses_api_response(
            {
                "type": "message",
                "id": "msg_1",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "Recorded alpha", "annotations": []}],
            }
        )
    return _chat_completion({"role": "assistant", "content": "Recorded alpha"}, "stop")


class _FakeModel:
    """Answers both OpenAI routes: a tool call, then a final answer once the tool result arrives."""

    def __init__(self, failing_initial_calls: int = 0, failing_follow_ups: int = 0) -> None:
        self.calls: list[str] = []
        self._failures: Final = {"initial": failing_initial_calls, "follow_up": failing_follow_ups}

    def __call__(self, request: httpx.Request) -> httpx.Response:
        kind: Final = "follow_up" if _is_follow_up(request) else "initial"
        self.calls.append(kind)
        if self.calls.count(kind) <= self._failures[kind]:
            return httpx.Response(500, json={"error": {"message": "upstream failed", "type": "server_error"}})
        responses_api: Final = request.url.path.endswith("/responses")
        return _tool_call(responses_api) if kind == "initial" else _final_answer(responses_api)


def _serve(model: _FakeModel) -> None:
    respx.post(f"{_UPSTREAM}/chat/completions").mock(side_effect=model)
    respx.post(f"{_UPSTREAM}/responses").mock(side_effect=model)


def _router(num_retries: int, fallbacks: list[dict[str, list[str]]] | None = None) -> litellm.Router:
    deployment: Final = {"model": "openai/gpt-4.1-mini", "api_key": "sk-test", "api_base": _UPSTREAM}
    return litellm.Router(
        # two replicas, so the router retries right away instead of backing off
        model_list=[
            {"model_name": "repro-model", "litellm_params": deployment},
            {"model_name": "repro-model", "litellm_params": deployment},
            {"model_name": "backup-model", "litellm_params": deployment},
        ],
        num_retries=num_retries,
        fallbacks=fallbacks,
    )


def _chat(router: litellm.Router) -> Awaitable[object]:
    return router.acompletion(
        model="repro-model",
        messages=[{"role": "user", "content": "Record entry alpha"}],
        tools=_ledger_tools(),
        metadata={"user_api_key_auth": _ledger_caller()},
    )


def _responses(router: litellm.Router) -> Awaitable[object]:
    return router.aresponses(
        model="repro-model",
        input="Record entry alpha",
        tools=_ledger_tools(),
        litellm_metadata={"user_api_key_auth": _ledger_caller()},
    )


def _messages(router: litellm.Router) -> Awaitable[object]:
    return router.aanthropic_messages(
        model="repro-model",
        max_tokens=64,
        messages=[{"role": "user", "content": "Record entry alpha"}],
        tools=_ledger_tools(),
        litellm_metadata={"user_api_key_auth": _ledger_caller()},
    )


_ENDPOINTS: Final = pytest.mark.parametrize(
    "send", [_chat, _responses, _messages], ids=["chat_completions", "responses", "messages"]
)


@pytest.mark.asyncio
@_ENDPOINTS
@pytest.mark.parametrize(
    "router_kwargs",
    [{"num_retries": 2}, {"num_retries": 0, "fallbacks": [{"repro-model": ["backup-model"]}]}],
    ids=["retries", "fallbacks"],
)
@respx.mock
async def test_router_does_not_replay_an_executed_mcp_tool(
    ledger_gateway: _LedgerGateway,
    send: Callable[[litellm.Router], Awaitable[object]],
    router_kwargs: dict[str, object],
):
    """
    Regression test for #43153: an auto-approved MCP tool must run once per request.

    Given: The model asks for the tool, the tool runs, and every follow-up model call returns a 500
    When:  The router has retries or a fallback group configured
    Then:  The client gets the 500 after one tool execution and one follow-up call

    A retry or fallback re-runs the whole MCP loop, so before the fix every replay executed
    the tool again (three writes with num_retries=2, two with one fallback).
    """
    model: Final = _FakeModel(failing_follow_ups=3)
    _serve(model)

    with pytest.raises(litellm.InternalServerError):
        await send(_router(**router_kwargs))

    assert ledger_gateway.tool.await_count == 1
    assert model.calls == ["initial", "follow_up"]


@pytest.mark.asyncio
@_ENDPOINTS
@respx.mock
async def test_router_retries_when_a_guardrail_blocked_every_mcp_tool(
    ledger_gateway: _LedgerGateway,
    send: Callable[[litellm.Router], Awaitable[object]],
    monkeypatch: pytest.MonkeyPatch,
):
    """
    Given: A pre-call guardrail blocks the tool call, so no tool runs
    When:  The follow-up model call fails once with a 500
    Then:  The router still retries, because replaying the request cannot repeat a side effect
    """

    class BlockEveryToolCall(CustomGuardrail):
        async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
            raise GuardrailRaisedException(message="tool calls are blocked", blocked_content=True)

    monkeypatch.setattr(
        litellm, "callbacks", [BlockEveryToolCall(guardrail_name="block", event_hook="pre_mcp_call", default_on=True)]
    )
    model: Final = _FakeModel(failing_follow_ups=1)
    _serve(model)

    await send(_router(num_retries=2))

    assert ledger_gateway.tool.await_count == 0
    assert model.calls == ["initial", "follow_up", "initial", "follow_up"]


@pytest.mark.asyncio
@respx.mock
async def test_router_still_retries_a_failure_before_any_mcp_tool_ran(ledger_gateway: _LedgerGateway):
    """
    Given: The initial model call fails once with a 500, then asks for the tool
    When:  The router retries
    Then:  The retry succeeds and the tool runs once
    """
    model: Final = _FakeModel(failing_initial_calls=1)
    _serve(model)

    await _chat(_router(num_retries=2))

    assert ledger_gateway.tool.await_count == 1
    assert model.calls == ["initial", "initial", "follow_up"]


@pytest.mark.asyncio
@respx.mock
async def test_responses_tool_listing_failure_after_a_tool_ran_is_not_replayed(ledger_gateway: _LedgerGateway):
    """
    Given: A Responses request whose tool ran and whose follow-up succeeded
    When:  Listing the MCP tools again for the output items fails
    Then:  The router surfaces that failure instead of replaying the request and the tool
    """
    listing: Final = ledger_gateway.list_tools.return_value
    listing_error: Final = RuntimeError("tool listing failed")
    ledger_gateway.list_tools.side_effect = [listing, listing_error, listing, listing_error, listing, listing_error]
    model: Final = _FakeModel()
    _serve(model)

    with pytest.raises(Exception, match="tool listing failed"):
        await _responses(_router(num_retries=2))

    assert ledger_gateway.tool.await_count == 1
    assert ledger_gateway.list_tools.await_count == 2
    assert model.calls == ["initial", "follow_up"]
