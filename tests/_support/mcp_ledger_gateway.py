import json
from collections.abc import Mapping, Sequence
from typing import Final

import httpx
import pytest
import respx
from mcp.types import Tool

import litellm
from litellm import Router
from litellm.caching.caching import DualCache
from litellm.proxy import proxy_server
from litellm.proxy._types import LiteLLM_ObjectPermissionTable, UserAPIKeyAuth
from litellm.proxy.mcp_server import mcp_server_manager, server, tool_registry
from litellm.proxy.mcp_server.faults.list_outcomes import AggregateToolListing
from litellm.proxy.utils import ProxyLogging
from litellm.types.mcp_server.mcp_server_manager import MCPServer

LEDGER_TOOL: Final = "ledger-commit_write"
LEDGER_MODEL: Final = "ledger-model"
LEDGER_TOOLS: Final = ({"type": "mcp", "server_url": "litellm_proxy/ledger", "require_approval": "never"},)
_API_BASE: Final = "https://ledger-upstream.example.com/v1"
_ARGUMENTS: Final = json.dumps({"entry": "alpha"})


class Ledger:
    """The ledger MCP tool's server side: counts every call the gateway hands it."""

    def __init__(self, raises: Exception | None = None) -> None:
        self.calls: int = 0
        self._raises: Final = raises

    async def commit_write(self, entry: str) -> Mapping[str, str]:
        self.calls += 1
        if self._raises is not None:
            raise self._raises
        return {"written": entry}


def serve_ledger(monkeypatch: pytest.MonkeyPatch, ledger: Ledger) -> UserAPIKeyAuth:
    """Serve ``ledger`` as an auto-approved tool through the real MCP manager and tool registry."""
    manager: Final = mcp_server_manager.MCPServerManager()
    manager.registry = {
        "ledger": MCPServer(
            server_id="ledger",
            name="ledger",
            server_name="ledger",
            transport="http",
            url="https://ledger.example.com/mcp",
            spec_path="ledger.json",
            auth_type="none",
        )
    }
    manager.tool_name_to_mcp_server_name_mapping = {LEDGER_TOOL: "ledger"}
    registry: Final = tool_registry.MCPToolRegistry()
    registry.register_tool(LEDGER_TOOL, "Append one ledger entry", {"type": "object"}, ledger.commit_write)

    async def listing(**_: object) -> AggregateToolListing:
        return AggregateToolListing(tools=[Tool(name=LEDGER_TOOL, inputSchema={"type": "object"})], outcomes={})

    monkeypatch.setattr(tool_registry, "global_mcp_tool_registry", registry)
    monkeypatch.setattr(mcp_server_manager, "global_mcp_server_manager", manager)
    monkeypatch.setattr(proxy_server, "proxy_logging_obj", ProxyLogging(user_api_key_cache=DualCache()))
    monkeypatch.setattr(server, "_get_tools_from_mcp_servers", listing)
    return UserAPIKeyAuth(
        object_permission=LiteLLM_ObjectPermissionTable(object_permission_id="ledger-test", mcp_servers=["ledger"])
    )


class ModelUpstream:
    """An OpenAI-compatible upstream that asks for the ledger tool, then answers the continuation."""

    def __init__(self, tool_calls: int, continuation_fails: bool) -> None:
        self.tool_calls: Final = tool_calls
        self.continuation_fails: Final = continuation_fails
        self.initial_calls: int = 0
        self.continuation_calls: int = 0

    def chat(self, request: httpx.Request) -> httpx.Response:
        body: Final = json.loads(request.content)
        if not any(message.get("role") == "tool" for message in body["messages"]):
            self.initial_calls += 1
            return _chat(_chat_tool_calls(self.tool_calls), "tool_calls", stream=body.get("stream") is True)
        self.continuation_calls += 1
        if self.continuation_fails:
            return _server_error()
        return _chat({"role": "assistant", "content": "written"}, "stop", stream=body.get("stream") is True)

    def responses(self, request: httpx.Request) -> httpx.Response:
        body: Final = json.loads(request.content)
        items: Final = body["input"] if isinstance(body.get("input"), list) else []
        if not any(item.get("type") == "function_call_output" for item in items):
            self.initial_calls += 1
            return _response(tuple(_responses_tool_call(index) for index in range(self.tool_calls)))
        self.continuation_calls += 1
        if self.continuation_fails:
            return _server_error()
        return _response(
            (
                {
                    "type": "message",
                    "id": "msg_1",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": "written", "annotations": []}],
                },
            )
        )


def ledger_router(respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch, upstream: ModelUpstream) -> Router:
    """A router that retries the ledger model group twice and falls back to another group on the same upstream."""
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setenv("DISABLE_AIOHTTP_TRANSPORT", "True")
    litellm.in_memory_llm_clients_cache.flush_cache()
    respx_mock.post(f"{_API_BASE}/chat/completions").mock(side_effect=upstream.chat)
    respx_mock.post(f"{_API_BASE}/responses").mock(side_effect=upstream.responses)
    return Router(
        model_list=[
            _deployment(LEDGER_MODEL, "sk-ledger-a"),
            _deployment(LEDGER_MODEL, "sk-ledger-b"),
            _deployment("ledger-backup", "sk-ledger-c"),
        ],
        num_retries=2,
        fallbacks=[{LEDGER_MODEL: ["ledger-backup"]}],
        disable_cooldowns=True,
    )


def _deployment(model_name: str, api_key: str) -> Mapping[str, object]:
    return {
        "model_name": model_name,
        "litellm_params": {"model": "openai/gpt-4.1-mini", "api_key": api_key, "api_base": _API_BASE},
    }


def _server_error() -> httpx.Response:
    return httpx.Response(500, json={"error": {"message": "continuation failed", "type": "server_error"}})


def _chat_tool_calls(count: int) -> Mapping[str, object]:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"id": f"call_{index}", "type": "function", "function": {"name": LEDGER_TOOL, "arguments": _ARGUMENTS}}
            for index in range(count)
        ],
    }


def _chat(message: Mapping[str, object], finish_reason: str, stream: bool) -> httpx.Response:
    head: Final = {"id": "chatcmpl-ledger", "created": 1, "model": "gpt-4.1-mini"}
    usage: Final = {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
    if not stream:
        return httpx.Response(
            200,
            json={
                **head,
                "object": "chat.completion",
                "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
                "usage": usage,
            },
        )
    calls: Final = message.get("tool_calls")
    delta: Final = (
        {**message, "tool_calls": [{**call, "index": index} for index, call in enumerate(calls)]}
        if isinstance(calls, list)
        else message
    )
    events: Final = (
        {
            **head,
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        },
        {**head, "object": "chat.completion.chunk", "choices": [], "usage": usage},
    )
    return httpx.Response(
        200,
        content=b"".join(f"data: {json.dumps(event)}\n\n".encode() for event in events) + b"data: [DONE]\n\n",
        headers={"content-type": "text/event-stream"},
    )


def _responses_tool_call(index: int) -> Mapping[str, object]:
    return {
        "type": "function_call",
        "id": f"fc_{index}",
        "call_id": f"call_{index}",
        "name": LEDGER_TOOL,
        "arguments": _ARGUMENTS,
        "status": "completed",
    }


def _response(output: Sequence[Mapping[str, object]]) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "resp_ledger",
            "object": "response",
            "created_at": 1,
            "status": "completed",
            "model": "gpt-4.1-mini",
            "output": list(output),
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
            "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        },
    )
