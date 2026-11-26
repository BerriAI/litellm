import asyncio
import json
import threading
from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import import_module
from types import MappingProxyType
from typing import Final, cast
from unittest.mock import AsyncMock, patch

import httpx
import openai
import pytest
import respx
from fastapi import Request, Response
from fastapi.responses import StreamingResponse
from mcp.types import Tool as MCPTool

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.llms.anthropic.messages.adapters.transformation import (
    LiteLLMAnthropicMessagesAdapter,
)
from litellm.llms.anthropic.messages.native.handler import (
    anthropic_messages_handler,
)
from litellm.llms.anthropic.messages.native.mcp_handler import (
    MAX_MCP_TOOL_USE_ITERATIONS,
    _build_tool_result_message,
    _extract_tool_use_blocks,
)
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.spend_tracking.spend_tracking_utils import get_logging_payload
from litellm.proxy.utils import ProxyLogging
from litellm.responses.mcp.litellm_proxy_mcp_handler import LiteLLM_Proxy_MCP_Handler
from litellm.proxy._types import SpendLogsPayload
from litellm.types.utils import StandardLoggingPayload

MCP_REFERENCE = {
    "type": "mcp",
    "server_label": "litellm",
    "server_url": "litellm_proxy/mcp/deepwiki",
    "require_approval": "never",
}


def test_anthropic_messages_handler_routes_litellm_proxy_mcp_to_the_gateway():
    """
    Regression test (LIT-4517): /v1/messages must expand a litellm_proxy MCP
    reference through the MCP gateway.

    Given: A /v1/messages request whose tools carry a litellm_proxy MCP reference
    When:  The handler dispatches
    Then:  It hands off to the MCP gateway instead of the provider

    Without this hook the reference is forwarded to Anthropic verbatim and the API
    rejects the request ("Input tag 'mcp' found using 'type' does not match any of
    the expected tags"), because only /v1/chat/completions and /v1/responses ever
    had a gateway entry point. This pins the wiring, not the helper: deleting the
    dispatch makes the whole feature unreachable while every unit test still passes.
    """
    with patch(
        "litellm.llms.anthropic.messages.native.mcp_handler.anthropic_messages_with_mcp",
        new=AsyncMock(return_value={"routed": True}),
    ) as routed:
        result = anthropic_messages_handler(
            max_tokens=100,
            messages=[{"role": "user", "content": "hi"}],
            model="claude-sonnet-4-5",
            tools=[MCP_REFERENCE],
            custom_llm_provider="anthropic",
        )

    assert routed.called, "A litellm_proxy MCP reference must be dispatched to the MCP gateway"
    assert routed.call_args.kwargs["tools"] == [MCP_REFERENCE]
    assert routed.call_args.kwargs["model"] == "claude-sonnet-4-5"
    assert result is not None


@pytest.mark.asyncio
async def test_anthropic_messages_with_mcp_reaches_an_env_only_deployment_with_its_env_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setenv("AZURE_AI_API_BASE", "https://example-resource.cognitiveservices.azure.com")
    monkeypatch.setenv("AZURE_AI_API_KEY", "azure-env-key")
    litellm.in_memory_llm_clients_cache.flush_cache()

    with respx.mock(assert_all_called=True) as respx_mock:
        upstream: Final = respx_mock.route().mock(side_effect=httpx.ConnectError("cut at the wire"))
        with pytest.raises(openai.APIError):
            await litellm.anthropic.messages.acreate(
                max_tokens=100,
                messages=[{"role": "user", "content": "ping"}],
                model="azure_ai/gpt-6-luna",
                tools=[{**MCP_REFERENCE, "server_url": "litellm_proxy"}],
            )

    assert {call.request.url.host for call in upstream.calls} == {"example-resource.cognitiveservices.azure.com"}
    assert {call.request.headers.get("api-key") for call in upstream.calls} == {"azure-env-key"}
    assert not any(b"litellm_proxy" in call.request.content for call in upstream.calls)


def test_anthropic_messages_handler_skips_the_gateway_on_recursion():
    """The gateway's own follow-up call must not re-enter the gateway."""
    with patch(
        "litellm.llms.anthropic.messages.native.mcp_handler.anthropic_messages_with_mcp",
        new=AsyncMock(return_value={"routed": True}),
    ) as routed:
        with pytest.raises(ValueError, match="anthropic_messages_handler is not implemented for sync calls"):
            anthropic_messages_handler(
                max_tokens=100,
                messages=[{"role": "user", "content": "hi"}],
                model="claude-sonnet-4-5",
                tools=[MCP_REFERENCE],
                custom_llm_provider="anthropic",
                _skip_mcp_handler=True,
            )

    assert not routed.called, "_skip_mcp_handler must stop the gateway from recursing"


def test_anthropic_messages_handler_leaves_native_tools_alone():
    """A plain Anthropic tool is not an MCP reference and must not reach the gateway."""
    with patch(
        "litellm.llms.anthropic.messages.native.mcp_handler.anthropic_messages_with_mcp",
        new=AsyncMock(return_value={"routed": True}),
    ) as routed:
        with pytest.raises(ValueError, match="anthropic_messages_handler is not implemented for sync calls"):
            anthropic_messages_handler(
                max_tokens=100,
                messages=[{"role": "user", "content": "hi"}],
                model="claude-sonnet-4-5",
                tools=[{"name": "get_weather", "input_schema": {"type": "object"}}],
                custom_llm_provider="anthropic",
            )

    assert not routed.called, "Only litellm_proxy MCP references belong to the gateway"


def test_extract_tool_use_blocks_ignores_text_blocks():
    """Only tool_use blocks drive the loop; text blocks are the model's prose."""
    response = {
        "content": [
            {"type": "text", "text": "let me look that up"},
            {"type": "tool_use", "id": "toolu_1", "name": "read_wiki_structure", "input": {"repoName": "a/b"}},
        ]
    }

    blocks = _extract_tool_use_blocks(response)

    assert len(blocks) == 1
    assert blocks[0]["name"] == "read_wiki_structure"


def test_build_tool_result_message_uses_anthropic_tool_result_blocks():
    """
    Results must go back as tool_result blocks in a user message.

    Anthropic pairs each result to its request by tool_use_id; the OpenAI shape
    (a role="tool" message keyed by tool_call_id) is rejected here.
    """
    message = _build_tool_result_message([{"tool_call_id": "toolu_1", "result": "9 sections", "name": "read_wiki"}])

    assert message["role"] == "user"
    assert message["content"] == [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "9 sections"}]


def test_build_tool_result_message_survives_the_chat_completions_bridge():
    """
    Regression test (LIT-8474): a non-Anthropic model behind /v1/messages must see
    the executed tool result as a role="tool" message keyed by the tool_call_id.

    The bridge only translates list content, so a tuple-shaped user message was
    dropped and the model re-requested the tool until the iteration cap.
    """
    message = _build_tool_result_message(
        [
            {"tool_call_id": "call_1", "result": "5", "name": "add"},
            {"tool_call_id": "call_2", "result": "7", "name": "add"},
        ]
    )

    translated = LiteLLMAnthropicMessagesAdapter().translate_anthropic_messages_to_openai(
        [message], model="hosted_vllm/gpt-4o-mini", custom_llm_provider="hosted_vllm"
    )

    assert translated == [
        {"role": "tool", "tool_call_id": "call_1", "content": "5"},
        {"role": "tool", "tool_call_id": "call_2", "content": "7"},
    ]


@pytest.mark.asyncio
async def test_anthropic_messages_with_mcp_forwards_the_callers_mcp_credentials():
    """
    Regression test (LIT-4517): the caller's MCP auth must reach both tool listing
    and tool execution on /v1/messages.

    Given: A request carrying MCP auth headers and request tags
    When:  The gateway lists and then executes an MCP tool
    Then:  Both calls receive the caller's credentials, tags and trace ids

    Dropping them does not fail loudly; the tool still executes, just with no
    credentials, so every auth-requiring MCP server (interactive OAuth, bearer
    token, per-user env) silently returns nothing while the model claims it has
    no access. Only a no-auth server would look healthy.
    """
    from litellm.llms.anthropic.messages.native import mcp_handler
    from litellm.responses.mcp.request_context import MCPRequestContext

    context = MCPRequestContext(
        user_api_key_auth="auth-object",
        mcp_auth_header="legacy-header",
        mcp_server_auth_headers={"deepwiki": {"authorization": "Bearer per-server"}},
        oauth2_headers={"authorization": "Bearer oauth"},
        raw_headers={"x-trace": "abc"},
        request_tags=["team-a"],
        litellm_trace_id="trace-123",
        litellm_call_id="call-456",
        guardrail_context={"metadata": {"guardrails": ("block-all",)}},
    )

    process = AsyncMock(return_value=([], {}))
    execute = AsyncMock(return_value=[{"tool_call_id": "toolu_1", "result": "ok", "name": "t"}])
    responses = [
        {"stop_reason": "tool_use", "content": [{"type": "tool_use", "id": "toolu_1", "name": "t", "input": {}}]},
        {"stop_reason": "end_turn", "content": [{"type": "text", "text": "done"}]},
    ]

    with (
        patch.object(MCPRequestContext, "resolve", return_value=context),
        patch.object(
            mcp_handler.LiteLLM_Proxy_MCP_Handler
            if hasattr(mcp_handler, "LiteLLM_Proxy_MCP_Handler")
            else __import__(
                "litellm.responses.mcp.litellm_proxy_mcp_handler", fromlist=["LiteLLM_Proxy_MCP_Handler"]
            ).LiteLLM_Proxy_MCP_Handler,
            "process_mcp_tools_without_openai_transform",
            new=process,
        ),
        patch.object(
            import_module("litellm.responses.mcp.litellm_proxy_mcp_handler").LiteLLM_Proxy_MCP_Handler,
            "execute_tool_calls",
            new=execute,
        ),
        patch("litellm.anthropic_messages", new=AsyncMock(side_effect=responses)),
    ):
        await mcp_handler.anthropic_messages_with_mcp(
            max_tokens=100,
            messages=[{"role": "user", "content": "hi"}],
            model="claude-sonnet-4-5",
            tools=[MCP_REFERENCE],
        )

    listing = process.call_args.kwargs
    assert listing["mcp_auth_header"] == "legacy-header", "tool listing must use the caller's MCP auth"
    assert listing["mcp_server_auth_headers"] == {"deepwiki": {"authorization": "Bearer per-server"}}
    assert listing["request_tags"] == ["team-a"]
    assert listing["litellm_trace_id"] == "trace-123"

    execution = execute.call_args.kwargs
    assert execution["user_api_key_auth"] == "auth-object"
    assert execution["mcp_auth_header"] == "legacy-header", "tool execution must use the caller's MCP auth"
    assert execution["mcp_server_auth_headers"] == {"deepwiki": {"authorization": "Bearer per-server"}}
    assert execution["oauth2_headers"] == {"authorization": "Bearer oauth"}
    assert execution["raw_headers"] == {"x-trace": "abc"}
    assert execution["litellm_call_id"] == "call-456"
    assert execution["litellm_trace_id"] == "trace-123"
    assert execution["request_tags"] == ["team-a"]

    assert execution["guardrail_context"] == {"metadata": {"guardrails": ("block-all",)}}


@pytest.mark.asyncio
async def test_anthropic_messages_with_mcp_hands_execution_the_requests_served_tools():
    """
    Regression test: /v1/messages auto-execution must carry this request's
    resolved tool definitions into execution, matching the Responses and chat
    completions bridges.

    Given: A request whose MCP reference resolves to a definition carrying a
           description and input schema
    When:  The model asks for that tool and the gateway executes it
    Then:  _execute_tool_calls receives the definition under served_tools, so
           pre_mcp_call hooks can judge the call on what the model was shown

    Dropping it does not fail loudly; the call still runs, but the hook sees
    only the name and arguments, leaving /v1/messages permanently colder than
    the other two bridges even though all three resolve the same definitions.
    """
    from mcp.types import Tool

    from litellm.llms.anthropic.messages.native import mcp_handler
    from litellm.responses.mcp.request_context import MCPRequestContext

    served = [
        Tool(
            name="read_wiki_structure",
            description="Read the structure of a wiki",
            inputSchema={"type": "object", "properties": {"repoName": {"type": "string"}}},
        )
    ]

    process = AsyncMock(return_value=(served, {"read_wiki_structure": "deepwiki"}))
    execute = AsyncMock(
        return_value=[{"tool_call_id": "toolu_1", "result": "ok", "name": "read_wiki_structure"}]
    )
    responses = [
        {
            "stop_reason": "tool_use",
            "content": [{"type": "tool_use", "id": "toolu_1", "name": "read_wiki_structure", "input": {}}],
        },
        {"stop_reason": "end_turn", "content": [{"type": "text", "text": "done"}]},
    ]

    with (
        patch.object(MCPRequestContext, "resolve", return_value=MCPRequestContext(user_api_key_auth="auth")),
        patch.object(
            import_module("litellm.responses.mcp.litellm_proxy_mcp_handler").LiteLLM_Proxy_MCP_Handler,
            "process_mcp_tools_without_openai_transform",
            new=process,
        ),
        patch.object(
            import_module("litellm.responses.mcp.litellm_proxy_mcp_handler").LiteLLM_Proxy_MCP_Handler,
            "execute_tool_calls",
            new=execute,
        ),
        patch("litellm.anthropic_messages", new=AsyncMock(side_effect=responses)),
    ):
        await mcp_handler.anthropic_messages_with_mcp(
            max_tokens=100,
            messages=[{"role": "user", "content": "hi"}],
            model="claude-sonnet-4-5",
            tools=[MCP_REFERENCE],
        )

    execution = execute.call_args.kwargs
    assert execution.get("served_tools") == served, (
        "The request's resolved tool definitions must reach execution so pre_mcp_call "
        "hooks see the listed description and input schema"
    )


@pytest.mark.asyncio
async def test_anthropic_messages_with_mcp_stops_when_every_tool_call_is_skipped():
    """
    Regression test (LIT-4517): a tool_use turn whose calls all get skipped must
    end the loop, not send an empty tool_result message.

    Given: The model asks for a tool but the executor skips it (unresolvable name)
    When:  The gateway loop handles the empty result set
    Then:  It returns the last response instead of calling the model again

    _build_tool_result_message([]) produces a user message with empty content, and
    Anthropic rejects that, so the caller would get an unhandled 400 from the middle
    of the loop rather than the model's own answer.
    """
    from litellm.llms.anthropic.messages.native import mcp_handler
    from litellm.responses.mcp.request_context import MCPRequestContext

    tool_use_response = {
        "stop_reason": "tool_use",
        "content": [{"type": "tool_use", "id": "toolu_1", "name": "gone", "input": {}}],
    }
    anthropic_messages_mock = AsyncMock(return_value=tool_use_response)

    with (
        patch.object(MCPRequestContext, "resolve", return_value=MCPRequestContext(user_api_key_auth="auth")),
        patch.object(
            import_module("litellm.responses.mcp.litellm_proxy_mcp_handler").LiteLLM_Proxy_MCP_Handler,
            "process_mcp_tools_without_openai_transform",
            new=AsyncMock(return_value=([], {})),
        ),
        patch.object(
            import_module("litellm.responses.mcp.litellm_proxy_mcp_handler").LiteLLM_Proxy_MCP_Handler,
            "execute_tool_calls",
            new=AsyncMock(return_value=[]),
        ),
        patch("litellm.anthropic_messages", new=anthropic_messages_mock),
    ):
        result = await mcp_handler.anthropic_messages_with_mcp(
            max_tokens=100,
            messages=[{"role": "user", "content": "hi"}],
            model="claude-sonnet-4-5",
            tools=[MCP_REFERENCE],
        )

    assert anthropic_messages_mock.await_count == 1, (
        "With no tool results there is nothing to send back, so the loop must not call the model again"
    )
    assert result == tool_use_response


@pytest.mark.asyncio
async def test_anthropic_messages_with_mcp_classifies_only_the_response_the_client_gets():
    from litellm.llms.anthropic.messages.native import mcp_handler
    from litellm.responses.mcp.request_context import MCPRequestContext
    from litellm.router import Router

    classifier_router = Router(
        model_list=[
            {
                "model_name": "classifier",
                "litellm_params": {
                    "model": "openai/gpt-5.4-mini",
                    "api_key": "sk-fake",
                    "mock_response": '{"verdicts": {"toolu_bash": {"flagged": true, "explanation": "unrequested"}}}',
                },
            }
        ]
    )
    responses = [
        {"stop_reason": "tool_use", "content": [{"type": "tool_use", "id": "toolu_mcp", "name": "t", "input": {}}]},
        {
            "stop_reason": "tool_use",
            "content": [{"type": "tool_use", "id": "toolu_bash", "name": "Bash", "input": {"command": "rm -rf /"}}],
        },
    ]
    backend = AsyncMock(side_effect=responses)

    with (
        patch.object(MCPRequestContext, "resolve", return_value=MCPRequestContext(user_api_key_auth=None)),
        patch.object(
            import_module("litellm.responses.mcp.litellm_proxy_mcp_handler").LiteLLM_Proxy_MCP_Handler,
            "process_mcp_tools_without_openai_transform",
            new=AsyncMock(return_value=([], {})),
        ),
        patch.object(
            import_module("litellm.responses.mcp.litellm_proxy_mcp_handler").LiteLLM_Proxy_MCP_Handler,
            "execute_tool_calls",
            new=AsyncMock(side_effect=[[{"tool_call_id": "toolu_mcp", "result": "ok", "name": "t"}], []]),
        ),
        patch("litellm.anthropic_messages", new=backend),
        patch("litellm.proxy.proxy_server.general_settings", {"safeguards_classifier_model": "classifier"}),
    ):
        result = await mcp_handler.anthropic_messages_with_mcp(
            max_tokens=100,
            messages=[{"role": "user", "content": "hi"}],
            model="fireworks_ai/kimi",
            tools=[MCP_REFERENCE],
            safeguards=[{"type": "dangerous_tool_use", "classifier_context": {"permission_mode": "auto"}}],
            litellm_router=classifier_router,
        )

    assert backend.await_count == 2
    assert all("safeguards" not in call.kwargs for call in backend.await_args_list)
    assert result["safeguard_results"][0]["status"]["tool_uses"] == {
        "toolu_bash": {"type": "evaluated", "outcome": "flagged", "explanation": "unrequested"}
    }


async def _mcp_safeguards_turn_recording_beta_map_loads(custom_llm_provider: str, model: str) -> tuple[int, ...]:
    import threading

    from litellm import anthropic_beta_headers_manager
    from litellm.llms.anthropic.messages.native import mcp_handler
    from litellm.responses.mcp.request_context import MCPRequestContext

    loader_threads: Final[list[int]] = []  # mutable-ok: records the thread each beta map load ran on

    def recording_loader(url: str) -> object:
        loader_threads.append(threading.get_ident())
        return anthropic_beta_headers_manager.GetAnthropicBetaHeadersConfig.load_local_beta_headers_config()

    with (
        patch.object(anthropic_beta_headers_manager, "_BETA_HEADERS_CONFIG", None),
        patch.object(anthropic_beta_headers_manager, "get_beta_headers_config", new=recording_loader),
        patch.object(MCPRequestContext, "resolve", return_value=MCPRequestContext(user_api_key_auth=None)),
        patch.object(
            import_module("litellm.responses.mcp.litellm_proxy_mcp_handler").LiteLLM_Proxy_MCP_Handler,
            "process_mcp_tools_without_openai_transform",
            new=AsyncMock(return_value=([], {})),
        ),
        patch("litellm.anthropic_messages", new=AsyncMock(return_value={"stop_reason": "end_turn", "content": []})),
        patch("litellm.proxy.proxy_server.general_settings", {"safeguards_classifier_model": "classifier"}),
    ):
        await mcp_handler.anthropic_messages_with_mcp(
            max_tokens=100,
            messages=[{"role": "user", "content": "hi"}],
            model=model,
            custom_llm_provider=custom_llm_provider,
            tools=[MCP_REFERENCE],
            safeguards=[{"type": "dangerous_tool_use", "classifier_context": {"permission_mode": "auto"}}],
        )
    return tuple(loader_threads)


@pytest.mark.asyncio
async def test_anthropic_messages_with_mcp_loads_a_filtered_routes_beta_map_off_the_event_loop():
    import threading

    loader_threads: Final = await _mcp_safeguards_turn_recording_beta_map_loads("azure_ai", "claude-sonnet-4-6")

    assert len(loader_threads) == 1
    assert threading.get_ident() not in loader_threads


@pytest.mark.asyncio
async def test_anthropic_messages_with_mcp_never_loads_the_beta_map_for_first_party_anthropic():
    assert await _mcp_safeguards_turn_recording_beta_map_loads("anthropic", "claude-sonnet-4-6") == ()


_ROUNDS_HOST: Final = "mcp-messages-rounds.example.com"
_PLAIN_HOST: Final = "plain-messages-rounds.example.com"
_ROUNDS_MODEL: Final = "anthropic/claude-haiku-5-5"
_ADD_REFERENCE: Final = MappingProxyType(
    {"type": "mcp", "server_label": "math", "server_url": "litellm_proxy/mcp/math", "require_approval": "never"}
)
_ASK: Final = MappingProxyType({"role": "user", "content": "Add 2 and 3, then write a story about the sum"})
_ADDED: Final = ({"tool_call_id": "toolu_add", "result": "5", "name": "add"},)
_NEVER_LOGGED_SECONDS: Final = 60
_SAFEGUARDED_MODEL: Final = "coralbricks/glm-5.3"
_BRIDGED_MODEL: Final = "hosted_vllm/Qwen/Qwen3-8B"
_API_PATH: Final = MappingProxyType({_SAFEGUARDED_MODEL: "/v1", _BRIDGED_MODEL: "/v1"})


def _messages_round(stop_reason: str, input_tokens: int, output_tokens: int, **usage: int) -> Mapping[str, object]:
    asks_for_the_tool: Final = stop_reason == "tool_use"
    return MappingProxyType(
        {
            "id": f"msg_{stop_reason}_{input_tokens}",
            "type": "message",
            "role": "assistant",
            "model": "claude-haiku-5-5",
            "content": [
                {"type": "tool_use", "id": "toolu_add", "name": "add", "input": {"a": 2, "b": 3}}
                if asks_for_the_tool
                else {"type": "text", "text": "Five apples sat in a row"}
            ],
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens, **usage},
        }
    )


_ASKS_FOR_ADD: Final = _messages_round("tool_use", 418, 68)
_ASKS_AGAIN: Final = _messages_round("tool_use", 466, 71)
_ANSWERS: Final = _messages_round("end_turn", 492, 432)


def _ok(body: Mapping[str, object]) -> httpx.Response:
    return httpx.Response(200, content=json.dumps(body, default=dict))


class _RoundRecorder(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self.logged: tuple[StandardLoggingPayload, ...] = ()
        self.spend_rows: tuple[SpendLogsPayload, ...] = ()
        self.failed: tuple[StandardLoggingPayload, ...] = ()
        self._recorded: Final = asyncio.Condition()

    async def async_log_success_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: datetime, end_time: datetime
    ) -> None:
        self.spend_rows = (*self.spend_rows, get_logging_payload(dict(kwargs), response_obj, start_time, end_time))
        self.logged = (*self.logged, cast(StandardLoggingPayload, kwargs["standard_logging_object"]))
        await self._announce()

    async def async_log_failure_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: datetime, end_time: datetime
    ) -> None:
        self.failed = (*self.failed, cast(StandardLoggingPayload, kwargs["standard_logging_object"]))
        await self._announce()

    async def _announce(self) -> None:
        async with self._recorded:
            self._recorded.notify_all()

    async def settled(self, logged: int = 0, failed: int = 0) -> None:
        async with self._recorded:
            await asyncio.wait_for(
                self._recorded.wait_for(lambda: len(self.logged) >= logged and len(self.failed) >= failed),
                timeout=_NEVER_LOGGED_SECONDS,
            )
        await _settled()


async def _settled() -> None:
    await asyncio.sleep(0)
    await asyncio.wait_for(GLOBAL_LOGGING_WORKER.flush(), timeout=10)
    await asyncio.sleep(0)


def _rounds_logging_obj(recorder: _RoundRecorder, call_id: str, stream: bool) -> Logging:
    return Logging(
        model=_ROUNDS_MODEL,
        messages=[dict(_ASK)],
        stream=stream,
        call_type="anthropic_messages",
        start_time=datetime(2026, 1, 1),
        litellm_call_id=call_id,
        function_id=call_id,
        dynamic_async_success_callbacks=[recorder],
        dynamic_async_failure_callbacks=[recorder],
    )


@contextmanager
def _add_tool_served(execute: AsyncMock) -> Iterator[None]:
    add_tool: Final = MCPTool.model_validate({"name": "add", "inputSchema": {"type": "object"}}, by_name=False)
    with (
        patch.object(
            LiteLLM_Proxy_MCP_Handler,
            "process_mcp_tools_without_openai_transform",
            AsyncMock(return_value=([add_tool], {"add": "math"})),
        ),
        patch.object(LiteLLM_Proxy_MCP_Handler, "execute_tool_calls", execute),
    ):
        yield


def _serve_rounds(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch, host: str, rounds: Sequence[httpx.Response]
) -> respx.Route:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setenv("DISABLE_AIOHTTP_TRANSPORT", "True")
    litellm.in_memory_llm_clients_cache.flush_cache()
    return respx_mock.route(host=host).mock(side_effect=list(rounds))


@dataclass(frozen=True, slots=True)
class _Served:
    body: Mapping[str, object] | None
    stream_events: Sequence[Mapping[str, object]]

    def usage(self) -> Mapping[str, object]:
        if self.body is not None:
            return cast(Mapping[str, object], self.body["usage"])
        delta: Final = next(event for event in self.stream_events if event["type"] == "message_delta")
        return cast(Mapping[str, object], delta["usage"])

    def started_usage(self) -> Mapping[str, object]:
        start: Final = next(event for event in self.stream_events if event["type"] == "message_start")
        return cast(Mapping[str, object], cast(Mapping[str, object], start["message"])["usage"])


def _sse_events(chunks: Sequence[bytes]) -> Sequence[Mapping[str, object]]:
    lines: Final = (line for chunk in chunks for line in chunk.decode().splitlines())
    return tuple(json.loads(line.removeprefix("data: ")) for line in lines if line.startswith("data: "))


async def _served(result: object) -> _Served:
    if isinstance(result, dict):
        return _Served(cast(Mapping[str, object], result), ())
    chunks: Final = [chunk async for chunk in cast(AsyncIterator[bytes], result)]
    return _Served(None, _sse_events(chunks))


async def _auto_execute(
    respx_mock: respx.MockRouter,
    monkeypatch: pytest.MonkeyPatch,
    rounds: Sequence[httpx.Response],
    *,
    stream: bool,
    call_id: str,
    execute: AsyncMock | None = None,
    tools: Sequence[Mapping[str, object]] = (_ADD_REFERENCE,),
    model: str = _ROUNDS_MODEL,
    extra: Mapping[str, object] = MappingProxyType({}),
) -> tuple[_RoundRecorder, _Served | Exception]:
    _serve_rounds(respx_mock, monkeypatch, _ROUNDS_HOST, rounds)
    recorder: Final = _RoundRecorder()
    with _add_tool_served(execute or AsyncMock(return_value=list(_ADDED))):
        try:
            result: Final = await litellm.anthropic_messages(
                model=model,
                max_tokens=1000,
                messages=[dict(_ASK)],
                tools=[dict(tool) for tool in tools],
                api_key="sk-mcp-messages-rounds",
                api_base=f"https://{_ROUNDS_HOST}{_API_PATH.get(model, '')}",
                stream=stream,
                client=AsyncHTTPHandler(),
                litellm_logging_obj=_rounds_logging_obj(recorder, call_id, stream),
                **extra,
            )
        except Exception as error:
            await _settled()
            return recorder, error
        await _settled()
        logged_before_the_client_read_the_stream: Final = recorder.logged
        served: Final = await _served(result)
    await _settled()
    assert logged_before_the_client_read_the_stream == (() if stream else recorder.logged)
    return recorder, served


async def _plain_rows(
    respx_mock: respx.MockRouter,
    monkeypatch: pytest.MonkeyPatch,
    rounds: Sequence[Mapping[str, object]],
    model: str = _ROUNDS_MODEL,
    extra: Mapping[str, object] = MappingProxyType({}),
) -> Sequence[StandardLoggingPayload]:
    _serve_rounds(respx_mock, monkeypatch, _PLAIN_HOST, tuple(_ok(body) for body in rounds))
    recorder: Final = _RoundRecorder()
    for index in range(len(rounds)):
        await litellm.anthropic_messages(
            model=model,
            max_tokens=1000,
            messages=[dict(_ASK)],
            api_key="sk-plain-messages-rounds",
            api_base=f"https://{_PLAIN_HOST}{_API_PATH.get(model, '')}",
            client=AsyncHTTPHandler(),
            litellm_logging_obj=_rounds_logging_obj(recorder, f"plain-round-{index}", stream=False),
            **extra,
        )
        await recorder.settled(logged=index + 1)
    return recorder.logged


def _assert_bills(row: StandardLoggingPayload, plain_rows: Sequence[StandardLoggingPayload]) -> None:
    assert (row["prompt_tokens"], row["completion_tokens"]) == (
        sum(plain["prompt_tokens"] for plain in plain_rows),
        sum(plain["completion_tokens"] for plain in plain_rows),
    )
    assert row["response_cost"] == pytest.approx(sum(plain["response_cost"] for plain in plain_rows))
    assert row["response_cost"] > 0


@pytest.fixture(params=["route", "python_handler"])
def messages_entry(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> str:
    if request.param == "python_handler":
        monkeypatch.setenv("LITELLM_RUST", "false")
    return str(request.param)


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "rounds",
    [
        pytest.param((_ASKS_FOR_ADD, _ANSWERS), id="two-rounds"),
        pytest.param((_ASKS_FOR_ADD, _ASKS_AGAIN, _ANSWERS), id="three-rounds"),
    ],
)
@pytest.mark.asyncio
async def test_auto_execute_bills_every_round_on_one_row_logged_after_the_last_round(
    respx_mock: respx.MockRouter,
    monkeypatch: pytest.MonkeyPatch,
    messages_entry: str,
    stream: bool,
    rounds: Sequence[Mapping[str, object]],
):
    recorder, served = await _auto_execute(
        respx_mock,
        monkeypatch,
        tuple(_ok(body) for body in rounds),
        stream=stream,
        call_id=f"mcp-messages-rounds-{messages_entry}-{len(rounds)}-{stream}",
    )
    plain_rows: Final = await _plain_rows(respx_mock, monkeypatch, rounds)

    assert isinstance(served, _Served)
    assert recorder.failed == ()
    assert len(recorder.logged) == 1
    _assert_bills(recorder.logged[0], plain_rows)
    assert (recorder.spend_rows[0]["prompt_tokens"], recorder.spend_rows[0]["completion_tokens"]) == (
        recorder.logged[0]["prompt_tokens"],
        recorder.logged[0]["completion_tokens"],
    )
    summed: Final = (
        sum(cast(Mapping[str, int], body["usage"])["input_tokens"] for body in rounds),
        sum(cast(Mapping[str, int], body["usage"])["output_tokens"] for body in rounds),
    )
    assert (served.usage()["input_tokens"], served.usage()["output_tokens"]) == summed
    if stream:
        assert served.started_usage()["input_tokens"] == summed[0]


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.asyncio
async def test_auto_execute_without_a_tool_call_bills_its_one_round_like_a_plain_request(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch, messages_entry: str, stream: bool
):
    recorder, served = await _auto_execute(
        respx_mock,
        monkeypatch,
        (_ok(_ANSWERS),),
        stream=stream,
        call_id=f"mcp-messages-one-round-{messages_entry}-{stream}",
    )
    plain_rows: Final = await _plain_rows(respx_mock, monkeypatch, (_ANSWERS,))

    assert isinstance(served, _Served)
    assert len(recorder.logged) == 1
    _assert_bills(recorder.logged[0], plain_rows)
    assert (served.usage()["input_tokens"], served.usage()["output_tokens"]) == (492, 432)


@pytest.mark.asyncio
async def test_auto_execute_bills_every_round_up_to_the_iteration_cap(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
):
    capped_rounds: Final = MAX_MCP_TOOL_USE_ITERATIONS + 1
    recorder, served = await _auto_execute(
        respx_mock,
        monkeypatch,
        tuple(_ok(_ASKS_FOR_ADD) for _ in range(capped_rounds + 1)),
        stream=False,
        call_id="mcp-messages-iteration-cap",
    )
    plain_rows: Final = await _plain_rows(respx_mock, monkeypatch, (_ASKS_FOR_ADD,))

    assert isinstance(served, _Served)
    assert respx_mock.calls.call_count == capped_rounds + 1
    assert len(recorder.logged) == 1
    _assert_bills(recorder.logged[0], tuple(plain_rows) * capped_rounds)
    assert served.usage()["input_tokens"] == 418 * capped_rounds


@pytest.mark.asyncio
async def test_auto_execute_bills_the_round_it_made_when_every_tool_call_is_skipped(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
):
    recorder, served = await _auto_execute(
        respx_mock,
        monkeypatch,
        (_ok(_ASKS_FOR_ADD),),
        stream=True,
        call_id="mcp-messages-every-call-skipped",
        execute=AsyncMock(return_value=[]),
    )
    plain_rows: Final = await _plain_rows(respx_mock, monkeypatch, (_ASKS_FOR_ADD,))

    assert isinstance(served, _Served)
    assert len(recorder.logged) == 1
    _assert_bills(recorder.logged[0], plain_rows)


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.asyncio
async def test_auto_execute_bills_the_paid_round_on_the_failure_row_when_the_tool_raises(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch, stream: bool
):
    recorder, outcome = await _auto_execute(
        respx_mock,
        monkeypatch,
        (_ok(_ASKS_FOR_ADD),),
        stream=stream,
        call_id=f"mcp-messages-tool-raises-{stream}",
        execute=AsyncMock(side_effect=RuntimeError("the math server went away")),
    )
    await recorder.settled(failed=1)
    plain_rows: Final = await _plain_rows(respx_mock, monkeypatch, (_ASKS_FOR_ADD,))

    assert isinstance(outcome, Exception)
    assert recorder.logged == ()
    assert len(recorder.failed) == 1
    _assert_bills(recorder.failed[0], plain_rows)


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.asyncio
async def test_auto_execute_bills_every_paid_round_on_the_failure_row_when_a_follow_up_fails(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch, stream: bool
):
    rejected: Final = httpx.Response(
        400,
        content=json.dumps(
            {"type": "error", "error": {"type": "invalid_request_error", "message": "prompt is too long"}}
        ),
    )
    recorder, outcome = await _auto_execute(
        respx_mock,
        monkeypatch,
        (_ok(_ASKS_FOR_ADD), _ok(_ASKS_AGAIN), rejected),
        stream=stream,
        call_id=f"mcp-messages-follow-up-fails-{stream}",
    )
    await recorder.settled(failed=1)
    plain_rows: Final = await _plain_rows(respx_mock, monkeypatch, (_ASKS_FOR_ADD, _ASKS_AGAIN))

    assert isinstance(outcome, litellm.BadRequestError)
    assert recorder.logged == ()
    assert len(recorder.failed) == 1
    _assert_bills(recorder.failed[0], plain_rows)


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.asyncio
async def test_auto_execute_sums_prompt_caching_usage_across_rounds(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch, stream: bool
):
    writes_the_cache: Final = _messages_round(
        "tool_use", 418, 68, cache_creation_input_tokens=2048, cache_read_input_tokens=0
    )
    reads_the_cache: Final = _messages_round(
        "end_turn", 492, 432, cache_creation_input_tokens=0, cache_read_input_tokens=2048
    )
    recorder, served = await _auto_execute(
        respx_mock,
        monkeypatch,
        (_ok(writes_the_cache), _ok(reads_the_cache)),
        stream=stream,
        call_id=f"mcp-messages-prompt-caching-{stream}",
    )
    plain_rows: Final = await _plain_rows(respx_mock, monkeypatch, (writes_the_cache, reads_the_cache))

    assert isinstance(served, _Served)
    assert len(recorder.logged) == 1
    _assert_bills(recorder.logged[0], plain_rows)
    assert (
        served.usage()["cache_creation_input_tokens"],
        served.usage()["cache_read_input_tokens"],
    ) == (2048, 2048)


@pytest.mark.asyncio
async def test_auto_execute_charges_the_fixed_margin_once_per_request(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
):
    fixed_margin: Final = 0.25
    monkeypatch.setattr(litellm, "cost_margin_config", {"anthropic": {"fixed_amount": fixed_margin}})
    rounds: Final = (_ASKS_FOR_ADD, _ASKS_AGAIN, _ANSWERS)
    recorder, served = await _auto_execute(
        respx_mock,
        monkeypatch,
        tuple(_ok(body) for body in rounds),
        stream=False,
        call_id="mcp-messages-fixed-margin",
    )
    plain_rows: Final = await _plain_rows(respx_mock, monkeypatch, rounds)

    assert isinstance(served, _Served)
    assert all(plain["response_cost"] > fixed_margin for plain in plain_rows)
    assert len(recorder.logged) == 1
    assert recorder.logged[0]["response_cost"] == pytest.approx(
        sum(plain["response_cost"] for plain in plain_rows) - fixed_margin * (len(rounds) - 1)
    )


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.asyncio
async def test_auto_execute_bills_a_round_served_from_the_response_cache_at_zero(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch, stream: bool
):
    from litellm.caching.caching import Cache

    monkeypatch.setattr(litellm, "cache", Cache(type="local"))
    tool_results: Final = AsyncMock(
        side_effect=[list(_ADDED), [{"tool_call_id": "toolu_add", "result": "6", "name": "add"}]]
    )
    first_recorder, _ = await _auto_execute(
        respx_mock,
        monkeypatch,
        (_ok(_ASKS_FOR_ADD), _ok(_ANSWERS)),
        stream=False,
        call_id="mcp-messages-cache-fills",
        execute=tool_results,
    )
    recorder, served = await _auto_execute(
        respx_mock,
        monkeypatch,
        (_ok(_ANSWERS),),
        stream=stream,
        call_id=f"mcp-messages-cache-serves-the-first-round-{stream}",
        execute=tool_results,
        tools=({**_ADD_REFERENCE, "server_label": "math_again"},),
    )
    monkeypatch.setattr(litellm, "cache", None)
    plain_rows: Final = await _plain_rows(respx_mock, monkeypatch, (_ASKS_FOR_ADD, _ANSWERS))

    assert isinstance(served, _Served)
    assert tool_results.await_count == 2
    assert len(first_recorder.logged) == 1
    assert len(recorder.logged) == 1
    row: Final = recorder.logged[0]
    assert (row["prompt_tokens"], row["completion_tokens"]) == (
        sum(plain["prompt_tokens"] for plain in plain_rows),
        sum(plain["completion_tokens"] for plain in plain_rows),
    )
    assert row["response_cost"] == pytest.approx(plain_rows[1]["response_cost"])


@pytest.mark.parametrize(
    "tool",
    [
        pytest.param({**_ADD_REFERENCE, "require_approval": "always"}, id="not-auto-execute"),
        pytest.param(
            {"name": "add", "description": "Add two numbers", "input_schema": {"type": "object"}}, id="no-mcp"
        ),
    ],
)
@pytest.mark.asyncio
async def test_a_request_the_loop_does_not_run_bills_its_one_call_unchanged(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch, tool: Mapping[str, object]
):
    recorder, served = await _auto_execute(
        respx_mock, monkeypatch, (_ok(_ASKS_FOR_ADD),), stream=False, call_id="mcp-messages-no-loop", tools=(tool,)
    )
    plain_rows: Final = await _plain_rows(respx_mock, monkeypatch, (_ASKS_FOR_ADD,))

    assert isinstance(served, _Served)
    assert served.body is not None
    assert served.body["stop_reason"] == "tool_use"
    assert len(recorder.logged) == 1
    _assert_bills(recorder.logged[0], plain_rows)


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.asyncio
async def test_auto_execute_bills_every_round_when_the_proxy_classifies_the_final_tool_uses(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch, stream: bool
):
    from litellm.router import Router

    classifier_router: Final = Router(
        model_list=[
            {
                "model_name": "classifier",
                "litellm_params": {"model": "openai/gpt-5.4-mini", "api_key": "sk-fake", "mock_response": "{}"},
            }
        ]
    )
    with patch("litellm.proxy.proxy_server.general_settings", {"safeguards_classifier_model": "classifier"}):
        recorder, served = await _auto_execute(
            respx_mock,
            monkeypatch,
            (_ok(_ASKS_FOR_ADD), _ok(_ANSWERS)),
            stream=stream,
            call_id=f"mcp-messages-safeguards-{stream}",
            model=_SAFEGUARDED_MODEL,
            extra=MappingProxyType(
                {
                    "safeguards": [{"type": "dangerous_tool_use", "classifier_context": {"permission_mode": "auto"}}],
                    "litellm_router": classifier_router,
                }
            ),
        )
    plain_rows: Final = await _plain_rows(respx_mock, monkeypatch, (_ASKS_FOR_ADD, _ANSWERS), _SAFEGUARDED_MODEL)

    assert isinstance(served, _Served)
    assert len(recorder.logged) == 1
    _assert_bills(recorder.logged[0], plain_rows)
    assert (served.usage()["input_tokens"], served.usage()["output_tokens"]) == (418 + 492, 68 + 432)
    if not stream:
        assert served.body is not None
        assert "safeguard_results" in served.body


@contextmanager
def _upstream_answering(rounds: Sequence[Mapping[str, object]]) -> Iterator[str]:
    bodies: Final = iter(tuple(json.dumps(body, default=dict).encode() for body in rounds))

    class _Upstream(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            _ = self.rfile.read(int(self.headers["Content-Length"]))
            body: Final = next(bodies)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            _ = self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            pass

    server: Final = ThreadingHTTPServer(("127.0.0.1", 0), _Upstream)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()


async def _proxy_cost_headers(rounds: Sequence[Mapping[str, object]], stream: bool) -> Mapping[str, str]:
    from litellm.proxy.proxy_server import ProxyConfig
    from litellm.router import Router

    fastapi_response: Final = Response()
    with _upstream_answering(rounds) as api_base, _add_tool_served(AsyncMock(return_value=list(_ADDED))):
        result: Final = await ProxyBaseLLMRequestProcessing(
            data={
                "model": "claude-haiku-5-5",
                "max_tokens": 1000,
                "messages": [dict(_ASK)],
                "tools": [dict(_ADD_REFERENCE)],
                "stream": stream,
            }
        ).base_process_llm_request(
            request=Request(
                {
                    "type": "http",
                    "method": "POST",
                    "path": "/v1/messages",
                    "query_string": b"",
                    "headers": [(b"content-type", b"application/json")],
                }
            ),
            fastapi_response=fastapi_response,
            user_api_key_dict=UserAPIKeyAuth(api_key="sk-test", user_id="user-a"),
            route_type="anthropic_messages",
            proxy_logging_obj=ProxyLogging(user_api_key_cache=UserApiKeyCache()),
            general_settings={},
            proxy_config=ProxyConfig(),
            llm_router=Router(
                model_list=[
                    {
                        "model_name": "claude-haiku-5-5",
                        "litellm_params": {"model": _ROUNDS_MODEL, "api_key": "sk-mcp-rounds", "api_base": api_base},
                    }
                ]
            ),
            is_streaming_request=stream,
        )
        if not isinstance(result, StreamingResponse):
            return fastapi_response.headers
        _ = [chunk async for chunk in result.body_iterator]
    return result.headers


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.asyncio
async def test_auto_execute_cost_headers_price_every_round(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch, messages_entry: str, stream: bool
):
    rounds: Final = (_ASKS_FOR_ADD, _ANSWERS)
    _ = respx_mock.route(host="127.0.0.1").pass_through()
    headers: Final = await _proxy_cost_headers(rounds, stream)
    await _settled()
    plain_rows: Final = await _plain_rows(respx_mock, monkeypatch, rounds)

    summed_cost: Final = sum(plain["response_cost"] for plain in plain_rows)
    priced_headers: Final = (
        ("x-litellm-response-cost-original",)
        if stream
        else ("x-litellm-response-cost", "x-litellm-response-cost-original")
    )
    assert {name: float(headers[name]) for name in priced_headers} == {
        name: pytest.approx(summed_cost) for name in priced_headers
    }


def _chat_round(finish_reason: str, prompt_tokens: int, completion_tokens: int) -> Mapping[str, object]:
    calls_the_tool: Final = finish_reason == "tool_calls"
    return MappingProxyType(
        {
            "id": f"chatcmpl-{finish_reason}-{prompt_tokens}",
            "object": "chat.completion",
            "created": 1,
            "model": "Qwen/Qwen3-8B",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": finish_reason,
                    "message": {
                        "role": "assistant",
                        "content": None if calls_the_tool else "Five apples sat in a row",
                        **(
                            {
                                "tool_calls": [
                                    {
                                        "id": "call_add",
                                        "type": "function",
                                        "function": {"name": "add", "arguments": '{"a": 2, "b": 3}'},
                                    }
                                ]
                            }
                            if calls_the_tool
                            else {}
                        ),
                    },
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }
    )


@pytest.mark.parametrize(
    "pricing",
    [
        pytest.param(MappingProxyType({"input_cost_per_token": 1e-6, "output_cost_per_token": 2e-6}), id="priced"),
        pytest.param(MappingProxyType({}), id="unpriced"),
    ],
)
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.asyncio
async def test_auto_execute_bills_every_round_a_chat_completions_model_answered(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch, stream: bool, pricing: Mapping[str, object]
):
    rounds: Final = (_chat_round("tool_calls", 418, 68), _chat_round("stop", 492, 432))
    recorder, served = await _auto_execute(
        respx_mock,
        monkeypatch,
        tuple(_ok(body) for body in rounds),
        stream=stream,
        call_id=f"mcp-messages-bridged-{stream}-{bool(pricing)}",
        model=_BRIDGED_MODEL,
        extra=pricing,
    )
    plain_rows: Final = await _plain_rows(respx_mock, monkeypatch, rounds, _BRIDGED_MODEL, pricing)

    assert isinstance(served, _Served)
    assert len(recorder.logged) == 1
    row: Final = recorder.logged[0]
    assert (row["prompt_tokens"], row["completion_tokens"]) == (418 + 492, 68 + 432)
    assert row["response_cost"] == pytest.approx(sum(plain["response_cost"] for plain in plain_rows))
    assert (row["response_cost"] > 0) == bool(pricing)
    assert (served.usage()["input_tokens"], served.usage()["output_tokens"]) == (418 + 492, 68 + 432)


class _ThreadedRoundRecorder(_RoundRecorder):
    def __init__(self) -> None:
        super().__init__()
        self.announced: Final = threading.Event()

    async def _announce(self) -> None:
        self.announced.set()


@pytest.mark.parametrize("reads_the_stream", [True, False], ids=["read", "dropped"])
def test_auto_execute_bills_every_round_of_a_stream_held_outside_an_event_loop(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch, messages_entry: str, reads_the_stream: bool
):
    rounds: Final = (_ASKS_FOR_ADD, _ANSWERS)
    _serve_rounds(respx_mock, monkeypatch, _ROUNDS_HOST, tuple(_ok(body) for body in rounds))
    recorder: Final = _ThreadedRoundRecorder()
    with _add_tool_served(AsyncMock(return_value=list(_ADDED))):
        stream: Final = asyncio.run(
            litellm.anthropic_messages(
                model=_ROUNDS_MODEL,
                max_tokens=1000,
                messages=[dict(_ASK)],
                tools=[dict(_ADD_REFERENCE)],
                api_key="sk-mcp-messages-rounds",
                api_base=f"https://{_ROUNDS_HOST}",
                stream=True,
                client=AsyncHTTPHandler(),
                litellm_logging_obj=_rounds_logging_obj(recorder, f"mcp-messages-sync-{reads_the_stream}", True),
            )
        )
    assert recorder.logged == ()
    if reads_the_stream:
        _ = list(cast(Iterator[bytes], stream))
    else:
        del stream
    assert recorder.announced.wait(timeout=_NEVER_LOGGED_SECONDS)
    plain_rows: Final = asyncio.run(_plain_rows(respx_mock, monkeypatch, rounds))

    assert len(recorder.logged) == 1
    _assert_bills(recorder.logged[0], plain_rows)


@pytest.mark.asyncio
async def test_auto_execute_bills_its_one_round_when_the_model_stops_for_a_tool_it_never_named(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
):
    names_no_tool: Final = MappingProxyType({**_ANSWERS, "stop_reason": "tool_use"})
    execute: Final = AsyncMock(return_value=list(_ADDED))
    recorder, served = await _auto_execute(
        respx_mock,
        monkeypatch,
        (_ok(names_no_tool),),
        stream=True,
        call_id="mcp-messages-no-tool-named",
        execute=execute,
    )
    plain_rows: Final = await _plain_rows(respx_mock, monkeypatch, (names_no_tool,))

    assert isinstance(served, _Served)
    assert execute.await_count == 0
    assert len(recorder.logged) == 1
    _assert_bills(recorder.logged[0], plain_rows)


@pytest.mark.asyncio
async def test_anthropic_messages_with_mcp_streams_the_summed_usage_without_a_logging_object():
    from litellm.llms.anthropic.messages.native import mcp_handler

    rounds: Final = AsyncMock(side_effect=[dict(_ASKS_FOR_ADD), dict(_ANSWERS)])
    with _add_tool_served(AsyncMock(return_value=list(_ADDED))), patch("litellm.anthropic_messages", new=rounds):
        result: Final = await mcp_handler.anthropic_messages_with_mcp(
            max_tokens=1000,
            messages=[dict(_ASK)],
            model=_ROUNDS_MODEL,
            tools=[dict(_ADD_REFERENCE)],
            stream=True,
        )
    served: Final = await _served(result)

    assert rounds.await_count == 2
    assert (served.started_usage()["input_tokens"], served.usage()["output_tokens"]) == (418 + 492, 68 + 432)
