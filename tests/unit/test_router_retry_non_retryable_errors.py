"""
Test that the Router retry loop correctly handles non-retryable errors.

Verifies that:
1. Non-retryable errors (e.g., 400 ContextWindowExceeded) inside the retry loop
   break out immediately instead of being swallowed.
2. original_exception is updated to the latest error, not stuck on the first.
3. Retryable errors (e.g., 429 RateLimitError) still retry normally.

Regression tests for https://github.com/BerriAI/litellm/issues/21343
"""

import asyncio
import datetime
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import TYPE_CHECKING, Final, NamedTuple
from unittest.mock import AsyncMock, patch

import httpx
import pytest
import respx

import litellm
from litellm import Router
from litellm.exceptions import GuardrailRaisedException
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.litellm_core_utils.llm_response_utils.response_metadata import (
    _union_duration_ms,
    response_timing_metrics,
)
from litellm.litellm_core_utils.logging_utils import track_llm_api_timing
from litellm.litellm_core_utils.rules import Rules
from litellm.router_utils.mcp_tool_execution import mark_mcp_tools_executed
from litellm.types.router import RetryPolicy
from litellm.utils import function_setup

if TYPE_CHECKING:
    from litellm.proxy._experimental.mcp_server.mcp_server_manager import MCPServerManager
    from litellm.proxy._types import UserAPIKeyAuth


def _make_rate_limit_error(message="Rate limited"):
    """Create a RateLimitError for testing."""
    return litellm.RateLimitError(
        message=message,
        llm_provider="bedrock",
        model="anthropic.claude-v2",
    )


def _make_context_window_error(message="prompt is too long: 1205821 tokens > 200000"):
    """Create a ContextWindowExceededError for testing."""
    return litellm.ContextWindowExceededError(
        message=message,
        llm_provider="vertex_ai",
        model="claude-3-opus",
    )


def _make_bad_request_error(message="Invalid request"):
    """Create a BadRequestError for testing."""
    return litellm.BadRequestError(
        message=message,
        llm_provider="openai",
        model="gpt-4",
    )


def _make_not_found_error(message="Model not found"):
    """Create a NotFoundError for testing."""
    return litellm.NotFoundError(
        message=message,
        llm_provider="openai",
        model="gpt-99",
    )


def _create_router(num_retries=2, retry_policy=None):
    """Create a Router with two deployments for testing."""
    return Router(
        model_list=[
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "openai/gpt-4",
                    "api_key": "fake-key-1",
                },
            },
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "openai/gpt-4",
                    "api_key": "fake-key-2",
                },
            },
        ],
        num_retries=num_retries,
        retry_policy=retry_policy,
    )


def _base_kwargs():
    """Return kwargs required by async_function_with_retries."""
    return {
        "model": "test-model",
        "messages": [{"role": "user", "content": "test"}],
        "original_function": AsyncMock(),
        "metadata": {},
    }


@pytest.mark.asyncio
async def test_non_retryable_error_in_retry_loop_raises_immediately():
    """
    When a non-retryable error (400 ContextWindowExceeded) occurs inside the
    retry loop, the router should raise it immediately instead of swallowing it
    and raising the original error.

    Scenario: First call -> 429, Retry -> 400 (non-retryable)
    Expected: ContextWindowExceededError is raised, NOT RateLimitError
    """
    router = _create_router(num_retries=2)

    rate_limit_error = _make_rate_limit_error()
    context_window_error = _make_context_window_error()

    call_count = 0

    async def mock_make_call(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise rate_limit_error
        else:
            raise context_window_error

    with (
        patch.object(router, "make_call", side_effect=mock_make_call),
        patch.object(
            router,
            "_async_get_healthy_deployments",
            return_value=(["d1", "d2"], ["d1", "d2"]),
        ),
        patch.object(router, "_time_to_sleep_before_retry", return_value=0),
        patch.object(router, "log_retry", side_effect=lambda kwargs, e: kwargs),
    ):
        with pytest.raises(litellm.ContextWindowExceededError):
            await router.async_function_with_retries(
                num_retries=2,
                **_base_kwargs(),
            )


@pytest.mark.asyncio
async def test_bad_request_error_in_retry_loop_raises_immediately():
    """
    A generic 400 BadRequestError inside the retry loop should also break out
    immediately since 400 is not retryable.
    """
    router = _create_router(num_retries=2)

    rate_limit_error = _make_rate_limit_error()
    bad_request_error = _make_bad_request_error()

    call_count = 0

    async def mock_make_call(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise rate_limit_error
        else:
            raise bad_request_error

    with (
        patch.object(router, "make_call", side_effect=mock_make_call),
        patch.object(
            router,
            "_async_get_healthy_deployments",
            return_value=(["d1", "d2"], ["d1", "d2"]),
        ),
        patch.object(router, "_time_to_sleep_before_retry", return_value=0),
        patch.object(router, "log_retry", side_effect=lambda kwargs, e: kwargs),
    ):
        with pytest.raises(litellm.BadRequestError):
            await router.async_function_with_retries(
                num_retries=2,
                **_base_kwargs(),
            )


@pytest.mark.asyncio
async def test_original_exception_updated_to_latest_error():
    """
    When all retries are exhausted with retryable errors, the LAST error
    should be raised, not the first one.
    """
    router = _create_router(num_retries=2)

    call_count = 0

    async def mock_make_call(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        raise _make_rate_limit_error(f"Rate limit attempt {call_count}")

    with (
        patch.object(router, "make_call", side_effect=mock_make_call),
        patch.object(
            router,
            "_async_get_healthy_deployments",
            return_value=(["d1", "d2"], ["d1", "d2"]),
        ),
        patch.object(router, "_time_to_sleep_before_retry", return_value=0),
        patch.object(router, "log_retry", side_effect=lambda kwargs, e: kwargs),
    ):
        with pytest.raises(litellm.RateLimitError) as exc_info:
            await router.async_function_with_retries(
                num_retries=2,
                **_base_kwargs(),
            )
        # Should be the LAST error, not the first
        assert "Rate limit attempt 3" in str(exc_info.value)


@pytest.mark.asyncio
async def test_retryable_errors_still_retry_normally():
    """
    Retryable errors (429 RateLimitError) should still be retried the
    configured number of times before raising.
    """
    router = _create_router(num_retries=3)

    call_count = 0

    async def mock_make_call(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        raise _make_rate_limit_error(f"Rate limit attempt {call_count}")

    with (
        patch.object(router, "make_call", side_effect=mock_make_call),
        patch.object(
            router,
            "_async_get_healthy_deployments",
            return_value=(["d1", "d2"], ["d1", "d2"]),
        ),
        patch.object(router, "_time_to_sleep_before_retry", return_value=0),
        patch.object(router, "log_retry", side_effect=lambda kwargs, e: kwargs),
    ):
        with pytest.raises(litellm.RateLimitError):
            await router.async_function_with_retries(
                num_retries=3,
                **_base_kwargs(),
            )

        # Initial call + 3 retries = 4 total calls
        assert call_count == 4


@pytest.mark.asyncio
async def test_not_found_error_in_retry_loop_raises_immediately():
    """
    A 404 NotFoundError inside the retry loop should break out immediately.
    """
    router = _create_router(num_retries=2)

    rate_limit_error = _make_rate_limit_error()
    not_found_error = _make_not_found_error()

    call_count = 0

    async def mock_make_call(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise rate_limit_error
        else:
            raise not_found_error

    with (
        patch.object(router, "make_call", side_effect=mock_make_call),
        patch.object(
            router,
            "_async_get_healthy_deployments",
            return_value=(["d1", "d2"], ["d1", "d2"]),
        ),
        patch.object(router, "_time_to_sleep_before_retry", return_value=0),
        patch.object(router, "log_retry", side_effect=lambda kwargs, e: kwargs),
    ):
        with pytest.raises(litellm.NotFoundError):
            await router.async_function_with_retries(
                num_retries=2,
                **_base_kwargs(),
            )

        # Only 2 calls: initial + first retry that hits non-retryable
        assert call_count == 2


@pytest.mark.asyncio
async def test_retry_attempts_accumulate_timing_in_shared_request_metadata():
    received_at: Final = datetime.datetime.now()
    metadata: dict[str, object] = {
        "model_group": "test-model",
        "litellm_received_at": received_at,
    }
    logging_obj_raw, _ = function_setup(
        "acompletion",
        Rules(),
        datetime.datetime.now(),
        model="test-model",
        messages=[{"role": "user", "content": "test"}],
        metadata=metadata,
        litellm_call_id="retry-timing-test",
        is_async_call=True,
    )
    assert isinstance(logging_obj_raw, Logging)
    logging_obj: Final[Logging] = logging_obj_raw
    attempt_numbers: list[int] = []
    metadata_ids: list[int] = []

    @track_llm_api_timing()
    async def timed_attempt(*, logging_obj: Logging, **kwargs: object) -> str:
        del kwargs
        attempt_numbers.append(len(attempt_numbers) + 1)
        metadata_ids.append(id(logging_obj.model_call_details["litellm_params"]["metadata"]))
        await asyncio.sleep(0.01)
        if len(attempt_numbers) == 1:
            raise _make_rate_limit_error()
        return "success"

    async def invoke(original_function: Callable[..., Awaitable[str]], *args: object, **kwargs: object) -> str:
        return await original_function(*args, **kwargs)

    router = _create_router(num_retries=1)
    with (
        patch.object(router, "make_call", new=AsyncMock(side_effect=invoke)),
        patch.object(
            router,
            "_async_get_healthy_deployments",
            new=AsyncMock(return_value=(["d1"], ["d1"])),
        ),
        patch.object(router, "_time_to_sleep_before_retry", return_value=0),
    ):
        result = await router.async_function_with_retries(
            original_function=timed_attempt,
            model="test-model",
            messages=[{"role": "user", "content": "test"}],
            metadata=metadata,
            logging_obj=logging_obj,
            num_retries=1,
        )

    request_metadata: Final = logging_obj.model_call_details["litellm_params"]["metadata"]
    windows: Final = request_metadata["llm_api_timing_windows"]
    end_time: Final = datetime.datetime.fromtimestamp(max(window[1] for window in windows))
    timing_metrics: Final = response_timing_metrics(received_at, end_time, logging_obj)
    assert result == "success"
    assert attempt_numbers == [1, 2]
    assert request_metadata is metadata
    assert metadata_ids == [id(metadata), id(metadata)]
    assert len(windows) == 2
    union_duration_ms: Final = _union_duration_ms(windows, received_at.timestamp(), end_time.timestamp())
    assert union_duration_ms is not None
    total_response_time_ms: Final = (end_time.timestamp() - received_at.timestamp()) * 1000
    assert timing_metrics["litellm_overhead_time_ms"] == pytest.approx(
        round(total_response_time_ms - union_duration_ms, 4)
    )


def _make_follow_up_error(tools_executed: bool) -> litellm.InternalServerError:
    error: Final = litellm.InternalServerError(message="follow-up failed", llm_provider="openai", model="gpt-4")
    if tools_executed:
        mark_mcp_tools_executed(error)
    return error


@pytest.mark.asyncio
@pytest.mark.parametrize("retry_policy", [None, RetryPolicy(InternalServerErrorRetries=2)])
@pytest.mark.parametrize("tools_executed", [True, False])
async def test_500_after_mcp_tool_execution_is_not_retried(retry_policy, tools_executed):
    """
    Regression test for #43153: a retry re-runs the whole MCP gateway loop, so a
    failure raised after the loop executed tool calls must surface instead of
    executing those tools again. A retry policy that retries 500s must not override
    that, and a 500 raised before any tool ran must still be retried.
    """
    router: Final = _create_router(num_retries=2, retry_policy=retry_policy)
    error: Final = _make_follow_up_error(tools_executed)
    attempt: Final = AsyncMock(side_effect=error)

    with pytest.raises(litellm.InternalServerError) as exc_info:
        await router.async_function_with_retries(**{**_base_kwargs(), "original_function": attempt, "num_retries": 2})

    assert exc_info.value is error
    assert attempt.await_count == (1 if tools_executed else 3)


@pytest.mark.asyncio
async def test_500_after_mcp_tool_execution_stops_an_ongoing_retry_loop():
    """
    The first attempt fails before any tool ran and is retried; the retry executes
    the tools and then fails, so the loop must stop there instead of running them
    a second time.
    """
    router: Final = _create_router(num_retries=3)
    after_tools: Final = _make_follow_up_error(tools_executed=True)
    attempt: Final = AsyncMock(side_effect=[_make_follow_up_error(tools_executed=False), after_tools])

    with pytest.raises(litellm.InternalServerError) as exc_info:
        await router.async_function_with_retries(**{**_base_kwargs(), "original_function": attempt, "num_retries": 3})

    assert exc_info.value is after_tools
    assert attempt.await_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("tools_executed", [True, False])
async def test_500_after_mcp_tool_execution_does_not_fall_back(tools_executed):
    """
    A fallback re-runs the whole MCP gateway loop on the fallback group, so it must
    not start once the primary attempt executed tool calls.
    """
    router: Final = Router(
        model_list=[
            {"model_name": "test-model", "litellm_params": {"model": "openai/gpt-4", "api_key": "fake-key-1"}},
            {"model_name": "backup-model", "litellm_params": {"model": "openai/gpt-4", "api_key": "fake-key-2"}},
        ],
        num_retries=0,
        fallbacks=[{"test-model": ["backup-model"]}],
    )
    error: Final = _make_follow_up_error(tools_executed)
    fallback_response: Final = litellm.ModelResponse()

    async def primary_fails(**kwargs):
        if kwargs["model"] == "test-model":
            raise error
        return fallback_response

    attempt: Final = AsyncMock(side_effect=primary_fails)
    call: Final = router.async_function_with_fallbacks(
        **{**_base_kwargs(), "original_function": attempt, "num_retries": 0}
    )

    if tools_executed:
        with pytest.raises(litellm.InternalServerError) as exc_info:
            await call
        assert exc_info.value is error
    else:
        assert await call is fallback_response

    attempted_models: Final = [awaited.kwargs["model"] for awaited in attempt.await_args_list]
    assert attempted_models == (["test-model"] if tools_executed else ["test-model", "backup-model"])


_UPSTREAM: Final = "https://upstream.example/v1"
_TOOL_NAME: Final = "ledger-commit_write"


def _ledger_tools() -> list[dict[str, str]]:
    return [{"type": "mcp", "server_url": "litellm_proxy/ledger", "require_approval": "never"}]


def _ledger_caller() -> "UserAPIKeyAuth":
    from litellm.proxy._types import LiteLLM_ObjectPermissionTable, UserAPIKeyAuth

    return UserAPIKeyAuth(
        object_permission=LiteLLM_ObjectPermissionTable(object_permission_id="test", mcp_servers=["ledger"])
    )


class _LedgerGateway(NamedTuple):
    tool: AsyncMock
    list_tools: AsyncMock
    manager: "MCPServerManager"


@pytest.fixture
def ledger_gateway(monkeypatch: pytest.MonkeyPatch) -> _LedgerGateway:
    """Serve one auto-approved MCP tool through the real MCP manager and tool registry."""
    from mcp.types import Tool

    from litellm.caching.caching import DualCache
    from litellm.proxy import proxy_server
    from litellm.proxy._experimental.mcp_server import mcp_server_manager, server, tool_registry
    from litellm.proxy._experimental.mcp_server.faults.list_outcomes import AggregateToolListing
    from litellm.proxy.utils import ProxyLogging
    from litellm.types.mcp_server.mcp_server_manager import MCPServer

    manager: Final = mcp_server_manager.MCPServerManager()
    manager.registry = {
        "ledger": MCPServer(
            server_id="ledger",
            name="ledger",
            server_name="ledger",
            transport="http",
            url="https://ledger.example/mcp",
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
    return _LedgerGateway(tool=tool, list_tools=list_tools, manager=manager)


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


class _StreamBrokenBeforeFirstChunk(httpx.AsyncByteStream):
    async def __aiter__(self) -> AsyncIterator[bytes]:
        raise httpx.ReadError("connection dropped before the first chunk")
        yield b""


def _chat_tool_call_stream() -> httpx.Response:
    tool_call: Final = {
        "index": 0,
        "id": "call_1",
        "type": "function",
        "function": {"name": _TOOL_NAME, "arguments": json.dumps({"entry": "alpha"})},
    }
    chunks: Final = (
        {"role": "assistant", "content": None, "tool_calls": [tool_call]},
        {},
    )
    body: Final = "".join(
        "data: "
        + json.dumps(
            {
                "id": "chatcmpl-1",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": "gpt-4.1-mini",
                "choices": [{"index": 0, "delta": delta, "finish_reason": None if delta else "tool_calls"}],
            }
        )
        + "\n\n"
        for delta in chunks
    )
    return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=f"{body}data: [DONE]\n\n")


def _serve(model: _FakeModel) -> None:
    respx.post(f"{_UPSTREAM}/chat/completions").mock(side_effect=model)
    respx.post(f"{_UPSTREAM}/responses").mock(side_effect=model)


def _mcp_router(num_retries: int, fallbacks: list[dict[str, list[str]]] | None = None) -> litellm.Router:
    deployment: Final = {"model": "openai/gpt-4.1-mini", "api_key": "sk-test", "api_base": _UPSTREAM}
    return litellm.Router(
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
        await send(_mcp_router(**router_kwargs))

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

    await send(_mcp_router(num_retries=2))

    assert ledger_gateway.tool.await_count == 0
    assert model.calls == ["initial", "follow_up", "initial", "follow_up"]


@pytest.mark.asyncio
@_ENDPOINTS
@respx.mock
async def test_router_does_not_replay_a_tool_call_that_failed_after_it_was_sent(
    ledger_gateway: _LedgerGateway,
    send: Callable[[litellm.Router], Awaitable[object]],
    monkeypatch: pytest.MonkeyPatch,
):
    """
    Given: The tool runs, but its call fails afterwards, here because a during-call guardrail
           rejects the result, which leaves the router in the same spot as a timeout after the write
    When:  Every follow-up model call returns a 500
    Then:  The router does not replay the request, because the tool may already have written
    """
    tool_ran: Final = asyncio.Event()

    def write(**_: object) -> dict[str, bool]:
        tool_ran.set()
        return {"written": True}

    class RejectToolResult(CustomGuardrail):
        async def async_moderation_hook(self, data, user_api_key_dict, call_type):
            await tool_ran.wait()
            raise GuardrailRaisedException(message="tool result rejected", blocked_content=True)

    ledger_gateway.tool.side_effect = write
    monkeypatch.setattr(
        litellm, "callbacks", [RejectToolResult(guardrail_name="reject", event_hook="during_mcp_call", default_on=True)]
    )
    model: Final = _FakeModel(failing_follow_ups=3)
    _serve(model)

    with pytest.raises(litellm.InternalServerError):
        await send(_mcp_router(num_retries=2))

    assert ledger_gateway.tool.await_count == 1
    assert model.calls == ["initial", "follow_up"]


@pytest.mark.asyncio
@respx.mock
async def test_router_retries_when_an_mcp_tool_call_never_reached_its_server(
    ledger_gateway: _LedgerGateway,
    monkeypatch: pytest.MonkeyPatch,
):
    """
    Given: The MCP server needs a per-user header the caller has not set, so the tool call fails
           before anything is sent to the server
    When:  The follow-up model call fails once with a 500
    Then:  The router still retries, because replaying the request cannot repeat a side effect
    """
    registry: Final = ledger_gateway.manager.registry
    monkeypatch.setitem(
        registry,
        "ledger",
        registry["ledger"].model_copy(
            update={
                "spec_path": None,
                "static_headers": {"Authorization": "Bearer ${LEDGER_TOKEN}"},
                "env_vars": [{"name": "LEDGER_TOKEN", "scope": "user"}],
            }
        ),
    )
    model: Final = _FakeModel(failing_follow_ups=1)
    _serve(model)

    await _chat(_mcp_router(num_retries=2))

    assert model.calls == ["initial", "follow_up", "initial", "follow_up"]


@pytest.mark.asyncio
@respx.mock
async def test_router_does_not_fall_back_when_a_streamed_follow_up_breaks_after_a_tool_ran(
    ledger_gateway: _LedgerGateway,
):
    """
    Given: A streaming Chat Completions request whose tool ran
    When:  The follow-up stream breaks before its first chunk and a fallback group is configured
    Then:  The client gets the error instead of a fallback that runs the tool again
    """
    calls: Final[list[str]] = []

    def stream_model(request: httpx.Request) -> httpx.Response:
        kind: Final = "follow_up" if _is_follow_up(request) else "initial"
        calls.append(kind)
        if kind == "follow_up":
            return httpx.Response(
                200, headers={"content-type": "text/event-stream"}, stream=_StreamBrokenBeforeFirstChunk()
            )
        return _chat_tool_call_stream()

    respx.post(f"{_UPSTREAM}/chat/completions").mock(side_effect=stream_model)
    stream: Final = await _mcp_router(num_retries=0, fallbacks=[{"repro-model": ["backup-model"]}]).acompletion(
        model="repro-model",
        messages=[{"role": "user", "content": "Record entry alpha"}],
        tools=_ledger_tools(),
        metadata={"user_api_key_auth": _ledger_caller()},
        stream=True,
    )

    with pytest.raises(litellm.APIConnectionError):
        async for _ in stream:
            pass

    assert ledger_gateway.tool.await_count == 1
    assert calls == ["initial", "follow_up"]


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

    await _chat(_mcp_router(num_retries=2))

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
        await _responses(_mcp_router(num_retries=2))

    assert ledger_gateway.tool.await_count == 1
    assert ledger_gateway.list_tools.await_count == 2
    assert model.calls == ["initial", "follow_up"]
