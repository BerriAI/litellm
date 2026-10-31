from typing import Final

import pytest

from litellm.router_strategy.switchyard_router.execution import (
    SwitchyardReplay,
    child_call_kwargs,
    scoped_switchyard_headers,
)
from litellm.types.utils import ModelResponse


def test_child_calls_preserve_caller_without_sharing_log_identity_or_budget_reservation():
    parent: Final = {
        "metadata": {"user_api_key_hash": "caller", "user_api_key_budget_reservation": "reserved"},
        "litellm_call_id": "parent",
        "litellm_logging_obj": object(),
        "litellm_trace_id": "trace",
    }
    payload: Final = {"model": "fast", "messages": [{"role": "user", "content": "hello"}], "stream": False}
    first: Final = child_call_kwargs(parent, payload)
    second: Final = child_call_kwargs(parent, payload)
    assert first["litellm_call_id"] != second["litellm_call_id"]
    assert first["litellm_trace_id"] == "trace"
    assert "litellm_logging_obj" not in first
    assert first["metadata"] == {"user_api_key_hash": "caller", "internal_call_origin": "switchyard_router"}
    assert parent["metadata"]["user_api_key_budget_reservation"] == "reserved"


def test_session_cache_keys_are_scoped_and_never_forward_credentials():
    request: Final = {
        "metadata": {"user_api_key_hash": "first"},
        "proxy_server_request": {"headers": {"authorization": "secret", "x-session-id": "same-session"}},
    }
    first: Final = scoped_switchyard_headers(request)
    other: Final = scoped_switchyard_headers({**request, "metadata": {"user_api_key_hash": "second"}})
    assert tuple(first) == ("x-switchyard-session-id",)
    assert first == scoped_switchyard_headers(request)
    assert first != other
    assert "secret" not in str(first)
    assert "same-session" not in str(first)


def test_message_hash_fallback_is_scoped_without_client_session_header():
    request: Final = {"metadata": {"user_api_key_hash": "first"}, "messages": [{"role": "user", "content": "same"}]}
    first: Final = scoped_switchyard_headers(request, True)
    assert first == scoped_switchyard_headers(request, True)
    assert first != scoped_switchyard_headers({**request, "metadata": {"user_api_key_hash": "second"}}, True)


@pytest.mark.asyncio
@pytest.mark.parametrize("include_usage", [True, False])
async def test_buffered_stream_replays_content_and_preserves_original_usage(include_usage: bool):
    response: Final = ModelResponse(
        id="answer",
        model="fast",
        created=123,
        choices=[{"index": 0, "message": {"role": "assistant", "content": "hello world"}, "finish_reason": "stop"}],
        usage={"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
    )
    replay: Final = SwitchyardReplay(response, include_usage)
    frames: Final = [frame async for frame in replay]
    assert "".join(choice.delta.content or "" for frame in frames for choice in frame.choices) == "hello world"
    assert any(choice.finish_reason == "stop" for frame in frames for choice in frame.choices)
    if include_usage:
        assert frames[-1].usage == response.usage
        assert frames[-1].choices == []
    else:
        assert all(getattr(frame, "usage", None) is None for frame in frames)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [False, True])
async def test_stream_finalizes_once_on_completion_or_failure(failure: bool):
    from collections.abc import AsyncIterator
    from unittest.mock import AsyncMock

    from litellm.router_strategy.switchyard_router.execution import FinalizingStream
    from litellm.types.utils import ModelResponseStream

    async def source() -> AsyncIterator[ModelResponseStream]:
        yield ModelResponseStream(id="first")
        if failure:
            raise ValueError("provider failed")

    stream: Final = source()
    finalize: Final = AsyncMock()
    wrapped: Final = FinalizingStream(stream, finalize, stream.aclose, {}, None)
    assert (await anext(wrapped)).id == "first"
    with pytest.raises(ValueError if failure else StopAsyncIteration):
        await anext(wrapped)
    await wrapped.aclose()
    finalize.assert_awaited_once()


@pytest.mark.asyncio
async def test_stream_finalizes_on_disconnect_before_first_chunk():
    from collections.abc import AsyncIterator
    from unittest.mock import AsyncMock

    from litellm.router_strategy.switchyard_router.execution import FinalizingStream
    from litellm.types.utils import ModelResponseStream

    async def source() -> AsyncIterator[ModelResponseStream]:
        yield ModelResponseStream(id="unread")

    stream: Final = source()
    finalize: Final = AsyncMock()
    wrapped: Final = FinalizingStream(stream, finalize, stream.aclose, {}, None)
    await wrapped.aclose()
    with pytest.raises(StopAsyncIteration):
        await anext(wrapped)
    finalize.assert_awaited_once()


def test_judge_payload_does_not_inherit_parent_generation_controls():
    parent: Final = {
        "tools": [{"type": "function", "function": {"name": "run"}}],
        "tool_choice": "required",
        "max_completion_tokens": 1000,
        "response_format": {"type": "json_object"},
        "temperature": 0.9,
        "timeout": 10,
        "litellm_trace_id": "trace",
    }
    child: Final = child_call_kwargs(parent, {"messages": [{"role": "user", "content": "classify"}]})
    assert not {"tools", "tool_choice", "max_completion_tokens", "response_format", "temperature"}.intersection(child)
    assert child["timeout"] == 10
    assert child["litellm_trace_id"] == "trace"


def test_sdk_sessions_are_stable_within_router_and_isolated_between_routers():
    request: Final = {"litellm_session_id": "conversation"}
    first: Final = scoped_switchyard_headers(request, sdk_namespace="router-one")
    assert first == scoped_switchyard_headers(request, sdk_namespace="router-one")
    assert first != scoped_switchyard_headers(request, sdk_namespace="router-two")


def test_proxy_missing_identity_cannot_reuse_sdk_scope():
    request: Final = {"litellm_session_id": "conversation", "proxy_server_request": {"headers": {}}}
    assert scoped_switchyard_headers(request, sdk_namespace="router") != scoped_switchyard_headers(
        request, sdk_namespace="router"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("cancelled", [False, True])
async def test_stream_disconnect_bills_before_close_and_finalizes_under_cancellation(cancelled: bool):
    from collections.abc import AsyncIterator

    import anyio

    from litellm.router_strategy.switchyard_router.execution import FinalizingStream
    from litellm.types.utils import ModelResponseStream

    events: Final[list[str]] = []

    async def source() -> AsyncIterator[ModelResponseStream]:
        yield ModelResponseStream(id="first")

    async def bill() -> None:
        await anyio.lowlevel.checkpoint()
        events.append("bill")

    async def close() -> None:
        await anyio.lowlevel.checkpoint()
        events.append("close")

    async def finalize() -> None:
        await anyio.lowlevel.checkpoint()
        events.append("finalize")

    wrapped: Final = FinalizingStream(source(), finalize, close, {}, None, on_disconnect=bill)
    await anext(wrapped)
    with anyio.CancelScope() as scope:
        if cancelled:
            scope.cancel()
        await wrapped.aclose()
    await wrapped.aclose()
    assert events == ["bill", "close", "finalize"]
