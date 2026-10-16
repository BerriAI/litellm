import asyncio
import json
import math
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Final
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import respx
from openai.types.chat import ChatCompletion
from pydantic import TypeAdapter

import litellm
from litellm.caching.caching import Cache
from litellm.completion_extras.litellm_responses_transformation.handler import (
    ResponsesToCompletionBridgeHandler,
)
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLogging
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
from litellm.responses import bridge_executor
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import ModelResponse


def test_is_preformatted_cached_chat_stream_true():
    stream = MagicMock(spec=CustomStreamWrapper)
    stream.custom_llm_provider = "cached_response"
    assert (
        ResponsesToCompletionBridgeHandler._is_preformatted_cached_chat_stream(stream)
        is True
    )


def test_is_preformatted_cached_chat_stream_false_wrong_provider():
    stream = MagicMock(spec=CustomStreamWrapper)
    stream.custom_llm_provider = "openai"
    assert (
        ResponsesToCompletionBridgeHandler._is_preformatted_cached_chat_stream(stream)
        is False
    )


def test_is_preformatted_cached_chat_stream_false_wrong_type():
    assert (
        ResponsesToCompletionBridgeHandler._is_preformatted_cached_chat_stream(
            {"object": "chat.completion.chunk"}
        )
        is False
    )


@pytest.fixture
def _cache_on(monkeypatch):
    monkeypatch.setattr(litellm, "cache", MagicMock(spec=Cache))


def _bridge_kwargs(stream: bool):
    logging_obj = LiteLLMLogging(
        litellm_call_id="test-call",
        call_type="completion",
        model="gpt-5.4",
        messages=[{"role": "user", "content": "hi"}],
        function_id="fn-id",
        stream=stream,
        start_time=datetime.now(),
    )
    return {
        "model": "gpt-5.4",
        "custom_llm_provider": "openai",
        "messages": [{"role": "user", "content": "hi"}],
        "optional_params": {"stream": stream},
        "litellm_params": {},
        "headers": {},
        "model_response": ModelResponse(),
        "logging_obj": logging_obj,
    }


def test_completion_returns_cached_model_response_directly(_cache_on):
    """Non-streaming bridge cache hit: responses() returns a ModelResponse -> bridge returns it as-is."""
    cached = ModelResponse(id="chatcmpl-cached-nonstream", model="gpt-5.4")
    bridge = ResponsesToCompletionBridgeHandler()

    with (
        patch.object(
            bridge.transformation_handler,
            "transform_request",
            return_value={"model": "gpt-5.4", "input": "hi"},
        ),
        patch("litellm.responses", return_value=cached),
    ):
        result = bridge.completion(**_bridge_kwargs(stream=False))

    assert result is cached


@pytest.mark.asyncio
async def test_acompletion_returns_cached_model_response_directly(_cache_on):
    cached = ModelResponse(id="chatcmpl-cached-nonstream-async", model="gpt-5.4")
    bridge = ResponsesToCompletionBridgeHandler()

    with (
        patch.object(
            bridge.transformation_handler,
            "transform_request",
            return_value={"model": "gpt-5.4", "input": "hi"},
        ),
        patch("litellm.aresponses", new=AsyncMock(return_value=cached)),
    ):
        result = await bridge.acompletion(**_bridge_kwargs(stream=False))

    assert result is cached


def test_completion_skips_rewrapping_preformatted_cached_chat_stream(_cache_on):
    """Streaming bridge cache hit returning CustomStreamWrapper(cached_response) -> bridge skips re-wrapping."""
    stream = MagicMock(spec=CustomStreamWrapper)
    stream.custom_llm_provider = "cached_response"
    bridge = ResponsesToCompletionBridgeHandler()

    with (
        patch.object(
            bridge.transformation_handler,
            "transform_request",
            return_value={"model": "gpt-5.4", "input": "hi"},
        ),
        patch("litellm.responses", return_value=stream),
        patch.object(
            bridge,
            "_apply_post_stream_processing",
            side_effect=lambda s, *a, **kw: s,
        ) as post,
    ):
        result = bridge.completion(**_bridge_kwargs(stream=True))

    post.assert_called_once()
    assert result is stream


def test_completion_preserves_top_level_stream_flag_in_responses_request(_cache_on):
    stream = MagicMock(spec=CustomStreamWrapper)
    stream.custom_llm_provider = "cached_response"
    bridge = ResponsesToCompletionBridgeHandler()
    kwargs = _bridge_kwargs(stream=False)
    kwargs["stream"] = True
    kwargs["optional_params"].pop("stream")

    with (
        patch.object(
            bridge.transformation_handler,
            "transform_request",
            return_value={"model": "gpt-5.4", "input": "hi"},
        ) as transform_request,
        patch("litellm.responses", return_value=stream),
        patch.object(
            bridge,
            "_apply_post_stream_processing",
            side_effect=lambda s, *a, **kw: s,
        ),
    ):
        result = bridge.completion(**kwargs)

    assert result is stream
    assert transform_request.call_args.kwargs["optional_params"]["stream"] is True


@pytest.mark.asyncio
async def test_acompletion_skips_rewrapping_preformatted_cached_chat_stream(_cache_on):
    stream = MagicMock(spec=CustomStreamWrapper)
    stream.custom_llm_provider = "cached_response"
    bridge = ResponsesToCompletionBridgeHandler()

    with (
        patch.object(
            bridge.transformation_handler,
            "transform_request",
            return_value={"model": "gpt-5.4", "input": "hi"},
        ),
        patch("litellm.aresponses", new=AsyncMock(return_value=stream)),
        patch.object(
            bridge,
            "_apply_post_stream_processing",
            side_effect=lambda s, *a, **kw: s,
        ) as post,
    ):
        result = await bridge.acompletion(**_bridge_kwargs(stream=True))

    post.assert_called_once()
    assert result is stream


@pytest.mark.asyncio
async def test_acompletion_preserves_top_level_stream_flag_in_responses_request(_cache_on):
    stream = MagicMock(spec=CustomStreamWrapper)
    stream.custom_llm_provider = "cached_response"
    bridge = ResponsesToCompletionBridgeHandler()
    kwargs = _bridge_kwargs(stream=False)
    kwargs["stream"] = True
    kwargs["optional_params"].pop("stream")

    with (
        patch.object(
            bridge.transformation_handler,
            "transform_request",
            return_value={"model": "gpt-5.4", "input": "hi"},
        ) as transform_request,
        patch("litellm.aresponses", new=AsyncMock(return_value=stream)),
        patch.object(
            bridge,
            "_apply_post_stream_processing",
            side_effect=lambda s, *a, **kw: s,
        ),
    ):
        result = await bridge.acompletion(**kwargs)

    assert result is stream
    assert transform_request.call_args.kwargs["optional_params"]["stream"] is True


def _completed_chat_response() -> ModelResponse:
    return ModelResponse(
        id="chatcmpl-completed",
        model="gpt-5.4",
        choices=[
            {
                "index": 0,
                "message": {"role": "assistant", "content": "pong"},
                "finish_reason": "stop",
            }
        ],
    )


@pytest.mark.asyncio
async def test_acompletion_streams_completed_model_response(_cache_on):
    """A streaming request whose bridge call comes back already completed must still be
    handed back as an async-iterable stream. Returning the bare ModelResponse crashed the
    proxy's SSE generator with "'async for' requires an object with __aiter__ method".
    Regression for #33154."""
    completed = _completed_chat_response()
    bridge = ResponsesToCompletionBridgeHandler()

    with (
        patch.object(
            bridge.transformation_handler,
            "transform_request",
            return_value={"model": "gpt-5.4", "input": "hi"},
        ),
        patch("litellm.aresponses", new=AsyncMock(return_value=completed)),
    ):
        result = await bridge.acompletion(**_bridge_kwargs(stream=True))

    assert isinstance(result, CustomStreamWrapper), f"streaming request got {type(result)}"
    chunks = [chunk async for chunk in result]
    assert "".join(
        chunk.choices[0].delta.content or "" for chunk in chunks
    ) == "pong", f"completed response did not stream its content: {chunks}"
    assert [c for c in chunks if c.choices[0].finish_reason], "stream never emitted a finish_reason"


def test_completion_streams_completed_model_response(_cache_on):
    completed = _completed_chat_response()
    bridge = ResponsesToCompletionBridgeHandler()

    with (
        patch.object(
            bridge.transformation_handler,
            "transform_request",
            return_value={"model": "gpt-5.4", "input": "hi"},
        ),
        patch("litellm.responses", return_value=completed),
    ):
        result = bridge.completion(**_bridge_kwargs(stream=True))

    assert isinstance(result, CustomStreamWrapper), f"streaming request got {type(result)}"
    chunks = list(result)
    assert "".join(chunk.choices[0].delta.content or "" for chunk in chunks) == "pong", (
        f"completed response did not stream its content: {chunks}"
    )


_PROVIDER_NATIVE_MODEL_CASES = [
    ("perplexity", "perplexity/kimi-k3", "perplexity/kimi-k3"),
    ("perplexity", "openai/gpt-5.2", "openai/gpt-5.2"),
    ("openai", "gpt-5.4", "gpt-5.4"),
]


def _upstream_model_for(handed_model: str, custom_llm_provider: str) -> str:
    upstream_model, _, _, _ = litellm.get_llm_provider(
        model=handed_model,
        litellm_params=GenericLiteLLMParams(custom_llm_provider=custom_llm_provider),
    )
    return upstream_model


@pytest.mark.parametrize(
    "custom_llm_provider, bridge_model, expected_upstream_model",
    _PROVIDER_NATIVE_MODEL_CASES,
)
def test_completion_keeps_provider_native_model_id_through_responses(
    custom_llm_provider, bridge_model, expected_upstream_model
):
    """The Responses executor resolves the provider itself, so the bridge must not hand it an already-stripped model."""
    cached = ModelResponse(id="chatcmpl-cached", model=bridge_model)
    bridge = ResponsesToCompletionBridgeHandler()
    kwargs = _bridge_kwargs(stream=False)
    kwargs["model"] = bridge_model
    kwargs["custom_llm_provider"] = custom_llm_provider

    with (
        patch.object(
            bridge.transformation_handler,
            "transform_request",
            return_value={"model": bridge_model, "input": "hi"},
        ),
        patch.object(bridge_executor, "native_responses", return_value=cached) as responses_call,
    ):
        bridge.completion(**kwargs)

    handed_model = responses_call.call_args.args[1]
    assert _upstream_model_for(handed_model, custom_llm_provider) == expected_upstream_model


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "custom_llm_provider, bridge_model, expected_upstream_model",
    _PROVIDER_NATIVE_MODEL_CASES,
)
async def test_acompletion_keeps_provider_native_model_id_through_responses(
    custom_llm_provider, bridge_model, expected_upstream_model
):
    cached = ModelResponse(id="chatcmpl-cached-async", model=bridge_model)
    bridge = ResponsesToCompletionBridgeHandler()
    kwargs = _bridge_kwargs(stream=False)
    kwargs["model"] = bridge_model
    kwargs["custom_llm_provider"] = custom_llm_provider

    with (
        patch.object(
            bridge.transformation_handler,
            "transform_request",
            return_value={"model": bridge_model, "input": "hi"},
        ),
        patch.object(bridge_executor, "anative_responses", new=AsyncMock(return_value=cached)) as responses_call,
    ):
        await bridge.acompletion(**kwargs)

    handed_model = responses_call.call_args.args[1]
    assert _upstream_model_for(handed_model, custom_llm_provider) == expected_upstream_model


class _DeploymentHookCounter(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    async def async_pre_call_deployment_hook(self, kwargs, call_type) -> None:
        self.calls += 1


@pytest.mark.asyncio
async def test_bridged_chat_call_runs_deployment_hooks_once(monkeypatch):
    counter = _DeploymentHookCounter()
    monkeypatch.setattr(litellm, "callbacks", [counter])
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    litellm.in_memory_llm_clients_cache.flush_cache()
    litellm.register_model(
        {"openai/bridge-only-model": {"litellm_provider": "openai", "mode": "responses", "max_tokens": 100}}
    )

    with respx.mock(assert_all_called=False) as upstream:
        responses_route = upstream.post("https://example.invalid/v1/responses").mock(
            return_value=httpx.Response(
                200,
                json={
                    "id": "resp_bridge",
                    "object": "response",
                    "created_at": 1,
                    "status": "completed",
                    "model": "bridge-only-model",
                    "output": [
                        {
                            "type": "message",
                            "id": "msg_bridge",
                            "status": "completed",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": "from upstream", "annotations": []}],
                        }
                    ],
                    "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                },
            )
        )
        response = await litellm.acompletion(
            model="openai/bridge-only-model",
            api_base="https://example.invalid/v1",
            api_key="x",
            messages=[{"role": "user", "content": "hi"}],
        )

    assert response.choices[0].message.content == "from upstream"
    assert responses_route.call_count == 1
    assert counter.calls == 1


@pytest.mark.parametrize(
    "bridge_model, tools",
    [
        ("gpt-5.4", [{"type": "mcp", "server_label": "docs", "server_url": "litellm_proxy"}]),
        ("gpt-5.4", [{"type": "file_search", "vector_store_ids": ["vs_1"]}]),
        ("chat_completions/gpt-5.4", None),
    ],
)
def test_completion_keeps_responses_surface_for_what_only_it_serves(bridge_model, tools):
    cached = ModelResponse(id="chatcmpl-surface", model="gpt-5.4")
    bridge = ResponsesToCompletionBridgeHandler()
    kwargs = _bridge_kwargs(stream=False)
    kwargs["model"] = bridge_model

    with (
        patch.object(
            bridge.transformation_handler,
            "transform_request",
            return_value={"model": bridge_model, "input": "hi", "tools": tools},
        ),
        patch("litellm.responses", return_value=cached) as responses_call,
        patch.object(bridge_executor, "native_responses") as executor_call,
    ):
        result = bridge.completion(**kwargs)

    assert result is cached
    assert responses_call.call_count == 1
    assert executor_call.call_count == 0


_LOGGED_ROW: Final = TypeAdapter(dict[str, object])
_LOGGED_CHAT: Final = TypeAdapter(ChatCompletion)


class _SuccessRowRecorder(CustomLogger):
    rows: tuple[dict[str, object], ...] = ()

    async def async_log_success_event(
        self, kwargs: dict[str, object], response_obj: object, start_time: datetime, end_time: datetime
    ) -> None:
        self.rows = (*self.rows, _LOGGED_ROW.validate_python(kwargs["standard_logging_object"]))


def _responses_sse(events: Sequence[Mapping[str, object]]) -> bytes:
    return "".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events).encode()


@pytest.mark.asyncio
async def test_bridged_chat_stream_logs_one_chat_row_billed_from_provider_usage(monkeypatch: pytest.MonkeyPatch):
    recorder: Final = _SuccessRowRecorder()
    monkeypatch.setattr(litellm, "callbacks", [recorder])
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    litellm.in_memory_llm_clients_cache.flush_cache()
    prices: Final = {"input_cost_per_token": 1e-6, "output_cost_per_token": 4e-6, "cache_read_input_token_cost": 1e-7}
    litellm.register_model(
        {
            "openai/bridge-stream-model": {
                "litellm_provider": "openai",
                "mode": "responses",
                "max_tokens": 100,
                **prices,
            }
        }
    )
    usage: Final[dict[str, object]] = {
        "input_tokens": 100,
        "input_tokens_details": {"cached_tokens": 40},
        "output_tokens": 20,
        "output_tokens_details": {"reasoning_tokens": 5},
        "total_tokens": 120,
    }
    message: Final[dict[str, object]] = {
        "type": "message",
        "id": "msg_stream",
        "status": "completed",
        "role": "assistant",
        "content": [{"type": "output_text", "text": "streamed", "annotations": []}],
    }
    response: Final[dict[str, object]] = {
        "id": "resp_stream",
        "object": "response",
        "created_at": 1,
        "model": "bridge-stream-model",
        "output": [],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
    }
    sse: Final = _responses_sse(
        [
            {"type": "response.created", "sequence_number": 0, "response": {**response, "status": "in_progress"}},
            {"type": "response.output_item.added", "sequence_number": 1, "output_index": 0, "item": message},
            {
                "type": "response.output_text.delta",
                "sequence_number": 2,
                "item_id": "msg_stream",
                "output_index": 0,
                "content_index": 0,
                "delta": "streamed",
            },
            {"type": "response.output_item.done", "sequence_number": 3, "output_index": 0, "item": message},
            {
                "type": "response.completed",
                "sequence_number": 4,
                "response": {**response, "status": "completed", "output": [message], "usage": usage},
            },
        ]
    )

    with respx.mock(assert_all_called=False) as upstream:
        responses_route = upstream.post("https://example.invalid/v1/responses").mock(
            return_value=httpx.Response(200, headers={"content-type": "text/event-stream"}, content=sse)
        )
        stream = await litellm.acompletion(
            model="openai/bridge-stream-model",
            api_base="https://example.invalid/v1",
            api_key="x",
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
        )
        assert isinstance(stream, CustomStreamWrapper)
        text = "".join([chunk.choices[0].delta.content or "" async for chunk in stream if chunk.choices])
        for _ in range(100):
            if recorder.rows:
                break
            await asyncio.sleep(0.01)
        await asyncio.sleep(0.1)

    assert text == "streamed"
    assert responses_route.call_count == 1
    assert len(recorder.rows) == 1
    row: Final = recorder.rows[0]
    logged: Final = _LOGGED_CHAT.validate_python(row["response"])
    assert logged.object == "chat.completion"
    assert logged.choices[0].message.content == "streamed"
    assert row["prompt_tokens"] == 100
    assert row["completion_tokens"] == 20
    cost: Final = row["response_cost"]
    assert isinstance(cost, float)
    assert math.isclose(
        cost,
        60 * prices["input_cost_per_token"]
        + 40 * prices["cache_read_input_token_cost"]
        + 20 * prices["output_cost_per_token"],
    )
