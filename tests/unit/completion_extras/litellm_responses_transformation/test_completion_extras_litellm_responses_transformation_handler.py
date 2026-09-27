import json
from collections.abc import Iterator
from datetime import datetime
from itertools import chain
from typing import Final
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from openai.types.responses import ResponseOutputItemDoneEvent
from pydantic import JsonValue
from typing_extensions import ReadOnly, TypedDict

import litellm
from litellm.completion_extras.litellm_responses_transformation.handler import (
    ResponsesToCompletionBridgeHandler,
)
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLogging
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig
from litellm.responses.streaming_iterator import ResponsesAPIStreamingIterator, SyncResponsesAPIStreamingIterator
from litellm.types.llms.openai import ResponsesAPIResponse
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


def test_completion_returns_cached_model_response_directly():
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
async def test_acompletion_returns_cached_model_response_directly():
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


def test_completion_skips_rewrapping_preformatted_cached_chat_stream():
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


def test_completion_preserves_top_level_stream_flag_in_responses_request():
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
async def test_acompletion_skips_rewrapping_preformatted_cached_chat_stream():
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
async def test_acompletion_preserves_top_level_stream_flag_in_responses_request():
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


def _empty_responses_response() -> ResponsesAPIResponse:
    return ResponsesAPIResponse.model_construct(output=[], error=None)


def _output_item_done_event() -> dict[str, JsonValue]:
    return {
        "type": "response.output_item.done",
        "output_index": 0,
        "item": {
            "type": "message",
            "id": "msg_from_stream",
            "role": "assistant",
            "status": "completed",
            "content": [
                {
                    "type": "output_text",
                    "text": "Recovered from stream",
                    "annotations": [],
                }
            ],
        },
    }


def _output_text_done_event() -> dict[str, JsonValue]:
    return {
        "type": "response.output_text.done",
        "sequence_number": 1,
        "item_id": "msg_from_stream",
        "output_index": 0,
        "content_index": 0,
        "text": "Recovered from stream",
    }


def _empty_completed_event() -> dict[str, JsonValue]:
    return {
        "type": "response.completed",
        "sequence_number": 2,
        "response": {
            "id": "resp_test",
            "object": "response",
            "created_at": 0,
            "status": "completed",
            "model": "gpt-5.4",
            "output": [],
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
        },
    }


def _sse_http_response(*events: dict[str, JsonValue]) -> httpx.Response:
    body: Final = "".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events)
    return httpx.Response(200, content=body.encode(), request=httpx.Request("POST", "https://example.com/responses"))


class _StreamIteratorKwargs(TypedDict):
    response: ReadOnly[httpx.Response]
    model: ReadOnly[str]
    responses_api_provider_config: ReadOnly[OpenAIResponsesAPIConfig]
    logging_obj: ReadOnly[LiteLLMLogging]
    custom_llm_provider: ReadOnly[str]


def _real_stream_iterator_kwargs(recoverable_event: dict[str, JsonValue]) -> _StreamIteratorKwargs:
    return {
        "response": _sse_http_response(recoverable_event, _empty_completed_event()),
        "model": "gpt-5.4",
        "responses_api_provider_config": OpenAIResponsesAPIConfig(),
        "logging_obj": LiteLLMLogging(
            litellm_call_id="test-call",
            call_type="completion",
            model="gpt-5.4",
            messages=[{"role": "user", "content": "hi"}],
            function_id="fn-id",
            stream=True,
            start_time=datetime(2026, 1, 1),
        ),
        "custom_llm_provider": "chatgpt",
    }


def _recovered_texts(response: ResponsesAPIResponse) -> list[str]:
    content_parts: Final = chain.from_iterable(item["content"] for item in response.output)
    return [part["text"] for part in content_parts]


_RECOVERABLE_STREAM_EVENTS: Final = pytest.mark.parametrize(
    "recoverable_event",
    [_output_item_done_event(), _output_text_done_event()],
    ids=["output_item_done", "output_text_done"],
)


@_RECOVERABLE_STREAM_EVENTS
def test_collect_response_from_stream_recovers_output_items(recoverable_event: dict[str, JsonValue]) -> None:
    stream: Final = SyncResponsesAPIStreamingIterator(**_real_stream_iterator_kwargs(recoverable_event))

    response: Final = ResponsesToCompletionBridgeHandler()._collect_response_from_stream(stream)

    assert _recovered_texts(response) == ["Recovered from stream"]


@_RECOVERABLE_STREAM_EVENTS
@pytest.mark.asyncio
async def test_collect_response_from_async_stream_recovers_output_items(recoverable_event: dict[str, JsonValue]) -> None:
    stream: Final = ResponsesAPIStreamingIterator(**_real_stream_iterator_kwargs(recoverable_event))

    response: Final = await ResponsesToCompletionBridgeHandler()._collect_response_from_stream_async(stream)

    assert _recovered_texts(response) == ["Recovered from stream"]


@pytest.mark.parametrize(
    "terminal_payload",
    [
        {"id": "resp_test", "created_at": 0, "output": []},
        {"output": []},
    ],
    ids=["validated-terminal", "partial-terminal"],
)
def test_recovery_accepts_sdk_events_and_dictionary_terminals(terminal_payload: dict[str, JsonValue]) -> None:
    event: Final = ResponseOutputItemDoneEvent(**_output_item_done_event(), sequence_number=1)

    response: Final = ResponsesToCompletionBridgeHandler._coerce_response_object(
        terminal_payload,
        {"headers": {"x-request-id": "req_test"}},
        (object(), event),
    )

    assert response.output == [event.item.model_dump()]
    assert response._hidden_params["headers"] == {"x-request-id": "req_test"}
    assert terminal_payload["output"] == []


def test_recovery_preserves_complete_response_without_consuming_events() -> None:
    terminal: Final = ResponsesAPIResponse.model_construct(output=[_output_item_done_event()["item"]])

    def unavailable_events() -> Iterator[object]:
        raise AssertionError("Complete terminal output must bypass event recovery")
        yield

    response: Final = ResponsesToCompletionBridgeHandler._coerce_response_object(terminal, None, unavailable_events())

    assert response is terminal
    assert response.output == [_output_item_done_event()["item"]]


def test_recovery_keeps_empty_terminal_when_no_output_can_be_recovered() -> None:
    terminal: Final = _empty_responses_response()

    response: Final = ResponsesToCompletionBridgeHandler._coerce_response_object(terminal, None, (object(),))

    assert response is terminal
    assert response.output == []


@pytest.mark.asyncio
async def test_acompletion_streams_completed_model_response():
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


def test_completion_streams_completed_model_response():
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
    """responses() resolves the provider itself, so the bridge must not hand it an already-stripped model."""
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
        patch("litellm.responses", return_value=cached) as responses_call,
    ):
        bridge.completion(**kwargs)

    handed_model = responses_call.call_args.kwargs["model"]
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
        patch("litellm.aresponses", new=AsyncMock(return_value=cached)) as responses_call,
    ):
        await bridge.acompletion(**kwargs)

    handed_model = responses_call.call_args.kwargs["model"]
    assert _upstream_model_for(handed_model, custom_llm_provider) == expected_upstream_model
