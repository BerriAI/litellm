import asyncio

from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.responses.litellm_completion_transformation.streaming_iterator import (
    LiteLLMCompletionStreamingIterator,
)
from litellm.responses.litellm_completion_transformation.transformation import (
    LiteLLMCompletionResponsesConfig,
)
from litellm.responses.streaming_iterator import (
    CachedResponsesAPIStreamingIterator,
    _build_synthetic_response_events,
)
from litellm.types.llms.openai import ResponsesAPIStreamEvents
from litellm.types.utils import Choices, Message, ModelResponse, Usage


def _chat_response(finish_reason: str) -> ModelResponse:
    choice = Choices(index=0, finish_reason="stop", message=Message(role="assistant", content="cut off"))
    choice.finish_reason = finish_reason
    return ModelResponse(
        id="chatcmpl-1",
        model="some-model",
        choices=[choice],
        usage=Usage(prompt_tokens=10, completion_tokens=60, total_tokens=70),
    )


def _transform(finish_reason: str, request: dict):
    return LiteLLMCompletionResponsesConfig.transform_chat_completion_response_to_responses_api_response(
        request_input="hi",
        responses_api_request=request,
        chat_completion_response=_chat_response(finish_reason),
    )


@pytest.mark.parametrize(
    "finish_reason, reason",
    [("length", "max_output_tokens"), ("content_filter", "content_filter")],
)
def test_truncated_finish_reason_sets_incomplete_details(finish_reason, reason):
    result = _transform(finish_reason, {})
    assert result.status == "incomplete"
    assert result.incomplete_details is not None
    assert result.incomplete_details.reason == reason


def test_stop_finish_reason_has_no_incomplete_details():
    result = _transform("stop", {})
    assert result.status == "completed"
    assert result.incomplete_details is None


def test_request_sampling_params_are_echoed():
    result = _transform("stop", {"temperature": 0.3, "top_p": 0.9, "max_output_tokens": 60})
    assert (result.temperature, result.top_p, result.max_output_tokens) == (0.3, 0.9, 60)


@pytest.mark.parametrize(
    "finish_reason, event_type",
    [
        ("length", ResponsesAPIStreamEvents.RESPONSE_INCOMPLETE),
        ("stop", ResponsesAPIStreamEvents.RESPONSE_COMPLETED),
    ],
)
def test_stream_terminal_event_follows_status(finish_reason, event_type):
    iterator = LiteLLMCompletionStreamingIterator(
        model="some-model",
        litellm_custom_stream_wrapper=AsyncMock(),
        request_input="hi",
        responses_api_request={},
    )
    event = iterator._emit_response_completed_event(_chat_response(finish_reason))
    assert event is not None
    assert event.type == event_type


@pytest.mark.parametrize(
    "finish_reason, event_type",
    [
        ("length", ResponsesAPIStreamEvents.RESPONSE_INCOMPLETE),
        ("stop", ResponsesAPIStreamEvents.RESPONSE_COMPLETED),
    ],
)
def test_replayed_stream_terminal_event_follows_status(finish_reason, event_type):
    events = _build_synthetic_response_events(
        transformed=_transform(finish_reason, {}),
        logging_obj=None,
        chunk_size=10,
    )
    assert events[-1].type == event_type


def test_omitted_temperature_defaults_to_zero():
    assert _transform("stop", {}).temperature == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("finish_reason", ["length", "stop"])
async def test_replayed_stream_logs_success_exactly_once(finish_reason):
    logging_obj = MagicMock()
    logging_obj.dispatch_success_handlers = AsyncMock()
    iterator = CachedResponsesAPIStreamingIterator(
        response=_transform(finish_reason, {}),
        logging_obj=logging_obj,
    )
    async for _ in iterator:
        pass
    await asyncio.sleep(0)
    assert logging_obj.dispatch_success_handlers.await_count == 1
