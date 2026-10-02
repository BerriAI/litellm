from unittest.mock import AsyncMock

import pytest

from litellm.responses.litellm_completion_transformation.streaming_iterator import (
    LiteLLMCompletionStreamingIterator,
)
from litellm.responses.litellm_completion_transformation.transformation import (
    LiteLLMCompletionResponsesConfig,
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
