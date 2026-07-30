"""
Tests for the response id emitted by the chat-completions -> Responses API streaming bridge.

Every streaming event has to carry the same litellm-encoded id (provider + deployment id), so a
client that reads the id off `response.created` can send it back as `previous_response_id` and
still get routed to the deployment that served the session.
"""

from unittest.mock import AsyncMock

from litellm.responses.litellm_completion_transformation.streaming_iterator import (
    LiteLLMCompletionStreamingIterator,
)
from litellm.responses.utils import ResponsesAPIRequestUtils
from litellm.types.utils import Choices, Message, ModelResponse


def _build_iterator() -> LiteLLMCompletionStreamingIterator:
    return LiteLLMCompletionStreamingIterator(
        model="test-model",
        litellm_custom_stream_wrapper=AsyncMock(),
        request_input="Test input",
        responses_api_request={},
        custom_llm_provider="anthropic",
        litellm_metadata={"model_info": {"id": "deployment-123"}},
    )


def test_response_created_event_id_is_encoded_with_the_deployment_id():
    iterator = _build_iterator()

    created_event = iterator.create_response_created_event()

    decoded = ResponsesAPIRequestUtils._decode_responses_api_response_id(created_event.response.id)
    assert decoded["model_id"] == "deployment-123"
    assert decoded["custom_llm_provider"] == "anthropic"


def test_streaming_events_all_share_the_completed_event_id():
    iterator = _build_iterator()

    created_event = iterator.create_response_created_event()
    in_progress_event = iterator.create_response_in_progress_event()
    completed_event = iterator._emit_response_completed_event(
        ModelResponse(
            id="chatcmpl-1",
            choices=[Choices(finish_reason="stop", index=0, message=Message(role="assistant", content="hi"))],
            model="test-model",
        )
    )

    assert completed_event is not None
    assert created_event.response.id == completed_event.response.id
    assert in_progress_event.response.id == completed_event.response.id
