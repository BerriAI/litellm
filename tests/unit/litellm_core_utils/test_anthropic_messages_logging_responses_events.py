"""
Unit tests for _handle_anthropic_messages_response_logging guard against
ResponseIncompleteEvent / ResponseCompletedEvent (issue #27091).

These cover the regression: when the experimental Anthropic pass-through routes
through the Responses API backend the success handler receives a Responses API
event object, not a plain dict or AnthropicResponse. Before the fix, Pydantic
raised a ValidationError in background logging.
"""
from unittest.mock import MagicMock, patch
import pytest


def _make_logging_obj(stream: bool = False):
    """Build a minimal Logging instance without touching live infrastructure."""
    from litellm.litellm_core_utils.litellm_logging import Logging
    from litellm.utils import CallTypes

    obj = Logging.__new__(Logging)
    obj.stream = stream
    obj.call_type = CallTypes.anthropic_messages.value
    obj.model = "claude-sonnet-4-6"
    obj.model_call_details = {}  # no httpx_response stored
    return obj


def _make_incomplete_event():
    """Return a minimal ResponseIncompleteEvent with a stub .response."""
    from litellm.types.llms.openai import ResponseIncompleteEvent

    stub_response = MagicMock()
    stub_response.status = "incomplete"
    return ResponseIncompleteEvent(
        type="response.incomplete",
        response=stub_response,
    )


def _make_completed_event():
    """Return a minimal ResponseCompletedEvent with a stub .response."""
    from litellm.types.llms.openai import ResponseCompletedEvent

    stub_response = MagicMock()
    stub_response.status = "completed"
    return ResponseCompletedEvent(
        type="response.completed",
        response=stub_response,
    )


class TestHandleAnthropicMessagesLoggingResponsesEvents:
    """
    _handle_anthropic_messages_response_logging must not raise a ValidationError
    when it receives a Responses API event (ResponseIncompleteEvent,
    ResponseCompletedEvent, or any unexpected type).
    """

    def test_incomplete_event_does_not_raise(self):
        """ResponseIncompleteEvent must not trigger AnthropicResponse.model_validate."""
        obj = _make_logging_obj()
        event = _make_incomplete_event()
        # Should return without raising ValidationError
        result = obj._handle_anthropic_messages_response_logging(result=event)
        # Result passes through unchanged
        assert result is event

    def test_completed_event_does_not_raise(self):
        """ResponseCompletedEvent must not trigger AnthropicResponse.model_validate."""
        obj = _make_logging_obj()
        event = _make_completed_event()
        result = obj._handle_anthropic_messages_response_logging(result=event)
        assert result is event

    def test_unknown_type_does_not_raise(self):
        """Arbitrary non-dict, non-AnthropicResponse types must not crash."""
        obj = _make_logging_obj()
        random_obj = object()
        # Should not raise
        result = obj._handle_anthropic_messages_response_logging(result=random_obj)
        assert result is random_obj

    def test_model_response_still_returned_as_is(self):
        """ModelResponse inputs should continue to pass through unchanged."""
        from litellm import ModelResponse

        obj = _make_logging_obj()
        mr = ModelResponse(model="claude-sonnet-4-6")
        result = obj._handle_anthropic_messages_response_logging(result=mr)
        assert isinstance(result, ModelResponse)

    def test_streaming_model_response_returned_as_is(self):
        """Streaming ModelResponse should still short-circuit at the first guard."""
        from litellm import ModelResponse

        obj = _make_logging_obj(stream=True)
        mr = ModelResponse(model="claude-sonnet-4-6")
        result = obj._handle_anthropic_messages_response_logging(result=mr)
        assert isinstance(result, ModelResponse)

    def test_debug_log_emitted_for_incomplete_event(self, caplog):
        """A debug message must be emitted when a Responses API event is encountered."""
        import logging

        obj = _make_logging_obj()
        event = _make_incomplete_event()
        with caplog.at_level(logging.DEBUG, logger="LiteLLM"):
            obj._handle_anthropic_messages_response_logging(result=event)
        assert any(
            "ResponseIncompleteEvent" in rec.message for rec in caplog.records
        ), "Expected debug log mentioning ResponseIncompleteEvent"
