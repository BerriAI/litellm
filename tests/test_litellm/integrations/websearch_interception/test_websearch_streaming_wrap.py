"""
Unit tests for fake-stream wrapping of agentic loop responses in
BaseLLMHTTPHandler.

Tests that agentic loop responses are correctly wrapped in
FakeAnthropicMessagesStreamIterator when the original request was streaming
but converted to non-streaming for WebSearch interception.
"""

from unittest.mock import MagicMock

import pytest

from litellm.integrations.custom_logger import CustomLogger
from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler
from litellm.llms.anthropic.experimental_pass_through.messages.fake_stream_iterator import (
    FakeAnthropicMessagesStreamIterator,
)
from litellm.types.integrations.custom_logger import AgenticLoopPlan


def _anthropic_response() -> dict:
    return {
        "id": "msg_123",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "hello"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }


def _converted_stream_logging_obj(callback: CustomLogger) -> MagicMock:
    logging_obj = MagicMock()
    logging_obj.model_call_details = {"websearch_interception_converted_stream": True}
    logging_obj.dynamic_success_callbacks = [callback]
    return logging_obj


async def _run_hooks(handler: BaseLLMHTTPHandler, logging_obj: MagicMock):
    return await handler._call_agentic_completion_hooks(
        response=_anthropic_response(),
        model="claude-3-5-sonnet",
        messages=[{"role": "user", "content": "search the web"}],
        anthropic_messages_provider_config=MagicMock(),
        anthropic_messages_optional_request_params={},
        logging_obj=logging_obj,
        stream=True,
        custom_llm_provider="anthropic",
        kwargs={},
    )


class TestMaybeWrapInFakeStream:
    def setup_method(self):
        self.handler = BaseLLMHTTPHandler()

    def test_wraps_dict_when_converted_stream_flag_is_true(self):
        """When websearch_interception_converted_stream is True and response is dict, wrap it."""
        logging_obj = MagicMock()
        logging_obj.model_call_details = {
            "websearch_interception_converted_stream": True
        }

        result = self.handler._maybe_wrap_in_fake_stream(
            _anthropic_response(), logging_obj, "anthropic_messages"
        )

        assert isinstance(result, FakeAnthropicMessagesStreamIterator)

    def test_returns_response_unchanged_when_flag_is_false(self):
        """When flag is False, return response as-is."""
        logging_obj = MagicMock()
        logging_obj.model_call_details = {
            "websearch_interception_converted_stream": False
        }
        response = {"id": "msg_123", "content": []}

        result = self.handler._maybe_wrap_in_fake_stream(
            response, logging_obj, "anthropic_messages"
        )

        assert result is response

    def test_returns_response_unchanged_when_not_dict(self):
        """When response is not a dict (e.g., already a stream), return as-is."""
        logging_obj = MagicMock()
        logging_obj.model_call_details = {
            "websearch_interception_converted_stream": True
        }
        response = MagicMock()  # Not a dict

        result = self.handler._maybe_wrap_in_fake_stream(
            response, logging_obj, "anthropic_messages"
        )

        assert result is response

    def test_returns_response_when_logging_obj_is_none(self):
        """When logging_obj is None, return response as-is."""
        response = {"id": "msg_123", "content": []}

        result = self.handler._maybe_wrap_in_fake_stream(
            response, None, "anthropic_messages"
        )

        assert result is response

    def test_does_not_wrap_for_non_anthropic_surface(self):
        """Even with the flag set and a dict response, leave non-anthropic surfaces untouched."""
        logging_obj = MagicMock()
        logging_obj.model_call_details = {
            "websearch_interception_converted_stream": True
        }

        result = self.handler._maybe_wrap_in_fake_stream(
            _anthropic_response(), logging_obj, "responses"
        )

        assert isinstance(result, dict)


class _LegacyAgenticCallback(CustomLogger):
    """Triggers the agentic loop via the legacy async_run_agentic_loop path."""

    async def async_should_run_agentic_loop(
        self, response, model, messages, tools, stream, custom_llm_provider, kwargs
    ):
        return True, {"tool_calls": [{"name": "web_search"}]}

    async def async_run_agentic_loop(
        self,
        tools,
        model,
        messages,
        response,
        anthropic_messages_provider_config,
        anthropic_messages_optional_request_params,
        logging_obj,
        stream,
        kwargs,
    ):
        return _anthropic_response()


class _PlanAgenticCallback(CustomLogger):
    """Drives the plan-based paths by returning a caller-supplied plan."""

    def __init__(self, plan: AgenticLoopPlan):
        self._plan = plan

    async def async_should_run_agentic_loop(
        self, response, model, messages, tools, stream, custom_llm_provider, kwargs
    ):
        return True, {"tool_calls": [{"name": "web_search"}]}

    async def async_build_agentic_loop_plan(
        self,
        tools,
        model,
        messages,
        response,
        anthropic_messages_provider_config,
        anthropic_messages_optional_request_params,
        logging_obj,
        stream,
        kwargs,
    ):
        return self._plan


class _StubExecuteHandler(BaseLLMHTTPHandler):
    """Handler whose agentic-plan execution is stubbed to a dict so the
    _execute_anthropic_agentic_plan return path can be exercised in isolation."""

    async def _execute_anthropic_agentic_plan(self, **kwargs):
        return _anthropic_response()


class TestCallAgenticCompletionHooksWrapping:
    """Regression tests: every agentic-loop return path must wrap the dict
    response in a fake stream when the request was a converted stream."""

    def setup_method(self):
        self.handler = BaseLLMHTTPHandler()

    @pytest.mark.asyncio
    async def test_legacy_run_agentic_loop_path_wraps(self):
        logging_obj = _converted_stream_logging_obj(_LegacyAgenticCallback())

        result = await _run_hooks(self.handler, logging_obj)

        assert isinstance(result, FakeAnthropicMessagesStreamIterator)

    @pytest.mark.asyncio
    async def test_response_override_path_wraps(self):
        plan = AgenticLoopPlan(response_override=_anthropic_response())
        logging_obj = _converted_stream_logging_obj(_PlanAgenticCallback(plan))

        result = await _run_hooks(self.handler, logging_obj)

        assert isinstance(result, FakeAnthropicMessagesStreamIterator)

    @pytest.mark.asyncio
    async def test_terminate_path_wraps(self):
        plan = AgenticLoopPlan(terminate=True, stop_reason="done")
        logging_obj = _converted_stream_logging_obj(_PlanAgenticCallback(plan))

        result = await _run_hooks(self.handler, logging_obj)

        assert isinstance(result, FakeAnthropicMessagesStreamIterator)

    @pytest.mark.asyncio
    async def test_anthropic_plan_execution_path_wraps(self):
        plan = AgenticLoopPlan(run_agentic_loop=True)
        logging_obj = _converted_stream_logging_obj(_PlanAgenticCallback(plan))

        result = await _run_hooks(_StubExecuteHandler(), logging_obj)

        assert isinstance(result, FakeAnthropicMessagesStreamIterator)

    @pytest.mark.asyncio
    async def test_tail_path_wraps_when_no_loop_runs(self):
        plan = AgenticLoopPlan(run_agentic_loop=False)
        logging_obj = _converted_stream_logging_obj(_PlanAgenticCallback(plan))

        result = await _run_hooks(self.handler, logging_obj)

        assert isinstance(result, FakeAnthropicMessagesStreamIterator)

    @pytest.mark.asyncio
    async def test_legacy_path_not_wrapped_without_flag(self):
        logging_obj = _converted_stream_logging_obj(_LegacyAgenticCallback())
        logging_obj.model_call_details = {}

        result = await _run_hooks(self.handler, logging_obj)

        assert isinstance(result, dict)
