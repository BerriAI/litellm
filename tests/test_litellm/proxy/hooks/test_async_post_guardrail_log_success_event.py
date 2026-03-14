"""
Integration tests for async_post_guardrail_log_success_event.

Tests verify that the post-guardrail log success hook is invoked after
post_call_success_hook so callbacks can log the final (e.g. guardrail-modified)
response. Same patterns as test_post_call_success_hook_integration.py.
"""

import os
import sys
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../../../.."))

from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.utils import Choices, Message, ModelResponse, Usage


class PostGuardrailLogCaptureLogger(CustomLogger):
    """Logger that implements async_post_guardrail_log_success_event and captures args."""

    def __init__(self):
        self.called = False
        self.kwargs = None
        self.response_obj = None
        self.start_time = None
        self.end_time = None

    async def async_post_guardrail_log_success_event(
        self, kwargs, response_obj, start_time, end_time
    ):
        self.called = True
        self.kwargs = kwargs
        self.response_obj = response_obj
        self.start_time = start_time
        self.end_time = end_time


@pytest.mark.asyncio
async def test_post_guardrail_log_success_event_called_with_response():
    """
    Test that async_post_guardrail_log_success_event is called with the given response.
    """
    logger = PostGuardrailLogCaptureLogger()

    with patch("litellm.callbacks", [logger]):
        from litellm.caching.caching import DualCache
        from litellm.proxy.utils import ProxyLogging

        proxy_logging = ProxyLogging(user_api_key_cache=DualCache())

        response = ModelResponse(
            id="post-guardrail-response",
            choices=[
                Choices(
                    message=Message(content="Post-guardrail content", role="assistant"),
                    index=0,
                    finish_reason="stop",
                )
            ],
            model="test-model",
            usage=Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
        )
        data = {"model": "test-model", "messages": []}
        user_api_key_dict = UserAPIKeyAuth(api_key="test-key")

        await proxy_logging.async_post_guardrail_log_success_event(
            data=data,
            response=response,
            user_api_key_dict=user_api_key_dict,
            logging_obj=None,
        )

        assert logger.called is True
        assert logger.response_obj is response
        assert logger.response_obj.id == "post-guardrail-response"
        assert logger.kwargs["model"] == "test-model"
        assert logger.kwargs["user_api_key_dict"] is user_api_key_dict


@pytest.mark.asyncio
async def test_post_guardrail_log_success_event_invokes_noop_and_custom_implementations():
    """
    Test that callbacks using the no-op base implementation don't error, and callbacks
    with custom implementations receive the response correctly.
    """
    capture_logger = PostGuardrailLogCaptureLogger()
    no_op_logger = CustomLogger()  # base no-op async_post_guardrail_log_success_event

    with patch("litellm.callbacks", [no_op_logger, capture_logger]):
        from litellm.caching.caching import DualCache
        from litellm.proxy.utils import ProxyLogging

        proxy_logging = ProxyLogging(user_api_key_cache=DualCache())

        response = ModelResponse(
            id="response",
            choices=[
                Choices(
                    message=Message(content="Content", role="assistant"),
                    index=0,
                    finish_reason="stop",
                )
            ],
            model="test-model",
            usage=Usage(prompt_tokens=5, completion_tokens=10, total_tokens=15),
        )
        data = {"model": "test-model"}
        user_api_key_dict = UserAPIKeyAuth(api_key="key")

        await proxy_logging.async_post_guardrail_log_success_event(
            data=data,
            response=response,
            user_api_key_dict=user_api_key_dict,
            logging_obj=None,
        )

        assert capture_logger.called is True
        assert capture_logger.response_obj is response


@pytest.mark.asyncio
async def test_post_guardrail_log_success_event_exception_in_callback_does_not_break_others():
    """
    Test that when one callback raises in async_post_guardrail_log_success_event,
    other callbacks are still invoked (exceptions are caught and logged).
    """
    failing_logger = PostGuardrailLogCaptureLogger()

    async def raise_in_hook(*args, **kwargs):
        failing_logger.called = True
        raise RuntimeError("Intentional failure in post-guardrail log")

    failing_logger.async_post_guardrail_log_success_event = raise_in_hook

    success_logger = PostGuardrailLogCaptureLogger()

    with patch("litellm.callbacks", [failing_logger, success_logger]):
        from litellm.caching.caching import DualCache
        from litellm.proxy.utils import ProxyLogging

        proxy_logging = ProxyLogging(user_api_key_cache=DualCache())

        response = ModelResponse(
            id="response",
            choices=[
                Choices(
                    message=Message(content="Content", role="assistant"),
                    index=0,
                    finish_reason="stop",
                )
            ],
            model="test-model",
            usage=Usage(prompt_tokens=5, completion_tokens=10, total_tokens=15),
        )
        data = {"model": "test-model"}
        user_api_key_dict = UserAPIKeyAuth(api_key="key")

        await proxy_logging.async_post_guardrail_log_success_event(
            data=data,
            response=response,
            user_api_key_dict=user_api_key_dict,
            logging_obj=None,
        )

        assert failing_logger.called is True
        assert success_logger.called is True
        assert success_logger.response_obj is response


@pytest.mark.asyncio
async def test_post_guardrail_log_success_event_receives_logging_obj_timing():
    """
    Test that when logging_obj is provided with model_call_details / completion_start_time,
    start_time and end_time are passed to the hook.
    """
    logger = PostGuardrailLogCaptureLogger()
    start = datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
    end = datetime(2025, 1, 15, 12, 0, 5, tzinfo=timezone.utc)

    logging_obj = MagicMock()
    logging_obj.model_call_details = {"start_time": start, "end_time": end}
    logging_obj.completion_start_time = start

    with patch("litellm.callbacks", [logger]):
        from litellm.caching.caching import DualCache
        from litellm.proxy.utils import ProxyLogging

        proxy_logging = ProxyLogging(user_api_key_cache=DualCache())

        response = ModelResponse(
            id="response",
            choices=[
                Choices(
                    message=Message(content="Content", role="assistant"),
                    index=0,
                    finish_reason="stop",
                )
            ],
            model="test-model",
            usage=Usage(prompt_tokens=5, completion_tokens=10, total_tokens=15),
        )
        data = {"model": "test-model"}
        user_api_key_dict = UserAPIKeyAuth(api_key="key")

        await proxy_logging.async_post_guardrail_log_success_event(
            data=data,
            response=response,
            user_api_key_dict=user_api_key_dict,
            logging_obj=logging_obj,
        )

        assert logger.called is True
        assert logger.start_time is start
        assert logger.end_time is end
        assert logger.kwargs.get("start_time") == start
        assert logger.kwargs.get("end_time") == end


@pytest.mark.asyncio
async def test_post_guardrail_log_success_event_skips_guardrail_callbacks():
    """
    Test that CustomGuardrail callbacks are not invoked for async_post_guardrail_log_success_event
    (only CustomLogger callbacks are).
    """

    class GuardrailWithLogHook(CustomGuardrail):
        def __init__(self):
            self.called = False

        async def async_post_guardrail_log_success_event(
            self, kwargs, response_obj, start_time, end_time
        ):
            self.called = True

    guardrail = GuardrailWithLogHook()
    logger = PostGuardrailLogCaptureLogger()

    with patch("litellm.callbacks", [guardrail, logger]):
        from litellm.caching.caching import DualCache
        from litellm.proxy.utils import ProxyLogging

        proxy_logging = ProxyLogging(user_api_key_cache=DualCache())

        response = ModelResponse(
            id="response",
            choices=[
                Choices(
                    message=Message(content="Content", role="assistant"),
                    index=0,
                    finish_reason="stop",
                )
            ],
            model="test-model",
            usage=Usage(prompt_tokens=5, completion_tokens=10, total_tokens=15),
        )
        data = {"model": "test-model"}
        user_api_key_dict = UserAPIKeyAuth(api_key="key")

        await proxy_logging.async_post_guardrail_log_success_event(
            data=data,
            response=response,
            user_api_key_dict=user_api_key_dict,
            logging_obj=None,
        )

        assert guardrail.called is False
        assert logger.called is True
