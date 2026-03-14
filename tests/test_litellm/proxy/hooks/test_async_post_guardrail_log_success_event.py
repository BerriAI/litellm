"""
Tests for async_post_guardrail_log_success_event.

The hook runs after post-call hooks (e.g. guardrails) so loggers see the final
response. Only CustomLogger subclasses that override the method are invoked;
base no-op is skipped. end_time is when the hook runs; llm_end_time in kwargs
is when the LLM call finished (pre-guardrail).
"""

import os
import sys
from datetime import datetime, timezone
from typing import Any, Optional
from unittest.mock import AsyncMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../../../.."))

import litellm
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.utils import ModelResponse


class PostGuardrailLogger(CustomLogger):
    """Logger that overrides async_post_guardrail_log_success_event."""

    def __init__(self):
        self.called = False
        self.kwargs: Optional[dict] = None
        self.response_obj: Optional[Any] = None
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None

    async def async_post_guardrail_log_success_event(
        self, kwargs, response_obj, start_time, end_time
    ):
        self.called = True
        self.kwargs = kwargs
        self.response_obj = response_obj
        self.start_time = start_time
        self.end_time = end_time


class FailingPostGuardrailLogger(CustomLogger):
    """Logger that overrides and raises."""

    async def async_post_guardrail_log_success_event(
        self, kwargs, response_obj, start_time, end_time
    ):
        raise ValueError("callback failed")


@pytest.mark.asyncio
async def test_post_guardrail_log_called_with_response_and_kwargs():
    """Override is invoked with correct response and kwargs."""
    logger = PostGuardrailLogger()
    response = ModelResponse(id="r1", choices=[], model="gpt-4")
    user_api_key_dict = UserAPIKeyAuth(api_key="test-key")
    data = {"model": "gpt-4", "messages": []}

    with patch("litellm.callbacks", [logger]):
        from litellm.proxy.utils import ProxyLogging
        from litellm.caching.caching import DualCache

        proxy_logging = ProxyLogging(user_api_key_cache=DualCache())
        await proxy_logging.async_post_guardrail_log_success_event(
            data=data,
            response=response,
            user_api_key_dict=user_api_key_dict,
        )

    assert logger.called is True
    assert logger.response_obj is response
    assert logger.kwargs is not None
    assert logger.kwargs.get("model") == "gpt-4"
    assert logger.kwargs.get("user_api_key_dict") is user_api_key_dict
    assert logger.end_time is not None


@pytest.mark.asyncio
async def test_post_guardrail_log_base_custom_logger_not_invoked():
    """Base CustomLogger (no override) is not invoked; only overriders are called."""
    base_logger = CustomLogger()
    overriding_logger = PostGuardrailLogger()
    with patch.object(
        base_logger,
        "async_post_guardrail_log_success_event",
        new_callable=AsyncMock,
    ) as mock_base_method:
        with patch("litellm.callbacks", [base_logger, overriding_logger]):
            from litellm.proxy.utils import ProxyLogging
            from litellm.caching.caching import DualCache

            proxy_logging = ProxyLogging(user_api_key_cache=DualCache())
            await proxy_logging.async_post_guardrail_log_success_event(
                data={"model": "gpt-4"},
                response=ModelResponse(id="r1", choices=[], model="gpt-4"),
                user_api_key_dict=UserAPIKeyAuth(api_key="test-key"),
            )
        mock_base_method.assert_not_called()
    assert overriding_logger.called is True


@pytest.mark.asyncio
async def test_post_guardrail_log_base_no_op_never_called_when_only_base_in_callbacks():
    """When callbacks contain only base CustomLogger (no override), the hook is never invoked."""
    with patch.object(
        CustomLogger,
        "async_post_guardrail_log_success_event",
        new_callable=AsyncMock,
    ) as mock_base:
        with patch("litellm.callbacks", [CustomLogger()]):
            from litellm.proxy.utils import ProxyLogging
            from litellm.caching.caching import DualCache

            proxy_logging = ProxyLogging(user_api_key_cache=DualCache())
            await proxy_logging.async_post_guardrail_log_success_event(
                data={"model": "gpt-4"},
                response=ModelResponse(id="r1", choices=[], model="gpt-4"),
                user_api_key_dict=UserAPIKeyAuth(api_key="test-key"),
            )
        mock_base.assert_not_called()


@pytest.mark.asyncio
async def test_post_guardrail_log_llm_end_time_in_kwargs():
    """When logging_obj has model_call_details['end_time'], kwargs get llm_end_time."""
    logger = PostGuardrailLogger()
    llm_end = datetime(2025, 3, 10, 12, 0, 0, tzinfo=timezone.utc)
    logging_obj = type("LoggingObj", (), {})()
    logging_obj.model_call_details = {"end_time": llm_end}

    with patch("litellm.callbacks", [logger]):
        from litellm.proxy.utils import ProxyLogging
        from litellm.caching.caching import DualCache

        proxy_logging = ProxyLogging(user_api_key_cache=DualCache())
        await proxy_logging.async_post_guardrail_log_success_event(
            data={"model": "gpt-4"},
            response=ModelResponse(id="r1", choices=[], model="gpt-4"),
            user_api_key_dict=UserAPIKeyAuth(api_key="test-key"),
            logging_obj=logging_obj,
        )

    assert logger.called is True
    assert logger.kwargs.get("llm_end_time") == llm_end
    assert logger.end_time is not None
    assert logger.end_time != llm_end  # end_time is "now", llm_end_time is from details


@pytest.mark.asyncio
async def test_post_guardrail_log_exception_in_one_callback_does_not_block_others():
    """One callback raising does not prevent others from being called."""
    failing = FailingPostGuardrailLogger()
    ok = PostGuardrailLogger()

    with patch("litellm.callbacks", [failing, ok]):
        from litellm.proxy.utils import ProxyLogging
        from litellm.caching.caching import DualCache

        proxy_logging = ProxyLogging(user_api_key_cache=DualCache())
        await proxy_logging.async_post_guardrail_log_success_event(
            data={"model": "gpt-4"},
            response=ModelResponse(id="r1", choices=[], model="gpt-4"),
            user_api_key_dict=UserAPIKeyAuth(api_key="test-key"),
        )

    assert ok.called is True


@pytest.mark.asyncio
async def test_post_guardrail_log_guardrail_callbacks_not_invoked():
    """CustomGuardrail callbacks are not invoked by this hook."""
    logger = PostGuardrailLogger()

    class FakeGuardrail(CustomGuardrail):
        async def async_post_guardrail_log_success_event(
            self, kwargs, response_obj, start_time, end_time
        ):
            self.post_guardrail_log_called = True  # would be set if we were called

    guardrail = FakeGuardrail()
    guardrail.post_guardrail_log_called = False

    with patch("litellm.callbacks", [guardrail, logger]):
        from litellm.proxy.utils import ProxyLogging
        from litellm.caching.caching import DualCache

        proxy_logging = ProxyLogging(user_api_key_cache=DualCache())
        await proxy_logging.async_post_guardrail_log_success_event(
            data={"model": "gpt-4"},
            response=ModelResponse(id="r1", choices=[], model="gpt-4"),
            user_api_key_dict=UserAPIKeyAuth(api_key="test-key"),
        )

    assert logger.called is True
    assert getattr(guardrail, "post_guardrail_log_called", False) is False


@pytest.mark.asyncio
async def test_post_guardrail_log_no_callbacks():
    """No callbacks does not raise."""
    with patch("litellm.callbacks", []):
        from litellm.proxy.utils import ProxyLogging
        from litellm.caching.caching import DualCache

        proxy_logging = ProxyLogging(user_api_key_cache=DualCache())
        await proxy_logging.async_post_guardrail_log_success_event(
            data={"model": "gpt-4"},
            response=ModelResponse(id="r1", choices=[], model="gpt-4"),
            user_api_key_dict=UserAPIKeyAuth(api_key="test-key"),
        )
