"""
Integration tests for async_post_call_streaming_hook.

Tests verify that the streaming hook can transform streaming responses sent to clients.
"""

import os
import sys
import pytest
from typing import Any
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath("../../../.."))

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.utils import ModelResponseStream, StreamingChoices, Delta


class StreamingResponseTransformerLogger(CustomLogger):
    """Logger that transforms streaming responses"""

    def __init__(self, transform_content: str = None):
        self.called = False
        self.transform_content = transform_content
        self.received_response = None

    async def async_post_call_streaming_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        response: str,
    ) -> Any:
        self.called = True
        self.received_response = response
        if self.transform_content is not None:
            return self.transform_content
        return None


@pytest.mark.asyncio
async def test_streaming_hook_transforms_response():
    """
    Test that async_post_call_streaming_hook can transform streaming responses.
    """
    transformer = StreamingResponseTransformerLogger(transform_content="Modified streaming response")

    with patch("litellm.callbacks", [transformer]):
        from litellm.proxy.utils import ProxyLogging
        from litellm.caching.caching import DualCache

        proxy_logging = ProxyLogging(user_api_key_cache=DualCache())

        # Create a mock streaming response
        original_response = ModelResponseStream(
            id="original-stream",
            choices=[
                StreamingChoices(
                    delta=Delta(content="Original content", role="assistant"),
                    index=0,
                )
            ],
            model="test-model",
        )

        data = {"model": "test-model"}
        user_api_key_dict = UserAPIKeyAuth(api_key="test-key")

        # Call the hook
        result = await proxy_logging.async_post_call_streaming_hook(
            data=data,
            response=original_response,
            user_api_key_dict=user_api_key_dict,
        )

        # Verify hook was called
        assert transformer.called is True

        # Verify transformed response is returned
        assert result == "Modified streaming response"


@pytest.mark.asyncio
async def test_streaming_hook_returns_none_keeps_original():
    """
    Test that hook returning None keeps the original response.
    """

    class NoOpLogger(CustomLogger):
        def __init__(self):
            self.called = False

        async def async_post_call_streaming_hook(
            self,
            user_api_key_dict: UserAPIKeyAuth,
            response: str,
        ):
            self.called = True
            return None

    logger = NoOpLogger()

    with patch("litellm.callbacks", [logger]):
        from litellm.proxy.utils import ProxyLogging
        from litellm.caching.caching import DualCache

        proxy_logging = ProxyLogging(user_api_key_cache=DualCache())

        original_response = ModelResponseStream(
            id="original-stream",
            choices=[
                StreamingChoices(
                    delta=Delta(content="Original content", role="assistant"),
                    index=0,
                )
            ],
            model="test-model",
        )

        data = {"model": "test-model"}
        user_api_key_dict = UserAPIKeyAuth(api_key="test-key")

        result = await proxy_logging.async_post_call_streaming_hook(
            data=data,
            response=original_response,
            user_api_key_dict=user_api_key_dict,
        )

        # Should return original response object
        assert result.id == "original-stream"
        assert logger.called is True


@pytest.mark.asyncio
async def test_streaming_hook_works_with_sse_format():
    """
    Test that hook works with SSE-formatted strings (data: prefix).
    This was the only supported format before the fix.
    """
    transformer = StreamingResponseTransformerLogger(
        transform_content="data: {\"error\": \"custom error\"}\n\n"
    )

    with patch("litellm.callbacks", [transformer]):
        from litellm.proxy.utils import ProxyLogging
        from litellm.caching.caching import DualCache

        proxy_logging = ProxyLogging(user_api_key_cache=DualCache())

        original_response = ModelResponseStream(
            id="original-stream",
            choices=[
                StreamingChoices(
                    delta=Delta(content="Original content", role="assistant"),
                    index=0,
                )
            ],
            model="test-model",
        )

        data = {"model": "test-model"}
        user_api_key_dict = UserAPIKeyAuth(api_key="test-key")

        result = await proxy_logging.async_post_call_streaming_hook(
            data=data,
            response=original_response,
            user_api_key_dict=user_api_key_dict,
        )

        # Verify SSE-formatted response is returned
        assert result == "data: {\"error\": \"custom error\"}\n\n"


@pytest.mark.asyncio
async def test_streaming_hook_chains_multiple_callbacks():
    """
    Test that multiple callbacks can chain modifications.
    """

    class AppendLogger(CustomLogger):
        def __init__(self, suffix: str):
            self.suffix = suffix
            self.called = False

        async def async_post_call_streaming_hook(
            self,
            user_api_key_dict: UserAPIKeyAuth,
            response: str,
        ) -> str:
            self.called = True
            # Note: response here is the complete_response string, not the chunk
            return f"[{self.suffix}]"

    callback1 = AppendLogger("CB1")
    callback2 = AppendLogger("CB2")

    with patch("litellm.callbacks", [callback1, callback2]):
        from litellm.proxy.utils import ProxyLogging
        from litellm.caching.caching import DualCache

        proxy_logging = ProxyLogging(user_api_key_cache=DualCache())

        original_response = ModelResponseStream(
            id="original-stream",
            choices=[
                StreamingChoices(
                    delta=Delta(content="Hello", role="assistant"),
                    index=0,
                )
            ],
            model="test-model",
        )

        data = {"model": "test-model"}
        user_api_key_dict = UserAPIKeyAuth(api_key="test-key")

        result = await proxy_logging.async_post_call_streaming_hook(
            data=data,
            response=original_response,
            user_api_key_dict=user_api_key_dict,
        )

        # Both callbacks should have been called
        assert callback1.called is True
        assert callback2.called is True

        # Last callback's result should be used
        assert result == "[CB2]"


@pytest.mark.asyncio
async def test_streaming_hook_handles_exceptions():
    """
    Test that hook exceptions are propagated.
    """

    class FailingLogger(CustomLogger):
        async def async_post_call_streaming_hook(
            self,
            user_api_key_dict: UserAPIKeyAuth,
            response: str,
        ):
            raise RuntimeError("Streaming hook crashed!")

    logger = FailingLogger()

    with patch("litellm.callbacks", [logger]):
        from litellm.proxy.utils import ProxyLogging
        from litellm.caching.caching import DualCache

        proxy_logging = ProxyLogging(user_api_key_cache=DualCache())

        original_response = ModelResponseStream(
            id="original-stream",
            choices=[
                StreamingChoices(
                    delta=Delta(content="Hello", role="assistant"),
                    index=0,
                )
            ],
            model="test-model",
        )

        data = {"model": "test-model"}
        user_api_key_dict = UserAPIKeyAuth(api_key="test-key")

        # Exception should be propagated
        with pytest.raises(RuntimeError, match="Streaming hook crashed!"):
            await proxy_logging.async_post_call_streaming_hook(
                data=data,
                response=original_response,
                user_api_key_dict=user_api_key_dict,
            )
