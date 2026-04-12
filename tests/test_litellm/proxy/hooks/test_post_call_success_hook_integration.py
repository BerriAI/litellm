"""
Integration tests for async_post_call_success_hook.

Tests verify that the success hook can transform responses sent to clients.
This mirrors the behavior of CustomGuardrail hooks and streaming iterator hooks.
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
from litellm.types.utils import ModelResponse, Choices, Message, Usage


class ResponseTransformerLogger(CustomLogger):
    """Logger that transforms successful responses"""

    def __init__(self, transform_content: str = None):
        self.called = False
        self.transform_content = transform_content

    async def async_post_call_success_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        response: Any,
    ) -> Any:
        self.called = True
        if self.transform_content is not None:
            # Create a modified response with custom content
            return {
                "id": "transformed-response",
                "choices": [
                    {
                        "message": {"content": self.transform_content, "role": "assistant"},
                        "index": 0,
                    }
                ],
                "model": "test-model",
                "custom_field": "added_by_hook",
            }
        return response


@pytest.mark.asyncio
async def test_success_hook_transforms_response():
    """
    Test that async_post_call_success_hook can transform successful responses.
    """
    transformer = ResponseTransformerLogger(transform_content="Modified response")

    with patch("litellm.callbacks", [transformer]):
        from litellm.proxy.utils import ProxyLogging
        from litellm.caching.caching import DualCache

        proxy_logging = ProxyLogging(user_api_key_cache=DualCache())

        # Create a mock response
        original_response = ModelResponse(
            id="original-response",
            choices=[
                Choices(
                    message=Message(content="Original content", role="assistant"),
                    index=0,
                    finish_reason="stop",
                )
            ],
            model="test-model",
            usage=Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
        )

        data = {"model": "test-model"}
        user_api_key_dict = UserAPIKeyAuth(api_key="test-key")

        # Call the hook
        result = await proxy_logging.post_call_success_hook(
            data=data,
            response=original_response,
            user_api_key_dict=user_api_key_dict,
        )

        # Verify hook was called
        assert transformer.called is True

        # Verify transformed response is returned
        assert result is not None
        assert result["id"] == "transformed-response"
        assert result["choices"][0]["message"]["content"] == "Modified response"
        assert result["custom_field"] == "added_by_hook"


@pytest.mark.asyncio
async def test_success_hook_returns_none_keeps_original():
    """
    Test that hook returning None keeps the original response.
    """

    class NoOpLogger(CustomLogger):
        def __init__(self):
            self.called = False

        async def async_post_call_success_hook(self, *args, **kwargs):
            self.called = True
            return None

    logger = NoOpLogger()

    with patch("litellm.callbacks", [logger]):
        from litellm.proxy.utils import ProxyLogging
        from litellm.caching.caching import DualCache

        proxy_logging = ProxyLogging(user_api_key_cache=DualCache())

        original_response = ModelResponse(
            id="original-response",
            choices=[
                Choices(
                    message=Message(content="Original content", role="assistant"),
                    index=0,
                    finish_reason="stop",
                )
            ],
            model="test-model",
            usage=Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
        )

        data = {"model": "test-model"}
        user_api_key_dict = UserAPIKeyAuth(api_key="test-key")

        result = await proxy_logging.post_call_success_hook(
            data=data,
            response=original_response,
            user_api_key_dict=user_api_key_dict,
        )

        # Should return original response
        assert result.id == "original-response"
        assert logger.called is True


@pytest.mark.asyncio
async def test_success_hook_chains_multiple_callbacks():
    """
    Test that multiple callbacks can chain modifications.
    """

    class AddFieldLogger(CustomLogger):
        def __init__(self, field_name: str, field_value: Any):
            self.field_name = field_name
            self.field_value = field_value
            self.called = False

        async def async_post_call_success_hook(
            self,
            data: dict,
            user_api_key_dict: UserAPIKeyAuth,
            response: Any,
        ) -> Any:
            self.called = True
            # Convert response to dict if needed
            if hasattr(response, "model_dump"):
                resp_dict = response.model_dump()
            elif hasattr(response, "dict"):
                resp_dict = response.dict()
            elif isinstance(response, dict):
                resp_dict = response.copy()
            else:
                resp_dict = {}

            resp_dict[self.field_name] = self.field_value
            return resp_dict

    callback1 = AddFieldLogger("field1", "value1")
    callback2 = AddFieldLogger("field2", "value2")

    with patch("litellm.callbacks", [callback1, callback2]):
        from litellm.proxy.utils import ProxyLogging
        from litellm.caching.caching import DualCache

        proxy_logging = ProxyLogging(user_api_key_cache=DualCache())

        original_response = ModelResponse(
            id="original-response",
            choices=[
                Choices(
                    message=Message(content="Original content", role="assistant"),
                    index=0,
                    finish_reason="stop",
                )
            ],
            model="test-model",
            usage=Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
        )

        data = {"model": "test-model"}
        user_api_key_dict = UserAPIKeyAuth(api_key="test-key")

        result = await proxy_logging.post_call_success_hook(
            data=data,
            response=original_response,
            user_api_key_dict=user_api_key_dict,
        )

        # Both callbacks should have been called
        assert callback1.called is True
        assert callback2.called is True

        # Both fields should be present (chained modifications)
        assert result["field1"] == "value1"
        assert result["field2"] == "value2"


@pytest.mark.asyncio
async def test_success_hook_handles_exceptions():
    """
    Test that hook exceptions are propagated (not silently swallowed).
    """

    class FailingLogger(CustomLogger):
        async def async_post_call_success_hook(self, *args, **kwargs):
            raise RuntimeError("Hook crashed!")

    logger = FailingLogger()

    with patch("litellm.callbacks", [logger]):
        from litellm.proxy.utils import ProxyLogging
        from litellm.caching.caching import DualCache

        proxy_logging = ProxyLogging(user_api_key_cache=DualCache())

        original_response = ModelResponse(
            id="original-response",
            choices=[
                Choices(
                    message=Message(content="Original content", role="assistant"),
                    index=0,
                    finish_reason="stop",
                )
            ],
            model="test-model",
            usage=Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
        )

        data = {"model": "test-model"}
        user_api_key_dict = UserAPIKeyAuth(api_key="test-key")

        # Exception should be propagated
        with pytest.raises(RuntimeError, match="Hook crashed!"):
            await proxy_logging.post_call_success_hook(
                data=data,
                response=original_response,
                user_api_key_dict=user_api_key_dict,
            )
