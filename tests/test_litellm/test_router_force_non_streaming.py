"""
Test force_non_streaming litellm_params option.

When a deployment specifies `force_non_streaming: True` in its litellm_params,
the router should override `stream=True` from the client request to `stream=False`
before calling the LLM API. This is needed when backends don't support streaming
properly (e.g., vLLM streaming doesn't emit tool_use events for Anthropic format).

See: https://github.com/BerriAI/litellm/issues/5416
"""

import os
import sys
from unittest.mock import AsyncMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../../.."))

import litellm


@pytest.fixture
def router_with_force_non_streaming():
    """Create a router with one deployment that has force_non_streaming=True."""
    return litellm.Router(
        model_list=[
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "openai/test-model",
                    "api_key": "fake-key",
                    "api_base": "http://localhost:8000",
                    "force_non_streaming": True,
                },
            }
        ],
    )


@pytest.fixture
def router_without_force_non_streaming():
    """Create a router with a normal deployment (no force_non_streaming)."""
    return litellm.Router(
        model_list=[
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "openai/test-model",
                    "api_key": "fake-key",
                    "api_base": "http://localhost:8000",
                },
            }
        ],
    )


@pytest.mark.asyncio
async def test_force_non_streaming_overrides_stream_true(
    router_with_force_non_streaming,
):
    """
    When force_non_streaming=True in litellm_params and client sends stream=True,
    the actual call to litellm.acompletion should have stream=False.
    """
    captured_kwargs = {}

    async def mock_acompletion(*args, **kwargs):
        captured_kwargs.update(kwargs)
        # Return a minimal mock response
        return litellm.ModelResponse(
            id="test",
            choices=[
                {
                    "message": {"role": "assistant", "content": "test"},
                    "index": 0,
                    "finish_reason": "stop",
                }
            ],
            model="test-model",
        )

    with patch("litellm.acompletion", side_effect=mock_acompletion):
        await router_with_force_non_streaming.acompletion(
            model="test-model",
            messages=[{"role": "user", "content": "Hello"}],
            stream=True,  # Client wants streaming
        )

    # The router should have overridden stream to False
    assert captured_kwargs.get("stream") is False
    # force_non_streaming should NOT be passed through to the LLM call
    assert "force_non_streaming" not in captured_kwargs


@pytest.mark.asyncio
async def test_normal_streaming_preserved_without_force(
    router_without_force_non_streaming,
):
    """
    When force_non_streaming is NOT set, stream=True should be preserved.
    """
    captured_kwargs = {}

    async def mock_acompletion(*args, **kwargs):
        captured_kwargs.update(kwargs)
        return litellm.ModelResponse(
            id="test",
            choices=[
                {
                    "message": {"role": "assistant", "content": "test"},
                    "index": 0,
                    "finish_reason": "stop",
                }
            ],
            model="test-model",
        )

    with patch("litellm.acompletion", side_effect=mock_acompletion):
        await router_without_force_non_streaming.acompletion(
            model="test-model",
            messages=[{"role": "user", "content": "Hello"}],
            stream=True,
        )

    # stream=True should be preserved
    assert captured_kwargs.get("stream") is True


@pytest.mark.asyncio
async def test_force_non_streaming_not_passed_to_llm(
    router_with_force_non_streaming,
):
    """
    force_non_streaming should be consumed by the router and NOT passed
    through to the underlying litellm.acompletion call.
    """
    captured_kwargs = {}

    async def mock_acompletion(*args, **kwargs):
        captured_kwargs.update(kwargs)
        return litellm.ModelResponse(
            id="test",
            choices=[
                {
                    "message": {"role": "assistant", "content": "test"},
                    "index": 0,
                    "finish_reason": "stop",
                }
            ],
            model="test-model",
        )

    with patch("litellm.acompletion", side_effect=mock_acompletion):
        await router_with_force_non_streaming.acompletion(
            model="test-model",
            messages=[{"role": "user", "content": "Hello"}],
            stream=False,  # Even without stream=True, force_non_streaming should not leak
        )

    assert "force_non_streaming" not in captured_kwargs
