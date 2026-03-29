"""
Tests that run_async_fallback deep-copies kwargs before each fallback attempt,
preventing provider-specific mutations (e.g., Bedrock popping 'tools' from
optional_params) from corrupting subsequent fallback calls.

See: https://github.com/BerriAI/litellm/issues/24764
"""

import copy
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import litellm
from litellm.router_utils.fallback_event_handlers import run_async_fallback

def _make_router_mock() -> MagicMock:
    """Create a minimal mock that behaves like a LiteLLM Router."""
    router = MagicMock()
    router.log_retry = MagicMock(side_effect=lambda kwargs, e: kwargs)
    return router


@pytest.mark.asyncio
async def test_run_async_fallback_does_not_mutate_kwargs():
    """
    Simulate a scenario where the first fallback provider mutates kwargs
    (e.g., pops 'tools' from optional_params) and verify that the second
    fallback provider still receives the original kwargs with 'tools' intact.

    This reproduces the bug where Bedrock's converse_handler pops 'tools'
    and 'tool_choice' from optional_params, causing the Azure OpenAI
    fallback to receive tool_choice without tools.
    """
    router = _make_router_mock()

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"},
                    },
                },
            },
        }
    ]

    original_kwargs = {
        "model": "bedrock-model",
        "messages": [{"role": "user", "content": "What is the weather?"}],
        "optional_params": {
            "tools": copy.deepcopy(tools),
            "tool_choice": "auto",
            "stream": False,
        },
        "litellm_params": {
            "metadata": {"previous_models": ["bedrock-model"]},
        },
        "metadata": {"previous_models": ["bedrock-model"]},
        "original_function": AsyncMock(),
    }

    call_count = 0
    captured_kwargs: List[Dict[str, Any]] = []

    async def mock_function_with_fallbacks(*args: Any, **kwargs: Any) -> Any:
        nonlocal call_count
        call_count += 1
        # Capture a deep copy of what was received so we can inspect later
        captured_kwargs.append(copy.deepcopy(kwargs))

        if call_count == 1:
            # Simulate Bedrock-like mutation: pop tools and tool_choice
            kwargs.get("optional_params", {}).pop("tools", None)
            kwargs.get("optional_params", {}).pop("tool_choice", None)
            raise litellm.exceptions.InternalServerError(
                message="Simulated Bedrock timeout",
                llm_provider="bedrock",
                model="bedrock-model",
            )
        # Second call succeeds
        mock_response = MagicMock()
        mock_response._hidden_params = {}
        return mock_response

    router.async_function_with_fallbacks = AsyncMock(
        side_effect=mock_function_with_fallbacks
    )

    original_exception = litellm.exceptions.InternalServerError(
        message="Simulated Bedrock timeout",
        llm_provider="bedrock",
        model="bedrock-model",
    )

    await run_async_fallback(
        litellm_router=router,
        fallback_model_group=["azure-model-1", "azure-model-2"],
        original_model_group="bedrock-model",
        original_exception=original_exception,
        max_fallbacks=5,
        fallback_depth=0,
        **original_kwargs,
    )

    # Both fallback attempts should have been called
    assert call_count == 2, f"Expected 2 calls, got {call_count}"

    # The second fallback call should still have 'tools' in optional_params
    second_call_kwargs = captured_kwargs[1]
    optional_params = second_call_kwargs.get("optional_params", {})

    assert "tools" in optional_params, (
        "Second fallback lost 'tools' from optional_params. "
        "This means kwargs were mutated by the first fallback attempt."
    )
    assert "tool_choice" in optional_params, (
        "Second fallback lost 'tool_choice' from optional_params. "
        "This means kwargs were mutated by the first fallback attempt."
    )
    assert optional_params["tool_choice"] == "auto"
    assert len(optional_params["tools"]) == 1


@pytest.mark.asyncio
async def test_run_async_fallback_original_kwargs_unchanged():
    """
    Verify that the original kwargs dict passed to run_async_fallback
    is not modified, even if fallback providers mutate their copy.
    """
    router = _make_router_mock()

    original_optional_params = {
        "tools": [{"type": "function", "function": {"name": "test_fn"}}],
        "tool_choice": "auto",
    }

    kwargs = {
        "model": "primary-model",
        "messages": [{"role": "user", "content": "Hello"}],
        "optional_params": copy.deepcopy(original_optional_params),
        "metadata": {"previous_models": ["primary-model"]},
        "original_function": AsyncMock(),
    }

    # Keep a reference to the original optional_params dict
    kwargs_optional_params_ref = kwargs["optional_params"]

    async def mock_function_with_fallbacks(*args: Any, **kwargs: Any) -> Any:
        # Simulate mutation
        kwargs.get("optional_params", {}).pop("tools", None)
        mock_response = MagicMock()
        mock_response._hidden_params = {}
        return mock_response

    router.async_function_with_fallbacks = AsyncMock(
        side_effect=mock_function_with_fallbacks
    )

    original_exception = Exception("Simulated error")

    await run_async_fallback(
        litellm_router=router,
        fallback_model_group=["fallback-model"],
        original_model_group="primary-model",
        original_exception=original_exception,
        max_fallbacks=5,
        fallback_depth=0,
        **kwargs,
    )

    # The original optional_params reference should still have 'tools'
    assert "tools" in kwargs_optional_params_ref, (
        "Original kwargs['optional_params'] was mutated by fallback. "
        "safe_deep_copy should protect the caller's kwargs."
    )

