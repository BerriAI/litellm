"""
Tests for run_async_fallback metadata key handling.

Verifies that run_async_fallback uses the correct metadata variable name
("litellm_metadata" for Responses API routes, "metadata" for others)
instead of always hardcoding "metadata".

Regression test for https://github.com/BerriAI/litellm/issues/25402
"""

import os
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system-path

from litellm.router_utils.fallback_event_handlers import run_async_fallback


@pytest.mark.asyncio
async def test_fallback_uses_litellm_metadata_when_present():
    """When kwargs contain 'litellm_metadata' (Responses API), fallback should
    update that key instead of injecting a new 'metadata' key."""
    mock_router = MagicMock()
    mock_router.log_retry = MagicMock(side_effect=lambda kwargs, e: kwargs)

    # Simulate successful fallback response
    mock_response = MagicMock()
    mock_router.async_function_with_fallbacks = AsyncMock(return_value=mock_response)

    kwargs = {
        "model": "gpt-4.1",
        "litellm_metadata": {"existing_key": "value"},
        "original_function": AsyncMock(),
    }

    await run_async_fallback(
        litellm_router=mock_router,
        fallback_model_group=["gpt-4.1-paid"],
        original_model_group="gpt-4.1",
        original_exception=Exception("mock primary failure"),
        max_fallbacks=3,
        fallback_depth=0,
        **kwargs,
    )

    # The fallback should have updated litellm_metadata, not created a new metadata key
    call_kwargs = mock_router.async_function_with_fallbacks.call_args
    passed_kwargs = call_kwargs.kwargs if call_kwargs.kwargs else call_kwargs[1]

    assert (
        "metadata" not in passed_kwargs
    ), "run_async_fallback should not inject 'metadata' when 'litellm_metadata' is present"
    assert "litellm_metadata" in passed_kwargs
    assert passed_kwargs["litellm_metadata"]["model_group"] == "gpt-4.1-paid"
    assert passed_kwargs["litellm_metadata"]["existing_key"] == "value"


@pytest.mark.asyncio
async def test_fallback_uses_metadata_when_litellm_metadata_absent():
    """When kwargs do NOT contain 'litellm_metadata' (standard routes),
    fallback should use the default 'metadata' key."""
    mock_router = MagicMock()
    mock_router.log_retry = MagicMock(side_effect=lambda kwargs, e: kwargs)

    mock_response = MagicMock()
    mock_router.async_function_with_fallbacks = AsyncMock(return_value=mock_response)

    kwargs = {
        "model": "gpt-4",
        "metadata": {"existing_key": "value"},
        "original_function": AsyncMock(),
    }

    await run_async_fallback(
        litellm_router=mock_router,
        fallback_model_group=["gpt-3.5-turbo"],
        original_model_group="gpt-4",
        original_exception=Exception("mock primary failure"),
        max_fallbacks=3,
        fallback_depth=0,
        **kwargs,
    )

    call_kwargs = mock_router.async_function_with_fallbacks.call_args
    passed_kwargs = call_kwargs.kwargs if call_kwargs.kwargs else call_kwargs[1]

    assert "metadata" in passed_kwargs
    assert passed_kwargs["metadata"]["model_group"] == "gpt-3.5-turbo"
    assert passed_kwargs["metadata"]["existing_key"] == "value"
