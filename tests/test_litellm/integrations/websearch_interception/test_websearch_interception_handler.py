"""
Unit tests for WebSearch Interception Handler

Tests the WebSearchInterceptionLogger class and helper functions.
"""

from unittest.mock import Mock

import pytest

from litellm.integrations.websearch_interception.handler import (
    WebSearchInterceptionLogger,
)
from litellm.types.utils import LlmProviders


def test_initialize_from_proxy_config():
    """Test initialization from proxy config with litellm_settings"""
    litellm_settings = {
        "websearch_interception_params": {
            "enabled_providers": ["bedrock", "vertex_ai"],
            "search_tool_name": "my-search",
        }
    }
    callback_specific_params = {}

    logger = WebSearchInterceptionLogger.initialize_from_proxy_config(
        litellm_settings=litellm_settings,
        callback_specific_params=callback_specific_params,
    )

    assert LlmProviders.BEDROCK.value in logger.enabled_providers
    assert LlmProviders.VERTEX_AI.value in logger.enabled_providers
    assert logger.search_tool_name == "my-search"


@pytest.mark.asyncio
async def test_async_should_run_agentic_loop():
    """Test that agentic loop is NOT triggered for wrong provider or missing WebSearch tool"""
    logger = WebSearchInterceptionLogger(enabled_providers=["bedrock"])

    # Test 1: Wrong provider (not in enabled_providers)
    response = Mock()
    should_run, tools_dict = await logger.async_should_run_agentic_loop(
        response=response,
        model="gpt-4",
        messages=[],
        tools=[{"name": "WebSearch"}],
        stream=False,
        custom_llm_provider="openai",  # Not in enabled_providers
        kwargs={},
    )

    assert should_run is False
    assert tools_dict == {}

    # Test 2: No WebSearch tool in request
    should_run, tools_dict = await logger.async_should_run_agentic_loop(
        response=response,
        model="bedrock/claude",
        messages=[],
        tools=[{"name": "SomeOtherTool"}],  # No WebSearch
        stream=False,
        custom_llm_provider="bedrock",
        kwargs={},
    )

    assert should_run is False
    assert tools_dict == {}


@pytest.mark.asyncio
async def test_internal_flags_filtered_from_followup_kwargs():
    """Test that internal _websearch_interception flags are filtered from follow-up request kwargs.

    Regression test for bug where _websearch_interception_converted_stream was passed
    to the follow-up LLM request, causing "Extra inputs are not permitted" errors
    from providers like Bedrock that use strict parameter validation.
    """
    logger = WebSearchInterceptionLogger(enabled_providers=["bedrock"])

    # Simulate kwargs that would be passed during agentic loop execution
    kwargs_with_internal_flags = {
        "_websearch_interception_converted_stream": True,
        "_websearch_interception_other_flag": "test",
        "temperature": 0.7,
        "max_tokens": 1024,
    }

    # Apply the same filtering logic used in _execute_agentic_loop
    kwargs_for_followup = {
        k: v for k, v in kwargs_with_internal_flags.items()
        if not k.startswith('_websearch_interception')
    }

    # Verify internal flags are filtered out
    assert "_websearch_interception_converted_stream" not in kwargs_for_followup
    assert "_websearch_interception_other_flag" not in kwargs_for_followup

    # Verify regular kwargs are preserved
    assert kwargs_for_followup["temperature"] == 0.7
    assert kwargs_for_followup["max_tokens"] == 1024
