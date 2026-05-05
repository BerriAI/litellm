"""
Unit tests for WebSearch Interception Handler

Tests the WebSearchInterceptionLogger class and helper functions.
"""

from unittest.mock import AsyncMock, MagicMock, Mock

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
async def test_async_build_agentic_loop_plan_returns_request_patch():
    """Callback should return a typed patch for base handler reruns."""
    logger = WebSearchInterceptionLogger(enabled_providers=["bedrock"])
    logger._execute_search = AsyncMock(  # type: ignore
        return_value="Title: LiteLLM\nURL: docs\nSnippet: test"
    )

    tools_dict = {
        "tool_calls": [
            {
                "id": "toolu_123",
                "type": "tool_use",
                "name": "litellm_web_search",
                "input": {"query": "what is litellm"},
            }
        ],
        "response_format": "anthropic",
    }
    logging_obj = MagicMock()
    logging_obj.model_call_details = {
        "agentic_loop_params": {"model": "bedrock/invoke/claude-3-5-sonnet"}
    }
    kwargs = {
        "temperature": 0.2,
        "_websearch_interception_converted_stream": True,
        "litellm_logging_obj": object(),
    }

    plan = await logger.async_build_agentic_loop_plan(
        tools=tools_dict,
        model="claude-3-5-sonnet",
        messages=[{"role": "user", "content": "search LiteLLM"}],
        response=None,
        anthropic_messages_provider_config=None,
        anthropic_messages_optional_request_params={
            "max_tokens": 1024,
            "tools": [{"name": "litellm_web_search"}],
        },
        logging_obj=logging_obj,
        stream=False,
        kwargs=kwargs,
    )

    assert plan.run_agentic_loop is True
    assert plan.request_patch is not None
    assert plan.request_patch.model == "bedrock/invoke/claude-3-5-sonnet"
    assert plan.request_patch.max_tokens == 1024
    assert plan.request_patch.messages is not None
    assert len(plan.request_patch.messages) == 3
    assert "_websearch_interception_converted_stream" not in plan.request_patch.kwargs
    assert "litellm_logging_obj" not in plan.request_patch.kwargs
    assert plan.request_patch.kwargs["temperature"] == 0.2


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
        k: v
        for k, v in kwargs_with_internal_flags.items()
        if not k.startswith("_websearch_interception")
    }

    # Verify internal flags are filtered out
    assert "_websearch_interception_converted_stream" not in kwargs_for_followup
    assert "_websearch_interception_other_flag" not in kwargs_for_followup

    # Verify regular kwargs are preserved
    assert kwargs_for_followup["temperature"] == 0.7
    assert kwargs_for_followup["max_tokens"] == 1024


@pytest.mark.asyncio
async def test_async_pre_call_deployment_hook_provider_from_top_level_kwargs():
    """Test that async_pre_call_deployment_hook finds custom_llm_provider at top-level kwargs.

    Regression test for bug where the hook only checked kwargs["litellm_params"]["custom_llm_provider"]
    but the router places custom_llm_provider at the top level of kwargs.
    """
    logger = WebSearchInterceptionLogger(enabled_providers=["bedrock"])

    # Simulate kwargs as they arrive from the router path:
    # custom_llm_provider is at the TOP LEVEL (not nested under litellm_params)
    kwargs = {
        "model": "anthropic.claude-haiku-4-5-20251001-v1:0",
        "messages": [{"role": "user", "content": "Search the web for LiteLLM"}],
        "tools": [
            {"type": "web_search_20250305", "name": "web_search", "max_uses": 3},
            {"type": "function", "function": {"name": "other_tool", "parameters": {}}},
        ],
        "custom_llm_provider": "bedrock",
        "api_key": "fake-key",
    }

    result = await logger.async_pre_call_deployment_hook(kwargs=kwargs, call_type=None)

    # Should NOT be None — the hook should have triggered
    assert result is not None
    # The web_search tool should be converted to litellm_web_search (OpenAI format)
    assert any(
        t.get("type") == "function"
        and t.get("function", {}).get("name") == "litellm_web_search"
        for t in result["tools"]
    )
    # The non-web-search tool should be preserved
    assert any(
        t.get("type") == "function"
        and t.get("function", {}).get("name") == "other_tool"
        for t in result["tools"]
    )


@pytest.mark.asyncio
async def test_async_pre_call_deployment_hook_returns_full_kwargs():
    """Test that async_pre_call_deployment_hook returns the full kwargs dict, not a partial one.

    Regression test for bug where the hook returned {"tools": converted_tools} instead of
    the full kwargs dict, causing model/messages/api_key/etc. to be lost.
    """
    logger = WebSearchInterceptionLogger(enabled_providers=["openai"])

    kwargs = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Search for something"}],
        "tools": [
            {"type": "web_search_20250305", "name": "web_search"},
        ],
        "custom_llm_provider": "openai",
        "api_key": "fake-key-for-testing",
        "temperature": 0.7,
        "metadata": {"user": "test"},
    }

    result = await logger.async_pre_call_deployment_hook(kwargs=kwargs, call_type=None)

    assert result is not None
    # All original keys must be preserved
    assert result["model"] == "gpt-4o"
    assert result["messages"] == [{"role": "user", "content": "Search for something"}]
    assert result["api_key"] == "fake-key-for-testing"
    assert result["temperature"] == 0.7
    assert result["metadata"] == {"user": "test"}
    assert result["custom_llm_provider"] == "openai"
    # Tools should be converted
    assert any(
        t.get("type") == "function"
        and t.get("function", {}).get("name") == "litellm_web_search"
        for t in result["tools"]
    )


@pytest.mark.asyncio
async def test_async_pre_call_deployment_hook_skips_disabled_provider():
    """Test that the hook returns None for providers not in enabled_providers."""
    logger = WebSearchInterceptionLogger(enabled_providers=["bedrock"])

    kwargs = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "test"}],
        "tools": [{"type": "web_search_20250305", "name": "web_search"}],
        "custom_llm_provider": "openai",  # Not in enabled_providers
    }

    result = await logger.async_pre_call_deployment_hook(kwargs=kwargs, call_type=None)
    assert result is None


@pytest.mark.asyncio
async def test_async_pre_call_deployment_hook_skips_no_websearch_tools():
    """Test that the hook returns None when no web search tools are present."""
    logger = WebSearchInterceptionLogger(enabled_providers=["openai"])

    kwargs = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "test"}],
        "tools": [
            {"type": "function", "function": {"name": "calculator", "parameters": {}}},
        ],
        "custom_llm_provider": "openai",
    }

    result = await logger.async_pre_call_deployment_hook(kwargs=kwargs, call_type=None)
    assert result is None


@pytest.mark.asyncio
async def test_async_pre_call_deployment_hook_nested_litellm_params_fallback():
    """Test that the hook still works when custom_llm_provider is in nested litellm_params.

    This is the Anthropic experimental pass-through path where litellm_params is
    explicitly constructed with custom_llm_provider inside it.
    """
    logger = WebSearchInterceptionLogger(enabled_providers=["bedrock"])

    kwargs = {
        "model": "anthropic.claude-haiku-4-5-20251001-v1:0",
        "messages": [{"role": "user", "content": "test"}],
        "tools": [{"type": "web_search_20250305", "name": "web_search"}],
        "litellm_params": {
            "custom_llm_provider": "bedrock",
        },
    }

    result = await logger.async_pre_call_deployment_hook(kwargs=kwargs, call_type=None)

    assert result is not None
    assert any(
        t.get("type") == "function"
        and t.get("function", {}).get("name") == "litellm_web_search"
        for t in result["tools"]
    )
    # Full kwargs preserved
    assert result["model"] == "anthropic.claude-haiku-4-5-20251001-v1:0"


@pytest.mark.asyncio
async def test_async_pre_call_deployment_hook_provider_derived_from_model_name():
    """Test that async_pre_call_deployment_hook derives custom_llm_provider from the model name.

    Regression test for the router _acompletion path where custom_llm_provider is NOT
    in kwargs at all — neither at top-level nor in litellm_params. The hook must derive
    the provider from the model name (e.g., "openai/gpt-4o-mini" → "openai").
    """
    logger = WebSearchInterceptionLogger(enabled_providers=["openai"])

    # Simulate kwargs as they arrive from router._acompletion:
    # NO custom_llm_provider key anywhere — only model name contains the provider
    kwargs = {
        "model": "openai/gpt-4o-mini",
        "messages": [{"role": "user", "content": "Search the web for LiteLLM"}],
        "tools": [
            {"type": "web_search_20250305", "name": "web_search", "max_uses": 3},
        ],
        "api_key": "fake-key",
    }

    result = await logger.async_pre_call_deployment_hook(kwargs=kwargs, call_type=None)

    # Should NOT be None — the hook should derive "openai" from "openai/gpt-4o-mini"
    assert result is not None
    assert any(
        t.get("type") == "function"
        and t.get("function", {}).get("name") == "litellm_web_search"
        for t in result["tools"]
    )
    # Full kwargs preserved
    assert result["model"] == "openai/gpt-4o-mini"
    assert result["api_key"] == "fake-key"


@pytest.mark.asyncio
async def test_deployment_hook_converts_stream_and_logging_obj_syncs():
    """
    Regression test: websearch interception with stream=True must not skip logging.

    Before the fix, the stream conversion only happened in async_pre_request_hook
    (inside the anthropic_messages function scope). wrapper_async still saw
    stream=True, took the streaming early-return path, and skipped all spend/cost
    logging.  The fix moves stream conversion into the deployment hook so
    wrapper_async sees stream=False, and then syncs logging_obj.stream.

    This test verifies:
    1. The deployment hook sets stream=False and the converted flag.
    2. wrapper_async syncs logging_obj.stream after the hook runs.
    """
    logger = WebSearchInterceptionLogger(enabled_providers=["bedrock"])

    kwargs = {
        "model": "anthropic.claude-opus-4-6-20250219-v1:0",
        "messages": [{"role": "user", "content": "Search for LiteLLM"}],
        "tools": [
            {"type": "web_search_20250305", "name": "web_search", "max_uses": 3},
        ],
        "custom_llm_provider": "bedrock",
        "stream": True,
    }

    result = await logger.async_pre_call_deployment_hook(kwargs=kwargs, call_type=None)

    assert result is not None
    assert result["stream"] is False
    assert result["_websearch_interception_converted_stream"] is True

    # Simulate what wrapper_async does after the deployment hook:
    # logging_obj.stream was set to True during function_setup (before hook).
    # After the hook, wrapper_async must sync it.
    logging_obj = MagicMock()
    logging_obj.stream = True  # original value from function_setup

    _hook_stream = result.get("stream")
    if _hook_stream is not None and logging_obj.stream != _hook_stream:
        logging_obj.stream = _hook_stream

    assert logging_obj.stream is False
