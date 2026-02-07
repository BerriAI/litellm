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


def test_fake_stream_iterator_includes_output_tokens():
    """Test that FakeAnthropicMessagesStreamIterator includes output tokens in message_delta event.
    
    Regression test for GitHub issue #20187 where output tokens showed as 0 when using
    websearch_interception with Claude Code because the streaming response conversion
    wasn't properly including the output_tokens from the final response.
    """
    from litellm.llms.anthropic.experimental_pass_through.messages.fake_stream_iterator import (
        FakeAnthropicMessagesStreamIterator,
    )
    import json
    
    # Simulate a response from the agentic loop with usage data
    test_response = {
        "id": "msg_test123",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-4-5-20250929",
        "content": [
            {
                "type": "text",
                "text": "Here are today's breaking news headlines..."
            }
        ],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {
            "input_tokens": 38196,
            "output_tokens": 395
        }
    }
    
    # Create fake stream iterator
    fake_stream = FakeAnthropicMessagesStreamIterator(response=test_response)
    
    # Collect all chunks
    chunks = list(fake_stream)
    
    # Parse chunks to find message_start and message_delta events
    message_start_event = None
    message_delta_event = None
    
    for chunk in chunks:
        chunk_str = chunk.decode()
        if "event: message_start" in chunk_str:
            data_line = chunk_str.split("data: ")[1].strip()
            message_start_event = json.loads(data_line)
        elif "event: message_delta" in chunk_str:
            data_line = chunk_str.split("data: ")[1].strip()
            message_delta_event = json.loads(data_line)
    
    # Verify message_start has input_tokens and output_tokens=0 (as per Anthropic spec)
    assert message_start_event is not None, "Should have message_start event"
    assert message_start_event["message"]["usage"]["input_tokens"] == 38196
    assert message_start_event["message"]["usage"]["output_tokens"] == 0  # Always 0 in message_start
    
    # Verify message_delta has the actual output_tokens
    assert message_delta_event is not None, "Should have message_delta event"
    assert message_delta_event["usage"]["output_tokens"] == 395, \
        f"message_delta should have output_tokens=395, got {message_delta_event['usage']['output_tokens']}"


def test_fake_stream_iterator_preserves_stop_reason():
    """Test that FakeAnthropicMessagesStreamIterator preserves stop_reason from the original response.
    
    This is important for the client to know that the response completed successfully.
    """
    from litellm.llms.anthropic.experimental_pass_through.messages.fake_stream_iterator import (
        FakeAnthropicMessagesStreamIterator,
    )
    import json
    
    # Simulate a response with end_turn stop reason
    test_response = {
        "id": "msg_test123",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-4-5-20250929",
        "content": [
            {
                "type": "text",
                "text": "Response text"
            }
        ],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {
            "input_tokens": 100,
            "output_tokens": 50
        }
    }
    
    fake_stream = FakeAnthropicMessagesStreamIterator(response=test_response)
    chunks = list(fake_stream)
    
    # Find message_delta event
    message_delta_event = None
    for chunk in chunks:
        chunk_str = chunk.decode()
        if "event: message_delta" in chunk_str:
            data_line = chunk_str.split("data: ")[1].strip()
            message_delta_event = json.loads(data_line)
    
    assert message_delta_event is not None
    assert message_delta_event["delta"]["stop_reason"] == "end_turn"
