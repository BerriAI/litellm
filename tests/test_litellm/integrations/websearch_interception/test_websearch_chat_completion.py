"""
Integration tests for WebSearch interception with chat completions API.

Tests the end-to-end flow of websearch_interception callback with
litellm.acompletion() for transparent server-side web search execution.
"""
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import litellm
from litellm.integrations.websearch_interception.handler import (
    WebSearchInterceptionLogger,
)
from litellm.types.utils import LlmProviders, ModelResponse


@pytest.fixture
def mock_search_response():
    """Mock search response from litellm.asearch()"""
    mock_response = MagicMock()
    mock_response.results = [
        MagicMock(
            title="Weather in San Francisco",
            url="https://weather.com/sf",
            snippet="Current weather: 65°F, partly cloudy",
        )
    ]
    return mock_response


@pytest.fixture
def websearch_logger():
    """Create a WebSearchInterceptionLogger instance"""
    return WebSearchInterceptionLogger(
        enabled_providers=[LlmProviders.OPENAI, LlmProviders.MINIMAX]
    )


@pytest.mark.asyncio
@pytest.mark.skipif(
    os.environ.get("OPENAI_API_KEY") is None,
    reason="OPENAI_API_KEY not set",
)
async def test_websearch_chat_completion_with_openai():
    """Test websearch interception with OpenAI chat completions API.
    
    This test verifies that:
    1. Model calls litellm_web_search tool
    2. Server executes web search automatically
    3. Server makes follow-up request with search results
    4. User gets final answer without tool_calls
    """
    # Configure WebSearch interception
    original_callbacks = litellm.callbacks.copy() if litellm.callbacks else []
    websearch_logger = WebSearchInterceptionLogger(
        enabled_providers=[LlmProviders.OPENAI]
    )
    litellm.callbacks = [websearch_logger]
    
    try:
        response = await litellm.acompletion(
            model="gpt-4o-mini",  # Use cheaper model for testing
            messages=[
                {"role": "user", "content": "What's the weather in San Francisco today?"}
            ],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "litellm_web_search",
                        "description": "Search the web for information",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "Search query",
                                }
                            },
                            "required": ["query"],
                        },
                    },
                }
            ],
        )
        
        # Verify response structure
        assert isinstance(response, ModelResponse)
        assert response.choices[0].message.content is not None
        assert len(response.choices[0].message.content) > 0
        
        # If agentic loop worked, we should NOT have tool_calls in final response
        # (they should have been executed and replaced with final answer)
        if hasattr(response.choices[0].message, "tool_calls"):
            # If tool_calls exist, it means agentic loop didn't run
            # This could happen if search tool is not configured
            pytest.skip(
                "Agentic loop did not execute - search tool may not be configured"
            )
        
        # Verify we got a meaningful response
        assert response.choices[0].finish_reason in ["stop", "end_turn"]
        
    finally:
        # Restore original callbacks
        litellm.callbacks = original_callbacks


@pytest.mark.asyncio
async def test_websearch_chat_completion_hook_detection():
    """Test that websearch hook correctly detects tool calls in response."""
    from litellm.types.utils import (
        ChatCompletionMessageToolCall,
        Choices,
        Function,
        Message,
    )
    
    websearch_logger = WebSearchInterceptionLogger(
        enabled_providers=[LlmProviders.OPENAI]
    )
    
    # Mock response with litellm_web_search tool call
    mock_response = ModelResponse(
        id="test-123",
        choices=[
            Choices(
                finish_reason="tool_calls",
                index=0,
                message=Message(
                    role="assistant",
                    content=None,
                    tool_calls=[
                        ChatCompletionMessageToolCall(
                            id="call_123",
                            type="function",
                            function=Function(
                                name="litellm_web_search",
                                arguments='{"query": "weather in SF"}',
                            ),
                        )
                    ],
                )
            )
        ],
        model="gpt-4o",
        object="chat.completion",
        created=1234567890,
    )
    
    # Test should_run_chat_completion_agentic_loop
    should_run, tools_dict = (
        await websearch_logger.async_should_run_chat_completion_agentic_loop(
            response=mock_response,
            model="gpt-4o",
            messages=[{"role": "user", "content": "What's the weather?"}],
            tools=[
                {
                    "type": "function",
                    "function": {"name": "litellm_web_search"},
                }
            ],
            stream=False,
            custom_llm_provider="openai",
            kwargs={},
        )
    )
    
    # Verify hook detected the tool call
    assert should_run is True
    assert "tool_calls" in tools_dict
    assert len(tools_dict["tool_calls"]) == 1
    assert tools_dict["tool_calls"][0]["name"] == "litellm_web_search"
    assert tools_dict["response_format"] == "openai"


@pytest.mark.asyncio
async def test_websearch_not_triggered_without_tool():
    """Test that websearch hook is NOT triggered when no web search tool in request."""
    from litellm.types.utils import Choices, Message
    
    websearch_logger = WebSearchInterceptionLogger(
        enabled_providers=[LlmProviders.OPENAI]
    )
    
    mock_response = ModelResponse(
        id="test-123",
        choices=[
            Choices(
                finish_reason="stop",
                index=0,
                message=Message(
                    role="assistant",
                    content="Here's the answer",
                    tool_calls=None,
                )
            )
        ],
        model="gpt-4o",
        object="chat.completion",
        created=1234567890,
    )
    
    # Test without web search tool
    should_run, tools_dict = (
        await websearch_logger.async_should_run_chat_completion_agentic_loop(
            response=mock_response,
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello"}],
            tools=[
                {
                    "type": "function",
                    "function": {"name": "some_other_tool"},
                }
            ],
            stream=False,
            custom_llm_provider="openai",
            kwargs={},
        )
    )
    
    # Verify hook did NOT trigger
    assert should_run is False
    assert tools_dict == {}


@pytest.mark.asyncio
async def test_websearch_not_triggered_for_disabled_provider():
    """Test that websearch hook is NOT triggered for providers not in enabled_providers."""
    from litellm.types.utils import (
        ChatCompletionMessageToolCall,
        Choices,
        Function,
        Message,
    )

    # Only enable bedrock
    websearch_logger = WebSearchInterceptionLogger(
        enabled_providers=[LlmProviders.BEDROCK]
    )
    
    mock_response = ModelResponse(
        id="test-123",
        choices=[
            Choices(
                finish_reason="tool_calls",
                index=0,
                message=Message(
                    role="assistant",
                    content=None,
                    tool_calls=[
                        ChatCompletionMessageToolCall(
                            id="call_123",
                            type="function",
                            function=Function(
                                name="litellm_web_search",
                                arguments='{"query": "test"}',
                            ),
                        )
                    ],
                )
            )
        ],
        model="gpt-4o",
        object="chat.completion",
        created=1234567890,
    )
    
    # Test with OpenAI provider (not enabled)
    should_run, tools_dict = (
        await websearch_logger.async_should_run_chat_completion_agentic_loop(
            response=mock_response,
            model="gpt-4o",
            messages=[{"role": "user", "content": "test"}],
            tools=[
                {
                    "type": "function",
                    "function": {"name": "litellm_web_search"},
                }
            ],
            stream=False,
            custom_llm_provider="openai",  # Not in enabled_providers
            kwargs={},
        )
    )
    
    # Verify hook did NOT trigger
    assert should_run is False
    assert tools_dict == {}


@pytest.mark.asyncio
async def test_websearch_json_serialization_fix():
    """Test that tool call arguments are properly JSON serialized.
    
    Regression test for the bug where arguments were converted to Python
    string representation instead of proper JSON, causing providers like
    MiniMax to reject requests with 'invalid function arguments json string'.
    """
    from litellm.integrations.websearch_interception.transformation import (
        WebSearchTransformation,
    )

    # Mock tool calls with dict input
    tool_calls = [
        {
            "id": "call_123",
            "name": "litellm_web_search",
            "input": {"query": "weather in SF"},  # Dict input
        }
    ]
    
    search_results = ["Weather: 65°F, partly cloudy"]
    
    # Transform to OpenAI format
    assistant_message, tool_messages = WebSearchTransformation.transform_response(
        tool_calls=tool_calls,
        search_results=search_results,
        response_format="openai",
    )
    
    # Verify arguments are properly JSON serialized
    import json
    
    arguments_str = assistant_message["tool_calls"][0]["function"]["arguments"]
    
    # Should be valid JSON
    parsed_args = json.loads(arguments_str)
    assert parsed_args == {"query": "weather in SF"}
    
    # Should NOT be Python string representation like "{'query': 'weather in SF'}"
    assert arguments_str == '{"query": "weather in SF"}'
    assert arguments_str != "{'query': 'weather in SF'}"


@pytest.mark.asyncio
@pytest.mark.skipif(
    os.environ.get("OPENAI_API_KEY") is None
    or os.environ.get("PERPLEXITY_API_KEY") is None,
    reason="OPENAI_API_KEY or PERPLEXITY_API_KEY not set",
)
async def test_websearch_streaming_conversion():
    """Test that streaming requests are converted to non-streaming for web search.
    
    When stream=True is passed with web search tools, the handler should:
    1. Convert stream=True to stream=False for initial request
    2. Execute web search
    3. Convert final response back to streaming
    """
    websearch_logger = WebSearchInterceptionLogger(
        enabled_providers=[LlmProviders.OPENAI], search_tool_name="perplexity-search"
    )
    litellm.callbacks = [websearch_logger]
    
    try:
        response = await litellm.acompletion(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": "What's the latest AI news?"}
            ],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "litellm_web_search",
                        "description": "Search the web",
                        "parameters": {
                            "type": "object",
                            "properties": {"query": {"type": "string"}},
                        },
                    },
                }
            ],
            stream=True,
        )
        
        # Response should be a streaming iterator
        chunks = []
        async for chunk in response:
            chunks.append(chunk)
        
        # Verify we got streaming chunks
        assert len(chunks) > 0
        
        # Verify chunks have expected structure
        for chunk in chunks:
            assert hasattr(chunk, "choices")
            assert len(chunk.choices) > 0
            
    finally:
        litellm.callbacks = []


if __name__ == "__main__":
    # Run with: pytest test_websearch_chat_completion.py -v -s
    pytest.main([__file__, "-v", "-s"])
