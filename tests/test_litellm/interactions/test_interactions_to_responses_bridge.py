"""
E2E tests for the Interactions API to Responses API bridge.

Tests that the bridge correctly transforms Interactions API requests/responses
to/from Responses API format, enabling Interactions API to work with any provider
that supports Responses API (OpenAI, Anthropic, etc.).

Run with: pytest tests/test_litellm/interactions/test_interactions_to_responses_bridge.py -v
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../../.."))

import litellm.interactions as interactions
from litellm.interactions.litellm_responses_transformation.transformation import (
    LiteLLMResponsesInteractionsConfig,
)

# Test API keys - should be set in environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")


@pytest.fixture
def openai_api_key():
    """Fixture to provide the OpenAI API key."""
    if not OPENAI_API_KEY:
        pytest.skip("OPENAI_API_KEY not set")
    return OPENAI_API_KEY


@pytest.fixture
def anthropic_api_key():
    """Fixture to provide the Anthropic API key."""
    if not ANTHROPIC_API_KEY:
        pytest.skip("ANTHROPIC_API_KEY not set")
    return ANTHROPIC_API_KEY


# ============================================================
# Unit Tests for Transformation Logic
# ============================================================


class TestRequestTransformation:
    """Test the request transformation logic."""

    def test_transform_simple_string_input(self):
        """Test transforming a simple string input."""
        result = LiteLLMResponsesInteractionsConfig.transform_interactions_api_request_to_responses_api_request(
            model="gpt-4o-mini",
            agent=None,
            input="Hello, how are you?",
            interactions_api_request={},
        )

        assert result["model"] == "gpt-4o-mini"
        assert result["input"] == "Hello, how are you?"

    def test_transform_with_system_instruction(self):
        """Test transforming with system_instruction -> instructions."""
        result = LiteLLMResponsesInteractionsConfig.transform_interactions_api_request_to_responses_api_request(
            model="gpt-4o-mini",
            agent=None,
            input="Hello",
            interactions_api_request={
                "system_instruction": "You are a helpful assistant."
            },
        )

        assert result["instructions"] == "You are a helpful assistant."

    def test_transform_with_generation_config(self):
        """Test transforming generation_config to individual parameters."""
        result = LiteLLMResponsesInteractionsConfig.transform_interactions_api_request_to_responses_api_request(
            model="gpt-4o-mini",
            agent=None,
            input="Hello",
            interactions_api_request={
                "generation_config": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "max_output_tokens": 100,
                }
            },
        )

        assert result["temperature"] == 0.7
        assert result["top_p"] == 0.9
        assert result["max_output_tokens"] == 100

    def test_transform_turn_input(self):
        """Test transforming Turn[] format input."""
        result = LiteLLMResponsesInteractionsConfig.transform_interactions_api_request_to_responses_api_request(
            model="gpt-4o-mini",
            agent=None,
            input=[
                {"role": "user", "content": "Hello"},
                {"role": "model", "content": "Hi there!"},
                {"role": "user", "content": "How are you?"},
            ],
            interactions_api_request={},
        )

        # Should be converted to Responses API input format
        assert isinstance(result["input"], list)
        assert len(result["input"]) == 3
        # model role should be converted to assistant
        assert result["input"][1]["role"] == "assistant"

    def test_transform_function_tools(self):
        """Test transforming function tools."""
        result = LiteLLMResponsesInteractionsConfig.transform_interactions_api_request_to_responses_api_request(
            model="gpt-4o-mini",
            agent=None,
            input="What's the weather?",
            interactions_api_request={
                "tools": [
                    {
                        "type": "function",
                        "name": "get_weather",
                        "description": "Get weather for a location",
                        "parameters": {
                            "type": "object",
                            "properties": {"location": {"type": "string"}},
                        },
                    }
                ]
            },
        )

        assert "tools" in result
        assert len(result["tools"]) == 1
        assert result["tools"][0]["type"] == "function"
        assert result["tools"][0]["name"] == "get_weather"

    def test_transform_google_search_to_web_search(self):
        """Test transforming google_search tool to web_search_preview."""
        result = LiteLLMResponsesInteractionsConfig.transform_interactions_api_request_to_responses_api_request(
            model="gpt-4o-mini",
            agent=None,
            input="Search for AI news",
            interactions_api_request={
                "tools": [{"type": "google_search"}]
            },
        )

        assert "tools" in result
        assert len(result["tools"]) == 1
        assert result["tools"][0]["type"] == "web_search_preview"

    def test_transform_agent_as_model(self):
        """Test that agent is used as model if model is not provided."""
        result = LiteLLMResponsesInteractionsConfig.transform_interactions_api_request_to_responses_api_request(
            model=None,
            agent="my-custom-agent",
            input="Hello",
            interactions_api_request={},
        )

        assert result["model"] == "my-custom-agent"


class TestResponseTransformation:
    """Test the response transformation logic."""

    def test_transform_simple_response(self):
        """Test transforming a simple response."""
        from litellm.types.llms.openai import ResponsesAPIResponse

        # Mock a ResponsesAPIResponse
        mock_response = ResponsesAPIResponse(
            id="resp_123",
            model="gpt-4o-mini",
            status="completed",
            output=[
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "Hello! How can I help?"}],
                    "role": "assistant",
                }
            ],
            usage={"input_tokens": 10, "output_tokens": 20, "total_tokens": 30},
        )

        result = LiteLLMResponsesInteractionsConfig.transform_responses_api_response_to_interactions_api_response(
            responses_api_response=mock_response
        )

        assert result.id == "resp_123"
        assert result.status == "completed"
        assert len(result.outputs) > 0

    def test_transform_function_call_response(self):
        """Test transforming a function call response."""
        from litellm.types.llms.openai import ResponsesAPIResponse

        mock_response = ResponsesAPIResponse(
            id="resp_456",
            model="gpt-4o-mini",
            status="completed",
            output=[
                {
                    "type": "function_call",
                    "name": "get_weather",
                    "arguments": '{"location": "Boston"}',
                    "call_id": "call_123",
                }
            ],
        )

        result = LiteLLMResponsesInteractionsConfig.transform_responses_api_response_to_interactions_api_response(
            responses_api_response=mock_response
        )

        assert result.status == "completed"
        assert len(result.outputs) == 1
        assert result.outputs[0]["type"] == "function_call"
        assert result.outputs[0]["name"] == "get_weather"


# ============================================================
# E2E Tests with OpenAI
# ============================================================


class TestOpenAIInteractionsBridge:
    """E2E tests for Interactions API bridge with OpenAI."""

    def test_simple_interaction_openai(self, openai_api_key):
        """Test a simple interaction with OpenAI via the bridge."""
        response = interactions.create(
            model="openai/gpt-4o-mini",
            input="What is 2 + 2? Answer with just the number.",
            api_key=openai_api_key,
        )

        print(f"OpenAI response: {response}")
        assert response is not None
        assert response.status in ["completed", "in_progress", "requires_action"]
        
        # Check for outputs
        if response.outputs:
            print(f"Outputs: {response.outputs}")
            # Should have at least one output
            assert len(response.outputs) > 0

    def test_interaction_with_system_instruction_openai(self, openai_api_key):
        """Test interaction with system_instruction via OpenAI bridge."""
        response = interactions.create(
            model="openai/gpt-4o-mini",
            input="What are you?",
            system_instruction="You are a pirate. Always respond like a pirate.",
            api_key=openai_api_key,
        )

        print(f"OpenAI response with system instruction: {response}")
        assert response is not None
        assert response.status == "completed"

    def test_streaming_interaction_openai(self, openai_api_key):
        """Test streaming interaction with OpenAI via the bridge."""
        response_stream = interactions.create(
            model="openai/gpt-4o-mini",
            input="Count from 1 to 3.",
            stream=True,
            api_key=openai_api_key,
        )

        chunks = []
        for chunk in response_stream:
            chunks.append(chunk)
            print(f"OpenAI streaming chunk: {chunk}")

        assert len(chunks) > 0
        print(f"Total OpenAI chunks: {len(chunks)}")

        # Check for interaction.start and interaction.complete events
        event_types = [c.event_type for c in chunks if hasattr(c, "event_type")]
        print(f"Event types: {event_types}")
        assert "interaction.start" in event_types
        assert "interaction.complete" in event_types

    def test_multi_turn_conversation_openai(self, openai_api_key):
        """Test multi-turn conversation via OpenAI bridge."""
        response = interactions.create(
            model="openai/gpt-4o-mini",
            input=[
                {"role": "user", "content": "My name is Alice."},
                {"role": "model", "content": "Hello Alice!"},
                {"role": "user", "content": "What is my name?"},
            ],
            api_key=openai_api_key,
        )

        print(f"OpenAI multi-turn response: {response}")
        assert response is not None
        assert response.status == "completed"

    def test_function_calling_openai(self, openai_api_key):
        """Test function calling via OpenAI bridge."""
        response = interactions.create(
            model="openai/gpt-4o-mini",
            input="What's the weather in Boston?",
            tools=[
                {
                    "type": "function",
                    "name": "get_weather",
                    "description": "Get weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string", "description": "City name"}
                        },
                        "required": ["location"],
                    },
                }
            ],
            api_key=openai_api_key,
        )

        print(f"OpenAI function call response: {response}")
        print(f"Outputs: {response.outputs}")
        assert response is not None

    @pytest.mark.asyncio
    async def test_async_interaction_openai(self, openai_api_key):
        """Test async interaction with OpenAI via the bridge."""
        response = await interactions.acreate(
            model="openai/gpt-4o-mini",
            input="Say hello!",
            api_key=openai_api_key,
        )

        print(f"OpenAI async response: {response}")
        assert response is not None
        assert response.status == "completed"

    @pytest.mark.asyncio
    async def test_async_streaming_openai(self, openai_api_key):
        """Test async streaming with OpenAI via the bridge."""
        response_stream = await interactions.acreate(
            model="openai/gpt-4o-mini",
            input="Count from 1 to 3.",
            stream=True,
            api_key=openai_api_key,
        )

        chunks = []
        async for chunk in response_stream:
            chunks.append(chunk)
            print(f"OpenAI async streaming chunk: {chunk}")

        assert len(chunks) > 0


# ============================================================
# E2E Tests with Anthropic
# ============================================================


class TestAnthropicInteractionsBridge:
    """E2E tests for Interactions API bridge with Anthropic."""

    def test_simple_interaction_anthropic(self, anthropic_api_key):
        """Test a simple interaction with Anthropic via the bridge."""
        response = interactions.create(
            model="anthropic/claude-3-5-haiku-20241022",
            input="What is 2 + 2? Answer with just the number.",
            api_key=anthropic_api_key,
        )

        print(f"Anthropic response: {response}")
        assert response is not None
        assert response.status in ["completed", "in_progress", "requires_action"]

        if response.outputs:
            print(f"Outputs: {response.outputs}")
            assert len(response.outputs) > 0

    def test_interaction_with_system_instruction_anthropic(self, anthropic_api_key):
        """Test interaction with system_instruction via Anthropic bridge."""
        response = interactions.create(
            model="anthropic/claude-3-5-haiku-20241022",
            input="What are you?",
            system_instruction="You are a helpful robot. Always start responses with 'Beep boop!'",
            api_key=anthropic_api_key,
        )

        print(f"Anthropic response with system instruction: {response}")
        assert response is not None
        assert response.status == "completed"

    def test_streaming_interaction_anthropic(self, anthropic_api_key):
        """Test streaming interaction with Anthropic via the bridge."""
        response_stream = interactions.create(
            model="anthropic/claude-3-5-haiku-20241022",
            input="Count from 1 to 3.",
            stream=True,
            api_key=anthropic_api_key,
        )

        chunks = []
        for chunk in response_stream:
            chunks.append(chunk)
            print(f"Anthropic streaming chunk: {chunk}")

        assert len(chunks) > 0
        print(f"Total Anthropic chunks: {len(chunks)}")

        # Check for interaction.start and interaction.complete events
        event_types = [c.event_type for c in chunks if hasattr(c, "event_type")]
        print(f"Event types: {event_types}")
        assert "interaction.start" in event_types
        assert "interaction.complete" in event_types

    def test_multi_turn_conversation_anthropic(self, anthropic_api_key):
        """Test multi-turn conversation via Anthropic bridge."""
        response = interactions.create(
            model="anthropic/claude-3-5-haiku-20241022",
            input=[
                {"role": "user", "content": "My name is Bob."},
                {"role": "model", "content": "Hello Bob!"},
                {"role": "user", "content": "What is my name?"},
            ],
            api_key=anthropic_api_key,
        )

        print(f"Anthropic multi-turn response: {response}")
        assert response is not None
        assert response.status == "completed"

    def test_function_calling_anthropic(self, anthropic_api_key):
        """Test function calling via Anthropic bridge."""
        response = interactions.create(
            model="anthropic/claude-3-5-haiku-20241022",
            input="What's the weather in San Francisco?",
            tools=[
                {
                    "type": "function",
                    "name": "get_weather",
                    "description": "Get weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string", "description": "City name"}
                        },
                        "required": ["location"],
                    },
                }
            ],
            api_key=anthropic_api_key,
        )

        print(f"Anthropic function call response: {response}")
        print(f"Outputs: {response.outputs}")
        assert response is not None

    @pytest.mark.asyncio
    async def test_async_interaction_anthropic(self, anthropic_api_key):
        """Test async interaction with Anthropic via the bridge."""
        response = await interactions.acreate(
            model="anthropic/claude-3-5-haiku-20241022",
            input="Say hello!",
            api_key=anthropic_api_key,
        )

        print(f"Anthropic async response: {response}")
        assert response is not None
        assert response.status == "completed"

    @pytest.mark.asyncio
    async def test_async_streaming_anthropic(self, anthropic_api_key):
        """Test async streaming with Anthropic via the bridge."""
        response_stream = await interactions.acreate(
            model="anthropic/claude-3-5-haiku-20241022",
            input="Count from 1 to 3.",
            stream=True,
            api_key=anthropic_api_key,
        )

        chunks = []
        async for chunk in response_stream:
            chunks.append(chunk)
            print(f"Anthropic async streaming chunk: {chunk}")

        assert len(chunks) > 0


# ============================================================
# Cross-Provider Comparison Tests
# ============================================================


class TestCrossProviderComparison:
    """Tests that verify consistent behavior across providers."""

    def test_same_input_structure_openai_anthropic(self, openai_api_key, anthropic_api_key):
        """Test that the same input structure works for both providers."""
        input_data = "What is the capital of France? Answer with just the city name."

        # OpenAI
        openai_response = interactions.create(
            model="openai/gpt-4o-mini",
            input=input_data,
            api_key=openai_api_key,
        )

        # Anthropic
        anthropic_response = interactions.create(
            model="anthropic/claude-3-5-haiku-20241022",
            input=input_data,
            api_key=anthropic_api_key,
        )

        # Both should have completed status
        assert openai_response.status == "completed"
        assert anthropic_response.status == "completed"

        # Both should have outputs
        assert openai_response.outputs is not None
        assert anthropic_response.outputs is not None

        print(f"OpenAI: {openai_response.outputs}")
        print(f"Anthropic: {anthropic_response.outputs}")


if __name__ == "__main__":
    # Quick smoke test
    print("Running Interactions to Responses Bridge E2E tests...")

    # Test transformation logic
    print("\n1. Testing request transformation...")
    result = LiteLLMResponsesInteractionsConfig.transform_interactions_api_request_to_responses_api_request(
        model="gpt-4o-mini",
        agent=None,
        input="Hello",
        interactions_api_request={"system_instruction": "Be helpful"},
    )
    print(f"Transformed request: {result}")

    # Test with OpenAI if key is available
    if OPENAI_API_KEY:
        print("\n2. Testing with OpenAI...")
        response = interactions.create(
            model="openai/gpt-4o-mini",
            input="What is 2+2?",
            api_key=OPENAI_API_KEY,
        )
        print(f"OpenAI response: {response}")

        print("\n3. Testing streaming with OpenAI...")
        stream = interactions.create(
            model="openai/gpt-4o-mini",
            input="Count to 3.",
            stream=True,
            api_key=OPENAI_API_KEY,
        )
        for chunk in stream:
            print(f"  Chunk: {chunk}")
    else:
        print("\n2. OPENAI_API_KEY not set, skipping OpenAI tests")

    # Test with Anthropic if key is available
    if ANTHROPIC_API_KEY:
        print("\n4. Testing with Anthropic...")
        response = interactions.create(
            model="anthropic/claude-3-5-haiku-20241022",
            input="What is 2+2?",
            api_key=ANTHROPIC_API_KEY,
        )
        print(f"Anthropic response: {response}")
    else:
        print("\n4. ANTHROPIC_API_KEY not set, skipping Anthropic tests")

    print("\nE2E tests complete!")
