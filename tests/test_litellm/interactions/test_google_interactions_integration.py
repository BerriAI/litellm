"""
Integration tests for Google Interactions API.

Tests the litellm.interactions.create() and related methods against the Google AI Studio API.

Per OpenAPI spec: https://ai.google.dev/static/api/interactions.openapi.json

Run with: pytest tests/test_litellm/interactions/test_google_interactions_integration.py -v
"""

import asyncio
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../../.."))

import litellm
import litellm.interactions as interactions

# Test API key - should be set in environment
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


@pytest.fixture
def api_key():
    """Fixture to provide the API key."""
    if not GEMINI_API_KEY:
        pytest.skip("GEMINI_API_KEY not set")
    return GEMINI_API_KEY


class TestGoogleInteractionsCreate:
    """Tests for creating interactions via litellm.interactions.create()."""

    def test_create_simple_string_input(self, api_key):
        """Test creating an interaction with a simple string input."""
        response = interactions.create(
            model="gemini/gemini-2.5-flash",
            input="Hello, what is 2 + 2?",
            api_key=api_key,
        )
        print("SIMPLE RESPONSE: ", response)
        assert response is not None
        assert response.id is not None or response.status is not None
        
        # Check outputs per OpenAPI spec
        if response.outputs:
            assert len(response.outputs) > 0
            print(f"Response outputs: {response.outputs}")
        
        # Check usage per OpenAPI spec
        if response.usage:
            print(f"Usage: {response.usage}")

    def test_create_with_content_list(self, api_key):
        """Test creating an interaction with a structured content list (Turn format)."""
        response = interactions.create(
            model="gemini/gemini-2.5-flash",
            input=[
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "What is the capital of France?"}]
                }
            ],
            api_key=api_key,
        )
        
        assert response is not None
        print(f"Response: {response}")

    def test_create_with_system_instruction(self, api_key):
        """Test creating an interaction with system_instruction (per OpenAPI spec)."""
        response = interactions.create(
            model="gemini/gemini-2.5-flash",
            input="What are you?",
            system_instruction="You are a helpful pirate assistant. Always respond like a pirate.",
            api_key=api_key,
        )
        
        assert response is not None
        print(f"Response with system_instruction: {response}")

    def test_create_with_tools(self, api_key):
        """Test creating an interaction with tools (per OpenAPI spec)."""
        response = interactions.create(
            model="gemini/gemini-2.5-flash",
            input="What's the weather in Boston?",
            tools=[
                {
                    "type": "function",
                    "name": "get_weather",
                    "description": "Get the weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string", "description": "The city name"}
                        },
                        "required": ["location"]
                    }
                }
            ],
            api_key=api_key,
        )
        
        assert response is not None
        # Check if status is requires_action (function call)
        print(f"Response status: {response.status}")
        print(f"Response outputs: {response.outputs}")

    @pytest.mark.asyncio
    async def test_acreate_simple(self, api_key):
        """Test async interaction creation."""
        response = await interactions.acreate(
            model="gemini/gemini-2.5-flash",
            input="What is the speed of light?",
            api_key=api_key,
        )
        
        assert response is not None
        print(f"Async response: {response}")


class TestGoogleInteractionsStreaming:
    """Tests for streaming interactions."""

    def test_create_streaming(self, api_key):
        """Test creating a streaming interaction."""
        response_stream = interactions.create(
            model="gemini/gemini-2.5-flash",
            input="Count from 1 to 5 slowly.",
            stream=True,
            api_key=api_key,
        )
        
        # Collect all chunks
        chunks = []
        for chunk in response_stream:
            chunks.append(chunk)
            print(f"Streaming chunk: {chunk}")
        
        assert len(chunks) > 0
        print(f"Total chunks received: {len(chunks)}")

    @pytest.mark.asyncio
    async def test_acreate_streaming(self, api_key):
        """Test async streaming interaction."""
        response_stream = await interactions.acreate(
            model="gemini/gemini-2.5-flash",
            input="Count from 1 to 3.",
            stream=True,
            api_key=api_key,
        )
        
        # Collect all chunks
        chunks = []
        async for chunk in response_stream:
            chunks.append(chunk)
            print(f"Async streaming chunk: {chunk}")
        
        assert len(chunks) > 0
        print(f"Total async chunks received: {len(chunks)}")


class TestGoogleInteractionsMultiTurn:
    """Tests for multi-turn conversations using Turn[] input."""

    def test_multi_turn_conversation(self, api_key):
        """Test a multi-turn conversation per OpenAPI spec (Turn[] format)."""
        response = interactions.create(
            model="gemini/gemini-2.5-flash",
            input=[
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "My name is Alice."}]
                },
                {
                    "role": "model",
                    "content": [{"type": "text", "text": "Hello Alice! Nice to meet you."}]
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "What is my name?"}]
                }
            ],
            api_key=api_key,
        )
        
        assert response is not None
        print(f"Multi-turn response: {response}")


class TestGoogleInteractionsAgent:
    """Tests for agent interactions (per OpenAPI spec)."""

    @pytest.mark.skip(reason="Deep research agent may not be available in all accounts")
    def test_create_agent_interaction(self, api_key):
        """Test creating an agent interaction per OpenAPI spec."""
        response = interactions.create(
            agent="deep-research-pro-preview-12-2025",
            input="Research the current state of quantum computing",
            api_key=api_key,
        )
        
        assert response is not None
        print(f"Agent response: {response}")


class TestGoogleInteractionsGetDelete:
    """Tests for get and delete operations."""

    @pytest.mark.skip(reason="Get/Delete require valid interaction IDs from previous calls")
    def test_get_interaction(self, api_key):
        """Test getting an interaction by ID."""
        # First create an interaction
        create_response = interactions.create(
            model="gemini/gemini-2.5-flash",
            input="Hello",
            api_key=api_key,
        )
        
        if create_response.id:
            # Then get it
            get_response = interactions.get(
                interaction_id=create_response.id,
                api_key=api_key,
            )
            assert get_response is not None
            print(f"Get response: {get_response}")

    @pytest.mark.skip(reason="Get/Delete require valid interaction IDs from previous calls")
    def test_delete_interaction(self, api_key):
        """Test deleting an interaction by ID."""
        # First create an interaction
        create_response = interactions.create(
            model="gemini/gemini-2.5-flash",
            input="Hello",
            api_key=api_key,
        )
        
        if create_response.id:
            # Then delete it
            delete_result = interactions.delete(
                interaction_id=create_response.id,
                api_key=api_key,
            )
            assert delete_result.success is True
            print(f"Delete result: {delete_result}")


class TestGoogleInteractionsErrorHandling:
    """Tests for error handling."""

    def test_invalid_model(self, api_key):
        """Test error handling for invalid model."""
        with pytest.raises(Exception):
            interactions.create(
                model="gemini/invalid-model-name-xyz",
                input="Hello",
                api_key=api_key,
            )

    def test_missing_model_and_agent(self, api_key):
        """Test error when neither model nor agent is provided."""
        with pytest.raises(Exception):  # Can be ValueError or APIConnectionError
            interactions.create(
                input="Hello",
                api_key=api_key,
            )


class TestGoogleInteractionsResponseStructure:
    """Tests to verify the response structure matches OpenAPI spec."""

    def test_response_has_expected_fields(self, api_key):
        """Test that the response has fields per OpenAPI spec."""
        response = interactions.create(
            model="gemini/gemini-2.5-flash",
            input="Hello",
            api_key=api_key,
        )
        
        # Check fields per OpenAPI spec
        assert hasattr(response, 'id')
        assert hasattr(response, 'object')
        assert hasattr(response, 'status')
        assert hasattr(response, 'outputs')
        assert hasattr(response, 'usage')
        assert hasattr(response, 'model') or hasattr(response, 'agent')
        assert hasattr(response, 'role')
        assert hasattr(response, 'created')
        assert hasattr(response, 'updated')
        
        print(f"Response structure: id={response.id}, status={response.status}, object={response.object}")


if __name__ == "__main__":
    # Run a quick smoke test
    print("Running Google Interactions API smoke test...")
    
    api_key = GEMINI_API_KEY
    if not api_key:
        print("GEMINI_API_KEY not set, skipping smoke test")
        exit(1)
    
    print("\n1. Testing basic interaction...")
    response = interactions.create(
        model="gemini/gemini-2.5-flash",
        input="What is 2 + 2?",
        api_key=api_key,
    )
    print(f"Response: {response}")
    
    print("\n2. Testing streaming interaction...")
    stream = interactions.create(
        model="gemini/gemini-2.5-flash",
        input="Count to 3.",
        stream=True,
        api_key=api_key,
    )
    print("Streaming response chunks:")
    for chunk in stream:
        print(f"  {chunk}")
    
    print("\n3. Testing async interaction...")
    async def test_async():
        response = await interactions.acreate(
            model="gemini/gemini-2.5-flash",
            input="Say hello!",
            api_key=api_key,
        )
        return response
    
    async_response = asyncio.run(test_async())
    print(f"Async response: {async_response}")
    
    print("\nSmoke test complete!")
