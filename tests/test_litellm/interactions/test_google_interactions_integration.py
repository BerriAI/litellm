"""
Integration tests for Google Interactions API.

Tests the litellm.interactions() and related methods against the Google AI Studio API.

Run with: pytest tests/test_litellm/interactions/test_google_interactions_integration.py -v
"""

import os
import sys
import asyncio

import pytest

sys.path.insert(0, os.path.abspath("../../.."))

import litellm
from litellm._logging import verbose_logger


# Test API key - should be set in environment or here for testing
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or "AIzaSyAYC1jjsRJhGFC6IcRU6qyoXdGMDaDBsW8"


@pytest.fixture
def api_key():
    """Fixture to provide the API key."""
    return GEMINI_API_KEY


class TestGoogleInteractionsCreate:
    """Tests for creating interactions (litellm.interactions and litellm.ainteractions)."""

    def test_interactions_create_simple_string(self, api_key):
        """Test creating an interaction with a simple string input."""
        response = litellm.interactions(
            model="gemini/gemini-2.0-flash",
            contents="Hello, what is 2 + 2?",
            api_key=api_key,
        )
        
        assert response is not None
        assert response.candidates is not None
        assert len(response.candidates) > 0
        assert response.candidates[0].content is not None
        assert response.candidates[0].content.parts is not None
        assert len(response.candidates[0].content.parts) > 0
        # Check that we got some text response
        first_part = response.candidates[0].content.parts[0]
        assert first_part.text is not None
        assert len(first_part.text) > 0
        
        # Check usage metadata
        assert response.usage_metadata is not None
        assert response.usage_metadata.prompt_token_count is not None
        assert response.usage_metadata.candidates_token_count is not None
        
        print(f"Response text: {first_part.text}")
        print(f"Usage: {response.usage_metadata}")

    def test_interactions_create_with_contents_list(self, api_key):
        """Test creating an interaction with a structured contents list."""
        response = litellm.interactions(
            model="gemini/gemini-2.0-flash",
            contents=[
                {
                    "role": "user",
                    "parts": [{"text": "What is the capital of France?"}]
                }
            ],
            api_key=api_key,
        )
        
        assert response is not None
        assert response.candidates is not None
        assert len(response.candidates) > 0
        
        first_part = response.candidates[0].content.parts[0]
        assert first_part.text is not None
        # Should mention Paris
        print(f"Response: {first_part.text}")

    def test_interactions_create_with_generation_config(self, api_key):
        """Test creating an interaction with generation configuration."""
        response = litellm.interactions(
            model="gemini/gemini-2.0-flash",
            contents="Tell me a very short joke.",
            generation_config={
                "temperature": 0.7,
                "max_output_tokens": 100,
            },
            api_key=api_key,
        )
        
        assert response is not None
        assert response.candidates is not None
        assert len(response.candidates) > 0
        
        first_part = response.candidates[0].content.parts[0]
        assert first_part.text is not None
        print(f"Joke: {first_part.text}")

    def test_interactions_create_with_system_instruction(self, api_key):
        """Test creating an interaction with a system instruction."""
        response = litellm.interactions(
            model="gemini/gemini-2.0-flash",
            contents="What are you?",
            system_instruction={
                "role": "user",
                "parts": [{"text": "You are a helpful pirate assistant. Always respond like a pirate."}]
            },
            api_key=api_key,
        )
        
        assert response is not None
        assert response.candidates is not None
        assert len(response.candidates) > 0
        
        first_part = response.candidates[0].content.parts[0]
        assert first_part.text is not None
        print(f"Pirate response: {first_part.text}")

    @pytest.mark.asyncio
    async def test_ainteractions_create_simple(self, api_key):
        """Test async interaction creation."""
        response = await litellm.ainteractions(
            model="gemini/gemini-2.0-flash",
            contents="What is the speed of light?",
            api_key=api_key,
        )
        
        assert response is not None
        assert response.candidates is not None
        assert len(response.candidates) > 0
        
        first_part = response.candidates[0].content.parts[0]
        assert first_part.text is not None
        print(f"Async response: {first_part.text}")


class TestGoogleInteractionsStreaming:
    """Tests for streaming interactions."""

    def test_interactions_create_streaming(self, api_key):
        """Test creating a streaming interaction."""
        response_stream = litellm.interactions(
            model="gemini/gemini-2.0-flash",
            contents="Count from 1 to 5 slowly.",
            stream=True,
            api_key=api_key,
        )
        
        # Collect all chunks
        chunks = []
        for chunk in response_stream:
            chunks.append(chunk)
            if chunk.candidates and chunk.candidates[0].content:
                parts = chunk.candidates[0].content.parts
                if parts:
                    print(f"Streaming chunk: {parts[0].text}")
        
        assert len(chunks) > 0
        print(f"Total chunks received: {len(chunks)}")

    @pytest.mark.asyncio
    async def test_ainteractions_create_streaming(self, api_key):
        """Test async streaming interaction."""
        response_stream = await litellm.ainteractions(
            model="gemini/gemini-2.0-flash",
            contents="Count from 1 to 3.",
            stream=True,
            api_key=api_key,
        )
        
        # Collect all chunks
        chunks = []
        async for chunk in response_stream:
            chunks.append(chunk)
            if chunk.candidates and chunk.candidates[0].content:
                parts = chunk.candidates[0].content.parts
                if parts:
                    print(f"Async streaming chunk: {parts[0].text}")
        
        assert len(chunks) > 0
        print(f"Total async chunks received: {len(chunks)}")


class TestGoogleInteractionsMultiTurn:
    """Tests for multi-turn conversations."""

    def test_interactions_multi_turn_conversation(self, api_key):
        """Test a multi-turn conversation."""
        response = litellm.interactions(
            model="gemini/gemini-2.0-flash",
            contents=[
                {
                    "role": "user",
                    "parts": [{"text": "My name is Alice."}]
                },
                {
                    "role": "model",
                    "parts": [{"text": "Hello Alice! Nice to meet you. How can I help you today?"}]
                },
                {
                    "role": "user",
                    "parts": [{"text": "What is my name?"}]
                }
            ],
            api_key=api_key,
        )
        
        assert response is not None
        assert response.candidates is not None
        assert len(response.candidates) > 0
        
        first_part = response.candidates[0].content.parts[0]
        assert first_part.text is not None
        # Should mention Alice
        print(f"Multi-turn response: {first_part.text}")


class TestGoogleInteractionsErrorHandling:
    """Tests for error handling."""

    def test_interactions_invalid_model(self, api_key):
        """Test error handling for invalid model."""
        with pytest.raises(Exception):
            litellm.interactions(
                model="gemini/invalid-model-name-xyz",
                contents="Hello",
                api_key=api_key,
            )

    def test_interactions_empty_contents(self, api_key):
        """Test error handling for empty contents."""
        # This should either raise an error or return an empty response
        # Depending on API behavior
        try:
            response = litellm.interactions(
                model="gemini/gemini-2.0-flash",
                contents="",
                api_key=api_key,
            )
            # If it succeeds, just check it's valid
            assert response is not None
        except Exception as e:
            # Expected behavior - empty content may cause an error
            print(f"Expected error for empty content: {e}")


# Additional test for verifying the response structure
class TestGoogleInteractionsResponseStructure:
    """Tests to verify the response structure matches the expected format."""

    def test_response_has_expected_fields(self, api_key):
        """Test that the response has all expected fields."""
        response = litellm.interactions(
            model="gemini/gemini-2.0-flash",
            contents="Hello",
            api_key=api_key,
        )
        
        # Check main response fields
        assert hasattr(response, 'candidates')
        assert hasattr(response, 'usage_metadata')
        assert hasattr(response, 'model')
        
        # Check candidates structure
        if response.candidates:
            candidate = response.candidates[0]
            assert hasattr(candidate, 'content')
            assert hasattr(candidate, 'finish_reason')
            assert hasattr(candidate, 'safety_ratings')
            
            # Check content structure
            if candidate.content:
                assert hasattr(candidate.content, 'role')
                assert hasattr(candidate.content, 'parts')
                
                # Check parts structure
                if candidate.content.parts:
                    part = candidate.content.parts[0]
                    assert hasattr(part, 'text')
        
        # Check usage metadata
        if response.usage_metadata:
            assert hasattr(response.usage_metadata, 'prompt_token_count')
            assert hasattr(response.usage_metadata, 'candidates_token_count')
            assert hasattr(response.usage_metadata, 'total_token_count')


if __name__ == "__main__":
    # Run a quick smoke test
    print("Running Google Interactions API smoke test...")
    
    api_key = GEMINI_API_KEY
    
    print("\n1. Testing basic interaction...")
    response = litellm.interactions(
        model="gemini/gemini-2.0-flash",
        contents="What is 2 + 2?",
        api_key=api_key,
    )
    print(f"Response: {response.candidates[0].content.parts[0].text}")
    print(f"Tokens used: {response.usage_metadata}")
    
    print("\n2. Testing streaming interaction...")
    stream = litellm.interactions(
        model="gemini/gemini-2.0-flash",
        contents="Count to 3.",
        stream=True,
        api_key=api_key,
    )
    print("Streaming response: ", end="")
    for chunk in stream:
        if chunk.candidates and chunk.candidates[0].content and chunk.candidates[0].content.parts:
            text = chunk.candidates[0].content.parts[0].text
            if text:
                print(text, end="", flush=True)
    print()
    
    print("\n3. Testing async interaction...")
    async def test_async():
        response = await litellm.ainteractions(
            model="gemini/gemini-2.0-flash",
            contents="Say hello!",
            api_key=api_key,
        )
        return response
    
    async_response = asyncio.run(test_async())
    print(f"Async response: {async_response.candidates[0].content.parts[0].text}")
    
    print("\nSmoke test complete!")
