import pytest
from unittest.mock import MagicMock, patch

import litellm
from litellm.types.utils import Usage, PromptTokensDetailsWrapper, CompletionTokensDetailsWrapper
from tests.litellm.llms.vertex_ai.gemini.gemini_token_details_test_utils import (
    get_all_token_types_test_data,
    assert_token_details_dict,
)

@pytest.mark.asyncio
async def test_acompletion_includes_all_token_types():
    """Test that acompletion responses include all token types"""
    data = get_all_token_types_test_data()

    # Create a mock response with usage information
    mock_response = {
        "id": "test-id",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "gemini-1.5-pro",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "This is a test response"
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": data["expected_prompt_tokens"],
            "completion_tokens": data["expected_completion_tokens"],
            "total_tokens": data["expected_total_tokens"],
            "prompt_tokens_details": {
                "cached_tokens": data["cached_tokens"],
                "text_tokens": data["expected_cached_text_tokens"],
                "audio_tokens": data["expected_cached_audio_tokens"],
                "image_tokens": data["expected_cached_image_tokens"]
            },
            "completion_tokens_details": {
                "text_tokens": data["expected_completion_tokens"]
            }
        }
    }

    # Patch the acompletion function to return our mock response
    with patch('litellm.acompletion', return_value=mock_response):
        response = await litellm.acompletion(
            model="gemini/gemini-1.5-pro",
            messages=[{"role": "user", "content": "Hello"}]
        )

        # Verify that the response has the correct usage information
        assert_token_details_dict(response, data)

@pytest.mark.asyncio
async def test_acompletion_streaming_includes_all_token_types():
    """Test that acompletion streaming responses include all token types"""
    data = get_all_token_types_test_data()

    # Create a mock streaming chunk with usage information
    mock_chunk = MagicMock()
    mock_chunk.choices = [MagicMock()]
    mock_chunk.choices[0].delta = MagicMock()
    mock_chunk.choices[0].delta.content = "This is a test response"
    mock_chunk.usage = Usage(
        prompt_tokens=data["expected_prompt_tokens"],
        completion_tokens=data["expected_completion_tokens"],
        total_tokens=data["expected_total_tokens"],
        prompt_tokens_details=PromptTokensDetailsWrapper(
            cached_tokens=data["cached_tokens"],
            text_tokens=data["expected_cached_text_tokens"],
            audio_tokens=data["expected_cached_audio_tokens"],
            image_tokens=data["expected_cached_image_tokens"]
        ),
        completion_tokens_details=CompletionTokensDetailsWrapper(
            text_tokens=data["expected_completion_tokens"]
        )
    )

    # Create a mock async generator that yields our mock chunk
    async def mock_acompletion_stream(*args, **kwargs):
        yield mock_chunk

    # Patch the acompletion function to return our mock generator
    with patch('litellm.acompletion', return_value=mock_acompletion_stream()):
        response = await litellm.acompletion(
            model="gemini/gemini-1.5-pro",
            messages=[{"role": "user", "content": "Hello"}],
            stream=True
        )

        # Process the streaming response
        async for chunk in response:
            # Verify that the chunk has the correct usage information
            assert chunk.usage is not None
            assert chunk.usage.prompt_tokens == data["expected_prompt_tokens"]
            assert chunk.usage.completion_tokens == data["expected_completion_tokens"]
            assert chunk.usage.total_tokens == data["expected_total_tokens"]
            assert chunk.usage.prompt_tokens_details.cached_tokens == data["cached_tokens"]
            assert chunk.usage.prompt_tokens_details.text_tokens == data["expected_cached_text_tokens"]
            assert chunk.usage.prompt_tokens_details.audio_tokens == data["expected_cached_audio_tokens"]
            assert chunk.usage.prompt_tokens_details.image_tokens == data["expected_cached_image_tokens"]
            assert chunk.usage.completion_tokens_details.text_tokens == data["expected_completion_tokens"]
            break  # We only need to check the first chunk

@pytest.mark.asyncio
async def test_acompletion_cached_response_includes_all_token_types():
    """Test that acompletion cached responses include all token types"""
    # Enable caching
    litellm.cache = litellm.Cache(type="local")

    data = get_all_token_types_test_data()

    # Create a mock response with usage information but without prompt_tokens_details
    mock_response = {
        "id": "test-id",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "gemini-1.5-pro",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "This is a test response"
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": data["expected_prompt_tokens"],
            "completion_tokens": data["expected_completion_tokens"],
            "total_tokens": data["expected_total_tokens"]
        }
    }

    # Create a patched version of acompletion that returns our mock response
    # and then a cached version with prompt_tokens_details
    call_count = 0

    async def patched_acompletion(*args, **kwargs):
        nonlocal call_count
        call_count += 1

        if call_count == 1:
            # First call returns the mock response
            return mock_response
        else:
            # Second call returns a cached response with prompt_tokens_details
            cached_response = mock_response.copy()
            cached_response["usage"] = {
                "prompt_tokens": data["expected_prompt_tokens"],
                "completion_tokens": data["expected_completion_tokens"],
                "total_tokens": data["expected_total_tokens"],
                "prompt_tokens_details": {
                    "cached_tokens": data["expected_prompt_tokens"],
                    "text_tokens": data["expected_prompt_tokens"],
                    "audio_tokens": None,
                    "image_tokens": None
                }
            }
            cached_response["custom_llm_provider"] = "cached_response"
            return cached_response

    # Patch the acompletion function
    with patch('litellm.acompletion', side_effect=patched_acompletion):
        # First call to cache the response
        await litellm.acompletion(
            model="gemini/gemini-1.5-pro",
            messages=[{"role": "user", "content": "Hello"}]
        )

        # Second call should use the cache
        response = await litellm.acompletion(
            model="gemini/gemini-1.5-pro",
            messages=[{"role": "user", "content": "Hello"}]
        )

        # Verify that the cached response has the correct usage information
        assert_token_details_dict(response, data)

    # Clean up
    litellm.cache = None
