#!/usr/bin/env python3
"""
Test to verify the Google GenAI generate_content handler functionality
"""
import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(
    0, os.path.abspath("../../../..")
)  # Adds the parent directory to the system path

import litellm
from litellm.google_genai.adapters.handler import GenerateContentToCompletionHandler
from litellm.google_genai.adapters.transformation import GoogleGenAIAdapter
from litellm.types.utils import ModelResponse


def test_non_stream_response_when_stream_requested_sync():
    """
    Test that when a non-stream response is returned but streaming was requested,
    the sync handler correctly transforms it to generate_content format.
    """
    from litellm.types.utils import Choices
    
    # Mock a non-stream response (ModelResponse with valid choices)
    mock_response = ModelResponse(
        id="test-123",
        choices=[
            Choices(
                index=0,
                message={
                    "role": "assistant",
                    "content": "Hello, world!"
                },
                finish_reason="stop"
            )
        ],
        created=1234567890,
        model="gpt-3.5-turbo",
        object="chat.completion"
    )
    
    # Create an instance of the adapter
    adapter = GoogleGenAIAdapter()
    
    # Test the adapter's translate_completion_to_generate_content method directly
    result = adapter.translate_completion_to_generate_content(mock_response)
    
    # Verify the result is a valid Google GenAI format response
    assert "candidates" in result
    assert isinstance(result["candidates"], list)
    assert len(result["candidates"]) > 0
    candidate = result["candidates"][0]
    assert "content" in candidate
    assert "parts" in candidate["content"]
    assert isinstance(candidate["content"]["parts"], list)
    assert len(candidate["content"]["parts"]) > 0
    assert "text" in candidate["content"]["parts"][0]
    assert candidate["content"]["parts"][0]["text"] == "Hello, world!"


@pytest.mark.asyncio
async def test_non_stream_response_when_stream_requested_async():
    """
    Test that when a non-stream response is returned but streaming was requested,
    the async handler correctly transforms it to generate_content format.
    """
    from litellm.types.utils import Choices
    
    # Mock a non-stream response (ModelResponse with valid choices)
    mock_response = ModelResponse(
        id="test-123",
        choices=[
            Choices(
                index=0,
                message={
                    "role": "assistant",
                    "content": "Hello, world!"
                },
                finish_reason="stop"
            )
        ],
        created=1234567890,
        model="gpt-3.5-turbo",
        object="chat.completion"
    )
    
    # Create an instance of the adapter
    adapter = GoogleGenAIAdapter()
    
    # Test the adapter's translate_completion_to_generate_content method directly
    result = adapter.translate_completion_to_generate_content(mock_response)
    
    # Verify the result is a valid Google GenAI format response
    assert "candidates" in result
    assert isinstance(result["candidates"], list)
    assert len(result["candidates"]) > 0
    candidate = result["candidates"][0]
    assert "content" in candidate
    assert "parts" in candidate["content"]
    assert isinstance(candidate["content"]["parts"], list)
    assert len(candidate["content"]["parts"]) > 0
    assert "text" in candidate["content"]["parts"][0]
    assert candidate["content"]["parts"][0]["text"] == "Hello, world!"


def test_stream_response_when_stream_requested_sync():
    """
    Test that when a stream response is returned and streaming was requested,
    the sync handler correctly transforms it to generate_content streaming format.
    """
    # Mock a stream response
    mock_stream = MagicMock()
    mock_stream.__iter__ = MagicMock(return_value=iter([]))
    
    # Mock the GoogleGenAIAdapter's translate_completion_output_params_streaming method
    with patch.object(
        GoogleGenAIAdapter, 
        "translate_completion_output_params_streaming", 
        return_value=mock_stream
    ) as mock_translate:
        with patch("litellm.completion", return_value=mock_stream):
            # Call the handler with stream=True
            result = GenerateContentToCompletionHandler.generate_content_handler(
                model="gemini-pro",
                contents=[{"role": "user", "parts": [{"text": "Hello"}]}],
                litellm_params={},  # Empty dict for params
                stream=True
            )
            
            # Verify that translate_completion_output_params_streaming was called
            mock_translate.assert_called_once_with(mock_stream)
            # Verify the result is the transformed stream
            assert result == mock_stream


@pytest.mark.asyncio
async def test_stream_response_when_stream_requested_async():
    """
    Test that when a stream response is returned and streaming was requested,
    the async handler correctly transforms it to generate_content streaming format.
    """
    # Mock a stream response
    mock_stream = MagicMock()
    mock_stream.__aiter__ = AsyncMock(return_value=iter([]))  # Return an empty async iterator
    
    # Mock the GoogleGenAIAdapter's translate_completion_output_params_streaming method
    with patch.object(
        GoogleGenAIAdapter, 
        "translate_completion_output_params_streaming", 
        return_value=mock_stream
    ) as mock_translate:
        with patch("litellm.acompletion", return_value=mock_stream):
            # Call the handler with stream=True
            result = await GenerateContentToCompletionHandler.async_generate_content_handler(
                model="gemini-pro",
                contents=[{"role": "user", "parts": [{"text": "Hello"}]}],
                litellm_params={},  # Empty dict for params
                stream=True
            )
            
            # Verify that translate_completion_output_params_streaming was called
            mock_translate.assert_called_once_with(mock_stream)
            # Verify the result is the transformed stream
            assert result == mock_stream


def test_stream_transformation_error_sync():
    """
    Test that when a stream transformation fails, the sync handler raises a ValueError.
    """
    # Mock a stream response
    mock_stream = MagicMock()
    mock_stream.__iter__ = MagicMock(return_value=iter([]))
    
    # Mock the GoogleGenAIAdapter's translate_completion_output_params_streaming method to return None
    with patch.object(
        GoogleGenAIAdapter, 
        "translate_completion_output_params_streaming", 
        return_value=None
    ):
        with patch("litellm.completion", return_value=mock_stream):
            # Call the handler with stream=True and expect a ValueError
            with pytest.raises(ValueError, match="Failed to transform streaming response"):
                GenerateContentToCompletionHandler.generate_content_handler(
                    model="gemini-pro",
                    contents=[{"role": "user", "parts": [{"text": "Hello"}]}],
                    litellm_params={},  # Empty dict for params
                    stream=True
                )


@pytest.mark.asyncio
async def test_stream_transformation_error_async():
    """
    Test that when a stream transformation fails, the async handler raises a ValueError.
    """
    # Mock a stream response
    mock_stream = MagicMock()
    mock_stream.__aiter__ = AsyncMock(return_value=mock_stream)
    
    # Mock the GoogleGenAIAdapter's translate_completion_output_params_streaming method to return None
    with patch.object(
        GoogleGenAIAdapter, 
        "translate_completion_output_params_streaming", 
        return_value=None
    ):
        with patch("litellm.acompletion", return_value=mock_stream):
            # Call the handler with stream=True and expect a ValueError
            with pytest.raises(ValueError, match="Failed to transform streaming response"):
                await GenerateContentToCompletionHandler.async_generate_content_handler(
                    model="gemini-pro",
                    contents=[{"role": "user", "parts": [{"text": "Hello"}]}],
                    litellm_params={},  # Empty dict for params
                    stream=True
                )