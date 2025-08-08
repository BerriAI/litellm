"""
Test cases for handling missing 'choices' field in Azure OpenAI responses.
Addresses issue #13139 where Azure can return responses without choices field.
"""
import os
import sys
import pytest

sys.path.insert(
    0, os.path.abspath("../../../")
)  # Adds the parent directory to the system path

from litellm.litellm_core_utils.llm_response_utils.convert_dict_to_response import (
    convert_to_model_response_object,
    convert_to_streaming_response,
    convert_to_streaming_response_async,
)
from litellm.types.utils import ModelResponse


def test_convert_dict_missing_choices_field():
    """Test that missing 'choices' field raises descriptive exception."""
    # Azure response without choices field (e.g., content filtering response)
    response_without_choices = {
        "id": "test-id",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "gpt-4o",
        "prompt_filter_results": [
            {
                "prompt_index": 0,
                "content_filter_results": {
                    "hate": {"filtered": False, "severity": "safe"},
                    "self_harm": {"filtered": False, "severity": "safe"},
                    "sexual": {"filtered": False, "severity": "safe"},
                    "violence": {"filtered": False, "severity": "safe"}
                }
            }
        ]
    }
    
    model_response = ModelResponse()
    
    with pytest.raises(Exception) as exc_info:
        convert_to_model_response_object(
            response_object=response_without_choices,
            model_response_object=model_response,
            response_type="completion"
        )
    
    assert "missing 'choices' field" in str(exc_info.value)
    assert "Response:" in str(exc_info.value)


def test_convert_dict_none_choices_field():
    """Test that None 'choices' field raises descriptive exception."""
    response_with_none_choices = {
        "id": "test-id",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "gpt-4o",
        "choices": None,
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 0,
            "total_tokens": 10
        }
    }
    
    model_response = ModelResponse()
    
    with pytest.raises(Exception) as exc_info:
        convert_to_model_response_object(
            response_object=response_with_none_choices,
            model_response_object=model_response,
            response_type="completion"
        )
    
    assert "missing 'choices' field" in str(exc_info.value)
    assert "Response:" in str(exc_info.value)


def test_streaming_response_missing_choices():
    """Test streaming response with missing 'choices' field."""
    response_without_choices = {
        "id": "test-id",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "gpt-4o"
    }
    
    with pytest.raises(Exception) as exc_info:
        # convert_to_streaming_response returns a generator
        gen = convert_to_streaming_response(response_object=response_without_choices)
        # Consume the generator to trigger the exception
        next(gen)
    
    assert "missing 'choices' field" in str(exc_info.value)


@pytest.mark.asyncio
async def test_async_streaming_response_missing_choices():
    """Test async streaming response with missing 'choices' field."""
    response_without_choices = {
        "id": "test-id",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "gpt-4o"
    }
    
    with pytest.raises(Exception) as exc_info:
        # convert_to_streaming_response_async returns an async generator
        gen = convert_to_streaming_response_async(response_object=response_without_choices)
        # Consume the async generator to trigger the exception
        await gen.__anext__()
    
    assert "missing 'choices' field" in str(exc_info.value)


def test_valid_response_still_works():
    """Ensure valid responses continue to work correctly."""
    valid_response = {
        "id": "test-id",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "gpt-4o",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Hello, how can I help you?"
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 7,
            "total_tokens": 17
        }
    }
    
    model_response = ModelResponse()
    
    # Should not raise exception
    result = convert_to_model_response_object(
        response_object=valid_response,
        model_response_object=model_response,
        response_type="completion"
    )
    
    assert result.id == "test-id"
    assert len(result.choices) == 1
    assert result.choices[0].message.content == "Hello, how can I help you?"