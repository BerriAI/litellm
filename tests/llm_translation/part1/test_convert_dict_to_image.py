import json
import os
import sys
from datetime import datetime

sys.path.insert(
    0, os.path.abspath("../../")
)  # Adds the parent directory to the system path

import litellm
import pytest
from datetime import timedelta
from litellm.types.utils import ImageResponse, ImageObject
from litellm.litellm_core_utils.llm_response_utils.convert_dict_to_response import (
    LiteLLMResponseObjectHandler,
)


def test_convert_to_image_response_basic():
    # Test basic conversion with minimal input
    response_dict = {
        "created": 1234567890,
        "data": [{"url": "http://example.com/image.jpg"}],
    }

    result = LiteLLMResponseObjectHandler.convert_to_image_response(response_dict)

    assert isinstance(result, ImageResponse)
    assert result.created == 1234567890
    assert result.data[0].url == "http://example.com/image.jpg"


def test_convert_to_image_response_with_hidden_params():
    # Test with hidden params
    response_dict = {
        "created": 1234567890,
        "data": [{"url": "http://example.com/image.jpg"}],
    }
    hidden_params = {"api_key": "test_key"}

    result = LiteLLMResponseObjectHandler.convert_to_image_response(
        response_dict, hidden_params=hidden_params
    )

    assert result._hidden_params == {"api_key": "test_key"}


def test_convert_to_image_response_multiple_images():
    # Test handling multiple images in response
    response_dict = {
        "created": 1234567890,
        "data": [
            {"url": "http://example.com/image1.jpg"},
            {"url": "http://example.com/image2.jpg"},
        ],
    }

    result = LiteLLMResponseObjectHandler.convert_to_image_response(response_dict)

    assert len(result.data) == 2
    assert result.data[0].url == "http://example.com/image1.jpg"
    assert result.data[1].url == "http://example.com/image2.jpg"


def test_convert_to_image_response_with_b64_json():
    # Test handling b64_json in response
    response_dict = {
        "created": 1234567890,
        "data": [{"b64_json": "base64encodedstring"}],
    }

    result = LiteLLMResponseObjectHandler.convert_to_image_response(response_dict)

    assert result.data[0].b64_json == "base64encodedstring"


def test_convert_to_image_response_with_extra_fields():
    response_dict = {
        "created": 1234567890,
        "data": [
            {
                "url": "http://example.com/image1.jpg",
                "content_filter_results": {"category": "violence", "flagged": True},
            },
            {
                "url": "http://example.com/image2.jpg",
                "content_filter_results": {"category": "violence", "flagged": True},
            },
        ],
    }

    result = LiteLLMResponseObjectHandler.convert_to_image_response(response_dict)

    assert result.data[0].url == "http://example.com/image1.jpg"
    assert result.data[1].url == "http://example.com/image2.jpg"


def test_convert_to_image_response_with_extra_fields_2():
    """
    Date from a non-OpenAI API could have some obscure field in addition to the expected ones. This should not break the conversion.
    """
    response_dict = {
        "created": 1234567890,
        "data": [
            {
                "url": "http://example.com/image1.jpg",
                "very_obscure_field": "some_value",
            },
            {
                "url": "http://example.com/image2.jpg",
                "very_obscure_field2": "some_other_value",
            },
        ],
    }

    result = LiteLLMResponseObjectHandler.convert_to_image_response(response_dict)

    assert result.data[0].url == "http://example.com/image1.jpg"
    assert result.data[1].url == "http://example.com/image2.jpg"


def test_convert_to_image_response_with_none_usage_fields():
    """
    Test handling of None values in usage fields, specifically for gpt-image-1 responses.
    
    This test verifies the fix for the bug where gpt-image-1 returns None values
    for usage statistics fields, which caused Pydantic validation errors.
    The fix should clean these None values and let ImageResponse constructor
    handle the default values.
    """
    response_dict = {
        "created": 1234567890,
        "data": [{"b64_json": "base64encodedstring"}],
        "usage": {
            "input_tokens": None,  # gpt-image-1 returns None instead of integer
            "input_tokens_details": None,  # gpt-image-1 returns None instead of object
            "output_tokens": None,  # gpt-image-1 returns None instead of integer
            "total_tokens": None,  # gpt-image-1 returns None instead of integer
        }
    }

    # This should not raise a ValidationError
    result = LiteLLMResponseObjectHandler.convert_to_image_response(response_dict)

    assert isinstance(result, ImageResponse)
    assert result.created == 1234567890
    assert result.data[0].b64_json == "base64encodedstring"
    
    # Usage should be properly initialized with default values
    assert result.usage is not None
    assert result.usage.input_tokens == 0
    assert result.usage.output_tokens == 0
    assert result.usage.total_tokens == 0
    assert result.usage.input_tokens_details is not None
    assert result.usage.input_tokens_details.image_tokens == 0
    assert result.usage.input_tokens_details.text_tokens == 0


def test_convert_to_image_response_with_partial_none_usage_fields():
    """
    Test handling of mixed None and valid values in usage fields.
    """
    response_dict = {
        "created": 1234567890,
        "data": [{"b64_json": "base64encodedstring"}],
        "usage": {
            "input_tokens": 10,  # Valid value
            "input_tokens_details": None,  # None value (should be cleaned)
            "output_tokens": None,  # None value (should be cleaned)
            "total_tokens": 10,  # Valid value
        }
    }

    # This should not raise a ValidationError
    result = LiteLLMResponseObjectHandler.convert_to_image_response(response_dict)

    assert isinstance(result, ImageResponse)
    assert result.created == 1234567890
    assert result.data[0].b64_json == "base64encodedstring"
    
    # Usage should be properly initialized with defaults where needed
    # Valid values should be preserved, None values should be cleaned and use defaults
    assert result.usage is not None
    assert result.usage.input_tokens == 10  # Valid value should be preserved
    assert result.usage.output_tokens == 0  # None value should become 0
    assert result.usage.total_tokens == 10  # Calculated as input_tokens + output_tokens (10 + 0)
    assert result.usage.input_tokens_details is not None
    assert result.usage.input_tokens_details.image_tokens == 0
    assert result.usage.input_tokens_details.text_tokens == 0


def test_convert_to_image_response_with_valid_usage_fields():
    """
    Test that valid usage fields are preserved correctly.
    """
    response_dict = {
        "created": 1234567890,
        "data": [{"b64_json": "base64encodedstring"}],
        "usage": {
            "input_tokens": 50,
            "input_tokens_details": {
                "image_tokens": 30,
                "text_tokens": 20,
            },
            "output_tokens": 10,
            "total_tokens": 60,
        }
    }

    result = LiteLLMResponseObjectHandler.convert_to_image_response(response_dict)

    assert isinstance(result, ImageResponse)
    assert result.created == 1234567890
    assert result.data[0].b64_json == "base64encodedstring"
    
    # Valid usage fields should be preserved
    assert result.usage is not None
    assert result.usage.input_tokens == 50
    assert result.usage.output_tokens == 10
    assert result.usage.total_tokens == 60
    assert result.usage.input_tokens_details is not None
    assert result.usage.input_tokens_details.image_tokens == 30
    assert result.usage.input_tokens_details.text_tokens == 20
