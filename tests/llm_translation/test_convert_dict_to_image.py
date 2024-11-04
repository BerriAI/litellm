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
