import json
import os
import sys
from datetime import datetime

sys.path.insert(
    0, os.path.abspath("../../")
)  # Adds the parent directory to the system path


from litellm.litellm_core_utils.llm_response_utils.convert_dict_to_embedding_response import (
    convert_dict_to_embedding_response,
)
from litellm.utils import EmbeddingResponse
from litellm.litellm_core_utils.llm_response_utils.convert_dict_to_image_generation_response import (
    convert_dict_to_image_generation_response,
)
from litellm.utils import ImageResponse

import litellm
import pytest
from datetime import timedelta


def test_convert_dict_to_image_generation_response_basic():
    """Test basic conversion with all fields present."""
    response_object = {
        "created": 1234567890,
        "data": [{"url": "https://example.com/image.jpg"}],
    }
    result = convert_dict_to_image_generation_response(
        model_response_object=None, response_object=response_object, hidden_params=None
    )

    assert isinstance(result, ImageResponse)
    assert result.created == 1234567890
    assert result.data == [{"url": "https://example.com/image.jpg"}]


def test_convert_dict_to_image_generation_response_with_existing_object():
    """Test conversion with an existing ImageResponse object."""
    existing_response = ImageResponse(created=1111111111)
    response_object = {"data": [{"url": "https://example.com/new_image.jpg"}]}
    result = convert_dict_to_image_generation_response(
        model_response_object=existing_response,
        response_object=response_object,
        hidden_params=None,
    )

    assert result.created == 1111111111
    assert result.data == [{"url": "https://example.com/new_image.jpg"}]


def test_convert_dict_to_image_generation_response_with_hidden_params():
    """Test conversion with hidden parameters."""
    response_object = {"data": [{"url": "https://example.com/image.jpg"}]}
    hidden_params = {"api_key": "secret"}

    result = convert_dict_to_image_generation_response(
        model_response_object=None,
        response_object=response_object,
        hidden_params=hidden_params,
    )

    assert result._hidden_params == hidden_params


def test_convert_dict_to_image_generation_response_none_response():
    """Test error handling for None response object."""
    with pytest.raises(Exception, match="Error in response object format"):
        convert_dict_to_image_generation_response(
            model_response_object=None, response_object=None, hidden_params=None
        )


def test_convert_dict_to_image_generation_response_full_openai_format():
    """Test conversion with full OpenAI API response format."""
    response_object = {
        "created": 1589478378,
        "data": [
            {
                "b64_json": "base64encodedimagedata...",
                "url": "https://example.com/generated_image.png",
                "revised_prompt": "A beautiful sunset over mountains",
            }
        ],
    }
    result = convert_dict_to_image_generation_response(
        model_response_object=None, response_object=response_object, hidden_params=None
    )

    assert isinstance(result, ImageResponse)
    assert result.created == 1589478378
    assert len(result.data) == 1
    assert result.data[0]["b64_json"] == "base64encodedimagedata..."
    assert result.data[0]["url"] == "https://example.com/generated_image.png"
    assert result.data[0]["revised_prompt"] == "A beautiful sunset over mountains"
