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


def test_convert_dict_to_embedding_response_basic():
    """Test basic conversion with all fields present."""
    response_object = {
        "model": "test-model",
        "object": "embedding",
        "data": [{"embedding": [0.1, 0.2, 0.3]}],
        "usage": {"prompt_tokens": 10, "total_tokens": 10},
    }
    result = convert_dict_to_embedding_response(
        model_response_object=None,
        response_object=response_object,
        start_time=None,
        end_time=None,
        hidden_params=None,
        _response_headers=None,
    )

    assert isinstance(result, EmbeddingResponse)
    assert result.model == "test-model"
    assert result.object == "embedding"
    assert result.data == [{"embedding": [0.1, 0.2, 0.3]}]
    assert result.usage.prompt_tokens == 10
    assert result.usage.total_tokens == 10
    assert result.usage.completion_tokens == 0


def test_convert_dict_to_embedding_response_with_existing_object():
    """Test conversion with an existing EmbeddingResponse object."""
    existing_response = EmbeddingResponse(model="existing-model")
    response_object = {
        "model": "new-model",
        "object": "embedding",
        "data": [{"embedding": [0.4, 0.5, 0.6]}],
    }
    result = convert_dict_to_embedding_response(
        model_response_object=existing_response,
        response_object=response_object,
        start_time=None,
        end_time=None,
        hidden_params=None,
        _response_headers=None,
    )

    assert result.model == "new-model"
    assert result.data == [{"embedding": [0.4, 0.5, 0.6]}]


def test_convert_dict_to_embedding_response_with_timing():
    """Test conversion with timing information."""
    response_object = {
        "data": [{"embedding": [0.7, 0.8, 0.9]}],
    }
    start_time = datetime.now()
    end_time = start_time + timedelta(seconds=2)

    result = convert_dict_to_embedding_response(
        model_response_object=None,
        response_object=response_object,
        start_time=start_time,
        end_time=end_time,
        hidden_params=None,
        _response_headers=None,
    )

    assert result._response_ms == pytest.approx(2000, rel=1e-2)


def test_convert_dict_to_embedding_response_with_hidden_params():
    """Test conversion with hidden parameters."""
    response_object = {
        "data": [{"embedding": [1.0, 1.1, 1.2]}],
    }
    hidden_params = {"api_key": "secret"}

    result = convert_dict_to_embedding_response(
        model_response_object=None,
        response_object=response_object,
        start_time=None,
        end_time=None,
        hidden_params=hidden_params,
        _response_headers=None,
    )

    assert result._hidden_params == hidden_params


def test_convert_dict_to_embedding_response_with_response_headers():
    """Test conversion with response headers."""
    response_object = {
        "data": [{"embedding": [1.3, 1.4, 1.5]}],
    }
    response_headers = {"Content-Type": "application/json"}

    result = convert_dict_to_embedding_response(
        model_response_object=None,
        response_object=response_object,
        start_time=None,
        end_time=None,
        hidden_params=None,
        _response_headers=response_headers,
    )

    assert result._response_headers == response_headers


def test_convert_dict_to_embedding_response_none_response():
    """Test error handling for None response object."""
    with pytest.raises(Exception, match="Error in response object format"):
        convert_dict_to_embedding_response(
            model_response_object=None,
            response_object=None,
            start_time=None,
            end_time=None,
            hidden_params=None,
            _response_headers=None,
        )


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
    assert result.data[0].b64_json == "base64encodedimagedata..."
    assert result.data[0].url == "https://example.com/generated_image.png"
    assert result.data[0].revised_prompt == "A beautiful sunset over mountains"
