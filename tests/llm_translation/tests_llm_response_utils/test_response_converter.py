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

from litellm.litellm_core_utils.llm_response_utils.convert_dict_to_chat_completion_response import (
    convert_dict_to_chat_completion_response,
    _process_choices_in_response,
)
from litellm.types.utils import (
    ModelResponse,
    Message,
    Choices,
    PromptTokensDetails,
    CompletionTokensDetails,
    ChatCompletionMessageToolCall,
)

from litellm.litellm_core_utils.llm_response_utils.response_converter import (
    convert_to_model_response_object,
    _handle_error_in_response_object,
    _get_openai_headers,
    _set_headers_in_hidden_params,
)


def test_convert_to_model_response_object():
    """
    Test the convert_to_model_response_object function

    NOTE: unit tests for each response type can be found in this directory
    """
    response_object = {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1677652288,
        "model": "gpt-3.5-turbo-0613",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello, how can I assist you today?",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 9, "completion_tokens": 12, "total_tokens": 21},
    }

    result = convert_to_model_response_object(
        response_object=response_object,
        response_type="completion",
        model_response_object=ModelResponse(),
    )

    assert isinstance(result, ModelResponse)
    assert result.id == "chatcmpl-123"
    assert result.choices[0].message.content == "Hello, how can I assist you today?"


def test_handle_error_in_response_object():
    """
    If `error` key is in the response object, then raise an exception
    """
    error_response = {"error": {"code": 400, "message": "Bad Request"}}

    with pytest.raises(Exception) as exc_info:
        _handle_error_in_response_object(error_response)

    assert exc_info.value.status_code == 400
    assert exc_info.value.message == "Bad Request"


def test_get_openai_headers():
    """
    should only return the openai headers
    """
    response_headers = {
        "x-ratelimit-limit-requests": "100",
        "x-ratelimit-remaining-requests": "99",
        "x-ratelimit-limit-tokens": "1000",
        "x-ratelimit-remaining-tokens": "950",
        "other-header": "value",
    }

    result = _get_openai_headers(response_headers)

    assert result == {
        "x-ratelimit-limit-requests": "100",
        "x-ratelimit-remaining-requests": "99",
        "x-ratelimit-limit-tokens": "1000",
        "x-ratelimit-remaining-tokens": "950",
    }


def test_set_headers_in_hidden_params():
    """
    1. should add the openai header as is
    2. should add the custom header with the llm_provider prefix
    """
    hidden_params = {}
    response_headers = {"x-ratelimit-limit-requests": "100", "other-header": "value"}

    result = _set_headers_in_hidden_params(hidden_params, response_headers)

    assert "additional_headers" in result
    assert (
        result["additional_headers"]["llm_provider-x-ratelimit-limit-requests"] == "100"
    )
    assert result["additional_headers"]["llm_provider-other-header"] == "value"
    assert result["additional_headers"]["x-ratelimit-limit-requests"] == "100"
