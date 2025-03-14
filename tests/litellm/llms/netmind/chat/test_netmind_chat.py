from unittest.mock import MagicMock, patch
import pytest
import sys
import os
from openai.types.chat.chat_completion import ChatCompletion


sys.path.insert(0, os.path.abspath("../../../../../"))

import litellm
from litellm import completion
from tests.local_testing.test_completion import response_format_tests

user_message = "How are you doing today?"
messages = [{"content": user_message, "role": "user"}]


def mock_response():
    res = {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1677652288,
        "model": "gpt-4",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hello, this is a test response."},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 9, "completion_tokens": 12, "total_tokens": 21},
    }
    response = MagicMock()
    response.parse.return_value = ChatCompletion.model_validate(res)
    return response


@patch("litellm.llms.openai.openai.OpenAI")
def test_completion_netmind_chat(mock_openai):
    litellm.set_verbose = True
    model_name = "netmind/meta-llama/Llama-3.3-70B-Instruct"

    mock_client = mock_openai.return_value
    mock_completion = MagicMock()
    mock_completion.create.return_value = mock_response()
    mock_client.chat.completions.with_raw_response = mock_completion

    try:
        response = completion(
            model=model_name,
            messages=messages,
            max_tokens=10,
        )
        assert isinstance(response, litellm.ModelResponse)
        assert len(response.choices[0].message.content.strip()) > 0
        response_format_tests(response=response)
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")
