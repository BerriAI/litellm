import json
import os
import sys
from datetime import datetime
from unittest.mock import AsyncMock

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path

from litellm.types.utils import TextCompletionResponse


def test_convert_dict_to_text_completion_response():
    input_dict = {
        "id": "cmpl-ALVLPJgRkqpTomotoOMi3j0cAaL4L",
        "choices": [
            {
                "finish_reason": "length",
                "index": 0,
                "logprobs": {
                    "text_offset": [0, 5],
                    "token_logprobs": [None, -12.203847],
                    "tokens": ["hello", " crisp"],
                    "top_logprobs": [None, {",": -2.1568563}],
                },
                "text": "hello crisp",
            }
        ],
        "created": 1729688739,
        "model": "davinci-002",
        "object": "text_completion",
        "system_fingerprint": None,
        "usage": {
            "completion_tokens": 1,
            "prompt_tokens": 1,
            "total_tokens": 2,
            "completion_tokens_details": None,
            "prompt_tokens_details": None,
        },
    }

    response = TextCompletionResponse(**input_dict)

    assert response.id == "cmpl-ALVLPJgRkqpTomotoOMi3j0cAaL4L"
    assert len(response.choices) == 1
    assert response.choices[0].finish_reason == "length"
    assert response.choices[0].index == 0
    assert response.choices[0].text == "hello crisp"
    assert response.created == 1729688739
    assert response.model == "davinci-002"
    assert response.object == "text_completion"
    assert response.system_fingerprint is None
    assert response.usage.completion_tokens == 1
    assert response.usage.prompt_tokens == 1
    assert response.usage.total_tokens == 2
    assert response.usage.completion_tokens_details is None
    assert response.usage.prompt_tokens_details is None

    # Test logprobs
    assert response.choices[0].logprobs.text_offset == [0, 5]
    assert response.choices[0].logprobs.token_logprobs == [None, -12.203847]
    assert response.choices[0].logprobs.tokens == ["hello", " crisp"]
    assert response.choices[0].logprobs.top_logprobs == [None, {",": -2.1568563}]
