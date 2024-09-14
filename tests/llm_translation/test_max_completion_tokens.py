import json
import os
import sys

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
from datetime import datetime
from unittest.mock import AsyncMock
from dotenv import load_dotenv

load_dotenv()
import httpx
import pytest
from respx import MockRouter

import litellm
from litellm import Choices, Message, ModelResponse

# Adds the parent directory to the system path


def return_mocked_response(model: str):
    if model == "bedrock/mistral.mistral-large-2407-v1:0":
        return {
            "metrics": {"latencyMs": 316},
            "output": {
                "message": {
                    "content": [{"text": "Hello! How are you doing today? How can"}],
                    "role": "assistant",
                }
            },
            "stopReason": "max_tokens",
            "usage": {"inputTokens": 5, "outputTokens": 10, "totalTokens": 15},
        }


@pytest.mark.parametrize(
    "model",
    [
        "bedrock/mistral.mistral-large-2407-v1:0",
    ],
)
@pytest.mark.respx
@pytest.mark.asyncio()
async def test_bedrock_max_completion_tokens(model: str, respx_mock: MockRouter):
    """
    Tests that:
    - max_completion_tokens is passed as max_tokens to bedrock models
    """
    litellm.set_verbose = True

    mock_response = return_mocked_response(model)
    _model = model.split("/")[1]
    print("\n\nmock_response: ", mock_response)
    url = f"https://bedrock-runtime.us-west-2.amazonaws.com/model/{_model}/converse"
    mock_request = respx_mock.post(url).mock(
        return_value=httpx.Response(200, json=mock_response)
    )

    response = await litellm.acompletion(
        model=model,
        max_completion_tokens=10,
        messages=[{"role": "user", "content": "Hello!"}],
    )

    assert mock_request.called
    request_body = json.loads(mock_request.calls[0].request.content)

    print("request_body: ", request_body)

    assert request_body == {
        "messages": [{"role": "user", "content": [{"text": "Hello!"}]}],
        "additionalModelRequestFields": {},
        "system": [],
        "inferenceConfig": {"maxTokens": 10},
    }
    print(f"response: {response}")
    assert isinstance(response, ModelResponse)
