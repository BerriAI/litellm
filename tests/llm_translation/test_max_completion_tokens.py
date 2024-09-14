import json
import os
import sys
from datetime import datetime
from unittest.mock import AsyncMock

import httpx
import pytest
from respx import MockRouter

import litellm
from litellm import Choices, Message, ModelResponse

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path


@pytest.mark.asyncio
@pytest.mark.respx
@pytest.mark.parametrize(
    "model",
    [
        "bedrock/mistral.mistral-large-2407-v1:0",
        "anthropic.claude-3-sonnet-20240229-v1:0",
    ],
)
async def test_bedrock_max_completion_tokens(respx_mock: MockRouter, model: str):
    """
    Tests that:
    - max_completion_tokens is passed as max_tokens to bedrock models
    """
    litellm.set_verbose = True

    mock_response = {
        "response_id": "47f1a66e/115fd033-054a-410a-ab45-245561477bf2",
        "text": "I'm an AI chatbot, so I don't have feelings or emotions in the traditional sense. But I'm always ready to assist and provide helpful responses! How can I help you today?",
        "generation_id": "ed1667d3-2d33-4cd4-9b8d-ad8eb813c2a4",
        "chat_history": [
            {"role": "USER", "message": "Hey! how's it going?"},
            {
                "role": "CHATBOT",
                "message": "I'm an AI chatbot, so I don't have feelings or emotions in the traditional sense. But I'm always ready to assist and provide helpful responses! How can I help you today?",
            },
        ],
        "finish_reason": "COMPLETE",
    }
    url = f"https://bedrock-runtime.us-west-2.amazonaws.com/model/{model}/converse"
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
        "model": model,
        "maxTokens": 10,
        "messages": [{"role": "user", "content": "Hello!"}],
    }

    print(f"response: {response}")
    assert isinstance(response, ModelResponse)
