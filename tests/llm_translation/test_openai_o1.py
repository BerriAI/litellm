import json
import os
import sys
from datetime import datetime
from unittest.mock import AsyncMock

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path


import httpx
import pytest
from respx import MockRouter

import litellm
from litellm import Choices, Message, ModelResponse


@pytest.mark.asyncio
@pytest.mark.respx
async def test_o1_handle_system_role(respx_mock: MockRouter):
    """
    Tests that:
    - max_tokens is translated to 'max_completion_tokens'
    - role 'system' is translated to 'user'
    """
    litellm.set_verbose = True

    mock_response = ModelResponse(
        id="cmpl-mock",
        choices=[Choices(message=Message(content="Mocked response", role="assistant"))],
        created=int(datetime.now().timestamp()),
        model="o1-preview",
    )

    mock_request = respx_mock.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=mock_response.dict())
    )

    response = await litellm.acompletion(
        model="o1-preview",
        max_tokens=10,
        messages=[{"role": "system", "content": "Hello!"}],
    )

    assert mock_request.called
    request_body = json.loads(mock_request.calls[0].request.content)

    print("request_body: ", request_body)

    assert request_body == {
        "model": "o1-preview",
        "max_completion_tokens": 10,
        "messages": [{"role": "user", "content": "Hello!"}],
    }

    print(f"response: {response}")
    assert isinstance(response, ModelResponse)


@pytest.mark.asyncio
@pytest.mark.respx
@pytest.mark.parametrize("model", ["gpt-4", "gpt-4-0314", "gpt-4-32k", "o1-preview"])
async def test_o1_max_completion_tokens(respx_mock: MockRouter, model: str):
    """
    Tests that:
    - max_completion_tokens is passed directly to OpenAI chat completion models
    """
    litellm.set_verbose = True

    mock_response = ModelResponse(
        id="cmpl-mock",
        choices=[Choices(message=Message(content="Mocked response", role="assistant"))],
        created=int(datetime.now().timestamp()),
        model=model,
    )

    mock_request = respx_mock.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=mock_response.dict())
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
        "max_completion_tokens": 10,
        "messages": [{"role": "user", "content": "Hello!"}],
    }

    print(f"response: {response}")
    assert isinstance(response, ModelResponse)


def test_litellm_responses():
    """
    ensures that type of completion_tokens_details is correctly handled / returned
    """
    from litellm import ModelResponse
    from litellm.types.utils import CompletionTokensDetails

    response = ModelResponse(
        usage={
            "completion_tokens": 436,
            "prompt_tokens": 14,
            "total_tokens": 450,
            "completion_tokens_details": {"reasoning_tokens": 0},
        }
    )

    print("response: ", response)

    assert isinstance(response.usage.completion_tokens_details, CompletionTokensDetails)
