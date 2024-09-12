import json
from datetime import datetime
from unittest.mock import AsyncMock

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
    assert response.choices[0].message.content == "Mocked response"


# ... existing code ...
