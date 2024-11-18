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


@pytest.mark.asyncio()
@pytest.mark.respx
async def test_vision_with_custom_model(respx_mock: MockRouter):
    """
    Tests that an OpenAI compatible endpoint when sent an image will receive the image in the request

    """
    import base64
    import requests

    litellm.set_verbose = True
    api_base = "https://my-custom.api.openai.com"

    # Fetch and encode a test image
    url = "https://dummyimage.com/100/100/fff&text=Test+image"
    response = requests.get(url)
    file_data = response.content
    encoded_file = base64.b64encode(file_data).decode("utf-8")
    base64_image = f"data:image/png;base64,{encoded_file}"

    mock_response = ModelResponse(
        id="cmpl-mock",
        choices=[Choices(message=Message(content="Mocked response", role="assistant"))],
        created=int(datetime.now().timestamp()),
        model="my-custom-model",
    )

    mock_request = respx_mock.post(f"{api_base}/chat/completions").mock(
        return_value=httpx.Response(200, json=mock_response.dict())
    )

    response = await litellm.acompletion(
        model="openai/my-custom-model",
        max_tokens=10,
        api_base=api_base,  # use the mock api
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": base64_image},
                    },
                ],
            }
        ],
    )

    assert mock_request.called
    request_body = json.loads(mock_request.calls[0].request.content)

    print("request_body: ", request_body)

    assert request_body == {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAGQAAABkBAMAAACCzIhnAAAAG1BMVEURAAD///+ln5/h39/Dv79qX18uHx+If39MPz9oMSdmAAAACXBIWXMAAA7EAAAOxAGVKw4bAAABB0lEQVRYhe2SzWrEIBCAh2A0jxEs4j6GLDS9hqWmV5Flt0cJS+lRwv742DXpEjY1kOZW6HwHFZnPmVEBEARBEARB/jd0KYA/bcUYbPrRLh6amXHJ/K+ypMoyUaGthILzw0l+xI0jsO7ZcmCcm4ILd+QuVYgpHOmDmz6jBeJImdcUCmeBqQpuqRIbVmQsLCrAalrGpfoEqEogqbLTWuXCPCo+Ki1XGqgQ+jVVuhB8bOaHkvmYuzm/b0KYLWwoK58oFqi6XfxQ4Uz7d6WeKpna6ytUs5e8betMcqAv5YPC5EZB2Lm9FIn0/VP6R58+/GEY1X1egVoZ/3bt/EqF6malgSAIgiDIH+QL41409QMY0LMAAAAASUVORK5CYII="
                        },
                    },
                ],
            }
        ],
        "model": "my-custom-model",
        "max_tokens": 10,
    }

    print(f"response: {response}")
    assert isinstance(response, ModelResponse)
