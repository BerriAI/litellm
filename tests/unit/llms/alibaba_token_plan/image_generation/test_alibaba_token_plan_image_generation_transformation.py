import json
from collections.abc import Mapping
from typing import Final

import httpx
import pytest

import litellm
from litellm.llms.alibaba_token_plan.common_utils import IMAGE_PATH
from litellm.llms.custom_httpx.http_handler import HTTPHandler
from litellm.types.utils import ImageResponse

GATEWAY: Final = "https://gateway.example/token-plan"


def _generate(transport: httpx.MockTransport, **params: object) -> ImageResponse:
    with httpx.Client(transport=transport) as client:
        response: Final = litellm.image_generation(
            model="alibaba_token_plan/qwen-image-3.0-pro",
            prompt="Draw a lighthouse",
            api_base=f"{GATEWAY}/compatible-mode/v1",
            api_key="explicit-key",
            max_retries=0,
            client=HTTPHandler(client=client),
            **params,
        )
    assert isinstance(response, ImageResponse)
    return response


def test_image_request_reaches_the_native_endpoint_with_mapped_and_native_params() -> None:
    def transport(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == f"{GATEWAY}/{IMAGE_PATH}"
        assert request.headers["Authorization"] == "Bearer explicit-key"
        assert json.loads(request.content) == {
            "model": "qwen-image-3.0-pro",
            "input": {"messages": [{"role": "user", "content": [{"text": "Draw a lighthouse"}]}]},
            "parameters": {"size": "1536*1024", "n": 2, "negative_prompt": "blurry", "watermark": False},
        }
        return httpx.Response(
            200,
            json={
                "output": {
                    "choices": [
                        {"message": {"content": [{"text": "Generated"}, {"image": "https://images.example/1.png"}]}},
                        {"message": {"content": [{"image": "https://images.example/2.png"}]}},
                    ]
                }
            },
        )

    response: Final = _generate(
        httpx.MockTransport(transport),
        n=2,
        size="1536x1024",
        response_format="url",
        extra_body={"negative_prompt": "blurry", "watermark": False},
    )
    assert response.data is not None
    assert [item.url for item in response.data] == ["https://images.example/1.png", "https://images.example/2.png"]


def test_base64_response_format_is_rejected_unless_dropped() -> None:
    def transport(request: httpx.Request) -> httpx.Response:
        assert "response_format" not in json.loads(request.content)["parameters"]
        return httpx.Response(200, json={"output": {"choices": [{"message": {"content": [{"image": "u"}]}}]}})

    with pytest.raises(litellm.UnsupportedParamsError, match="response_format='url'"):
        _generate(httpx.MockTransport(transport), response_format="b64_json")
    assert _generate(httpx.MockTransport(transport), response_format="b64_json", drop_params=True).data


@pytest.mark.parametrize(
    ("status_code", "body"),
    [
        (401, {"code": "InvalidApiKey", "message": "Invalid API-key provided."}),
        (200, {"code": "InvalidParameter", "message": "Size not supported"}),
    ],
)
def test_provider_errors_are_raised(status_code: int, body: Mapping[str, str]) -> None:
    with pytest.raises(litellm.APIError if status_code == 200 else litellm.AuthenticationError, match=body["message"]):
        _generate(httpx.MockTransport(lambda _request: httpx.Response(status_code, json=body)))
