import base64
import json
from io import BytesIO
from typing import Final

import httpx
import pytest

import litellm
from litellm.llms.alibaba_token_plan.common_utils import IMAGE_PATH
from litellm.llms.custom_httpx.http_handler import HTTPHandler
from litellm.types.utils import ImageResponse

GATEWAY: Final = "https://gateway.example/token-plan"
PNG_BYTES: Final = b"\x89PNG\r\n\x1a\nimage-content"
PNG_DATA_URL: Final = f"data:image/png;base64,{base64.b64encode(PNG_BYTES).decode('ascii')}"
REFERENCE_URL: Final = "https://images.example/reference.png"
OUTPUT_URL: Final = "https://images.example/edited.png"


def _edit(transport: httpx.MockTransport, **params: object) -> ImageResponse:
    with httpx.Client(transport=transport) as client:
        response: Final = litellm.image_edit(
            model="alibaba_token_plan/wan2.7-image",
            image=[BytesIO(PNG_BYTES), REFERENCE_URL],
            prompt="Change the background",
            api_base=f"{GATEWAY}/compatible-mode/v1",
            api_key="explicit-key",
            client=HTTPHandler(client=client),
            **params,
        )
    assert isinstance(response, ImageResponse)
    return response


def test_image_edit_sends_uploads_as_data_uris_with_mapped_and_native_params() -> None:
    def transport(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == f"{GATEWAY}/{IMAGE_PATH}"
        assert request.headers["Authorization"] == "Bearer explicit-key"
        assert json.loads(request.content) == {
            "model": "wan2.7-image",
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"image": PNG_DATA_URL},
                            {"image": REFERENCE_URL},
                            {"text": "Change the background"},
                        ],
                    }
                ]
            },
            "parameters": {"n": 2, "size": "1024*1024", "watermark": False},
        }
        return httpx.Response(200, json={"output": {"choices": [{"message": {"content": [{"image": OUTPUT_URL}]}}]}})

    response: Final = _edit(httpx.MockTransport(transport), n=2, size="1024x1024", watermark=False)
    assert response.data is not None
    assert [item.url for item in response.data] == [OUTPUT_URL]


def test_image_edit_rejects_base64_output_unless_dropped() -> None:
    def transport(request: httpx.Request) -> httpx.Response:
        assert "response_format" not in json.loads(request.content)["parameters"]
        return httpx.Response(200, json={"output": {"choices": [{"message": {"content": [{"image": OUTPUT_URL}]}}]}})

    with pytest.raises(litellm.UnsupportedParamsError, match="response_format='url'"):
        _edit(httpx.MockTransport(transport), response_format="b64_json")
    assert _edit(httpx.MockTransport(transport), response_format="b64_json", drop_params=True).data
