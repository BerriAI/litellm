import base64
import json
from collections.abc import Mapping
from functools import partial
from io import BytesIO
from typing import Final

import httpx
import pytest

import litellm
from litellm.llms.alibaba_token_plan.common_utils import IMAGE_ENDPOINT
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.types.utils import FileTypes, ImageObject, ImageResponse

PNG_BYTES: Final = b"\x89PNG\r\n\x1a\nimage-content"
PNG_DATA_URL: Final = f"data:image/png;base64,{base64.b64encode(PNG_BYTES).decode('ascii')}"
REFERENCE_URL: Final = "https://images.example/reference.png"
OUTPUT_URL: Final = "https://images.example/edited.png"


class _InjectedAsyncHTTPHandler(AsyncHTTPHandler):
    def __init__(self, client: httpx.AsyncClient) -> None:
        self.client = client


def _image_edit_response(request: httpx.Request, model: str, native_params: Mapping[str, object]) -> httpx.Response:
    assert request.method == "POST"
    assert str(request.url) == f"https://gateway.example/token-plan/{IMAGE_ENDPOINT}?tenant=test"
    assert request.headers["Authorization"] == "Bearer explicit-key"
    assert request.headers["Content-Type"] == "application/json"
    assert json.loads(request.content) == {
        "model": model,
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": [{"image": PNG_DATA_URL}, {"image": REFERENCE_URL}, {"text": "Change the background"}],
                }
            ]
        },
        "parameters": {"n": 2, "size": "1024*1024", **native_params},
    }
    return httpx.Response(
        200,
        json={"output": {"choices": [{"message": {"content": [{"image": OUTPUT_URL}, {"text": "Edited"}]}}]}},
    )


@pytest.mark.parametrize("model", ["qwen-image-3.0-pro", "wan2.7-image", "wan2.7-image-pro"])
@pytest.mark.parametrize("path", [IMAGE_ENDPOINT, "compatible-mode/v1", "apps/anthropic"])
def test_sync_image_edit_translates_uploads_and_native_parameters(model: str, path: str) -> None:
    native_params: Final = (
        {"bbox_list": [[[0, 0, 20, 20]], []], "enable_sequential": True, "watermark": False}
        if model.startswith("wan")
        else {"negative_prompt": "blurry", "prompt_extend_mode": "direct", "enable_thinking": False}
    )
    with httpx.Client(
        transport=httpx.MockTransport(partial(_image_edit_response, model=model, native_params=native_params))
    ) as client:
        response: Final = litellm.image_edit(
            model=f"alibaba_token_plan/{model}",
            image=[("source.png", PNG_BYTES, "image/png"), REFERENCE_URL],
            prompt="Change the background",
            size="1024x1024",
            n=2,
            response_format="url",
            api_key="explicit-key",
            api_base=f"https://gateway.example/token-plan/{path}?tenant=test",
            client=HTTPHandler(client=client),
            max_retries=0,
            **native_params,
        )
    assert isinstance(response, ImageResponse)
    assert response.data == [ImageObject(url=OUTPUT_URL)]


@pytest.mark.asyncio
async def test_async_image_edit_preserves_stream_cursor_and_all_image_references() -> None:
    with BytesIO(PNG_BYTES) as source:
        source.seek(3)
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(partial(_image_edit_response, model="qwen-image-3.0-pro", native_params={}))
        ) as client:
            response: Final = await litellm.aimage_edit(
                model="alibaba_token_plan/qwen-image-3.0-pro",
                image=[source, REFERENCE_URL],
                prompt="Change the background",
                n=2,
                size="1024x1024",
                api_key="explicit-key",
                api_base=f"https://gateway.example/token-plan/{IMAGE_ENDPOINT}?tenant=test",
                client=_InjectedAsyncHTTPHandler(client),
                max_retries=0,
            )
        assert source.tell() == 3
    assert response.data == [ImageObject(url=OUTPUT_URL)]


@pytest.mark.parametrize("image", [PNG_BYTES, ("image.png", PNG_BYTES), PNG_DATA_URL])
def test_single_image_input_variants(image: FileTypes | str) -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content)["input"]["messages"][0]["content"] == [
            {"image": PNG_DATA_URL},
            {"text": "Edit"},
        ]
        return httpx.Response(200, json={"output": {"choices": [{"message": {"content": [{"image": OUTPUT_URL}]}}]}})

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        response: Final = litellm.image_edit(
            model="alibaba_token_plan/qwen-image-3.0-pro",
            image=image,
            prompt="Edit",
            api_key="test-key",
            client=HTTPHandler(client=client),
        )
    assert isinstance(response, ImageResponse)
    assert response.data == [ImageObject(url=OUTPUT_URL)]


@pytest.mark.parametrize("params", [{"mask": "mask.png"}, {"response_format": "b64_json"}, {"quality": "high"}])
def test_unsupported_image_edit_parameters_fail_before_transport(params: Mapping[str, object]) -> None:
    with pytest.raises(litellm.UnsupportedParamsError):
        litellm.image_edit(
            model="alibaba_token_plan/qwen-image-3.0-pro", image=PNG_BYTES, prompt="Edit", api_key="key", **params
        )


@pytest.mark.parametrize("model,count", [("qwen-image-3.0-pro", 0), ("qwen-image-3.0-pro", 4), ("wan2.7-image", 10)])
def test_invalid_reference_count_fails_before_transport(model: str, count: int) -> None:
    with pytest.raises(litellm.BadRequestError, match="input images"):
        litellm.image_edit(model=f"alibaba_token_plan/{model}", image=[PNG_BYTES] * count, prompt="Edit", api_key="key")


@pytest.mark.parametrize("status", [200, 429])
def test_image_edit_errors_are_not_returned_as_images(status: int) -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json={"code": "InvalidParameter", "message": "Invalid input image"})

    expected_error: Final = litellm.BadGatewayError if status == 200 else litellm.RateLimitError
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        with pytest.raises(expected_error, match="Invalid input image") as error:
            litellm.image_edit(
                model="alibaba_token_plan/qwen-image-3.0-pro",
                image=PNG_BYTES,
                prompt="Edit",
                api_key="key",
                client=HTTPHandler(client=client),
                max_retries=0,
            )
    assert error.value.status_code == (502 if status == 200 else status)
