import base64
import io
import json
from typing import Final

import httpx
import pytest

import litellm
from litellm.llms.custom_httpx.http_handler import HTTPHandler

PNG_BYTES: Final = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
TOKENHUB_RESPONSE: Final[dict[str, object]] = {
    "created": 1792551838,
    "data": [{"output_format": "jpeg", "size": "2048x2048", "url": "https://cos.example/edited.jpg"}],
    "model": "doubao-seedream-5-0-lite",
    "tokenhub_usage": {"total_tokens": 21875},
    "request_id": "4f795d48-b227-4c35-af2c-1a0ba288b6b6",
}


def _client(captured: dict[str, object]) -> HTTPHandler:
    def respond(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["content_type"] = request.headers.get("content-type")
        captured["authorization"] = request.headers.get("authorization")
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=TOKENHUB_RESPONSE)

    return HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(respond)))


def test_edit_sends_uploaded_images_as_data_urls_in_a_json_body() -> None:
    captured: Final[dict[str, object]] = {}
    png_data_url: Final = f"data:image/png;base64,{base64.b64encode(PNG_BYTES).decode()}"

    response: Final = litellm.image_edit(
        model="tencent/seedream-image-v5.0-lite",
        image=[PNG_BYTES, io.BytesIO(PNG_BYTES), "https://ref.example/style.png"],
        prompt="turn the square into a paper lantern",
        size="2K",
        api_key="sk-tokenhub",
        client=_client(captured),
    )

    assert captured["url"] == "https://tokenhub-intl.tencentcloudmaas.com/v1/wand/si-image/generation"
    assert captured["content_type"] == "application/json"
    assert captured["authorization"] == "Bearer sk-tokenhub"
    assert captured["body"] == {
        "model": "seedream-image-v5.0-lite",
        "prompt": "turn the square into a paper lantern",
        "images": [png_data_url, png_data_url, "https://ref.example/style.png"],
        "size": "2K",
    }
    assert response.data[0].url == "https://cos.example/edited.jpg"
    assert response.usage.total_tokens == 21875


def test_edit_rejects_a_mask_without_drop_params() -> None:
    captured: Final[dict[str, object]] = {}

    with pytest.raises(litellm.UnsupportedParamsError):
        litellm.image_edit(
            model="tencent/seedream-image-v5.0-pro",
            image=PNG_BYTES,
            mask=PNG_BYTES,
            prompt="turn the square into a paper lantern",
            api_key="sk-tokenhub",
            client=_client(captured),
        )

    assert captured == {}


def test_edit_drops_a_mask_under_drop_params() -> None:
    captured: Final[dict[str, object]] = {}

    litellm.image_edit(
        model="tencent/seedream-image-v5.0-pro",
        image=PNG_BYTES,
        mask=PNG_BYTES,
        prompt="turn the square into a paper lantern",
        api_key="sk-tokenhub",
        client=_client(captured),
        drop_params=True,
    )

    assert captured["body"] == {
        "model": "seedream-image-v5.0-pro",
        "prompt": "turn the square into a paper lantern",
        "images": [f"data:image/png;base64,{base64.b64encode(PNG_BYTES).decode()}"],
    }


def test_edit_is_not_offered_for_tencent_models_outside_seedream() -> None:
    captured: Final[dict[str, object]] = {}

    with pytest.raises(Exception, match="image edit is not supported for tencent"):
        litellm.image_edit(
            model="tencent/hy-image-v3.5-preview",
            image=PNG_BYTES,
            prompt="turn the square into a paper lantern",
            api_key="sk-tokenhub",
            client=_client(captured),
        )

    assert captured == {}


def test_edit_forwards_tokenhub_only_params_and_extra_body() -> None:
    captured: Final[dict[str, object]] = {}

    litellm.image_edit(
        model="tencent/seedream-image-v5.0-flash",
        image=PNG_BYTES,
        prompt="turn the square into a paper lantern",
        api_key="sk-tokenhub",
        client=_client(captured),
        user="end-user-42",
        watermark=False,
        seed=7,
        extra_body={"sequential_image_generation": "disabled"},
    )

    assert captured["body"] == {
        "model": "seedream-image-v5.0-flash",
        "prompt": "turn the square into a paper lantern",
        "images": [f"data:image/png;base64,{base64.b64encode(PNG_BYTES).decode()}"],
        "user": "end-user-42",
        "watermark": False,
        "seed": 7,
        "sequential_image_generation": "disabled",
    }
