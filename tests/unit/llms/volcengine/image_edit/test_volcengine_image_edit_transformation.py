import base64
import io
import json
from typing import Final

import httpx
import pytest

import litellm
from litellm.llms.custom_httpx.http_handler import HTTPHandler

API_BASE: Final = "https://ark.cn-beijing.volces.com"
IMAGES_URL: Final = f"{API_BASE}/api/v3/images/generations"
SEEDREAM_5_5: Final = "doubao-seedream-5-5-261028"

PNG_BYTES: Final = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)
PNG_DATA_URL: Final = f"data:image/png;base64,{base64.b64encode(PNG_BYTES).decode()}"

ARK_RESPONSE: Final[dict[str, object]] = {
    "created": 1784696685,
    "model": SEEDREAM_5_5,
    "data": [{"url": "https://ark.example/edited.png", "size": "2048x2048", "output_format": "png"}],
    "usage": {"generated_images": 1, "input_images": 1, "output_tokens": 16384, "total_tokens": 16384},
}


def _capturing_client(captured: dict[str, object]) -> HTTPHandler:
    def respond(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["authorization"] = request.headers.get("authorization")
        captured["content_type"] = request.headers.get("content-type")
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=ARK_RESPONSE)

    return HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(respond)))


def test_edit_uploads_a_file_as_a_data_uri_on_the_generation_route() -> None:
    captured: Final[dict[str, object]] = {}

    response: Final = litellm.image_edit(
        model=f"volcengine/{SEEDREAM_5_5}",
        image=PNG_BYTES,
        prompt="make the lantern red",
        api_key="sk-ark",
        client=_capturing_client(captured),
        size="2K",
        response_format="url",
        user="end-user-42",
    )

    assert captured["url"] == IMAGES_URL
    assert captured["authorization"] == "Bearer sk-ark"
    assert captured["content_type"] == "application/json"
    assert captured["body"] == {
        "model": SEEDREAM_5_5,
        "prompt": "make the lantern red",
        "size": "2K",
        "response_format": "url",
        "safety_identifier": "end-user-42",
        "image": [PNG_DATA_URL],
    }
    assert response.data[0].url == "https://ark.example/edited.png"
    assert response.size == "2048x2048"


def test_edit_sends_urls_as_is_and_keeps_every_reference_image() -> None:
    captured: Final[dict[str, object]] = {}

    litellm.image_edit(
        model=f"volcengine/{SEEDREAM_5_5}",
        image=["https://ark.example/style.png", io.BytesIO(PNG_BYTES)],
        prompt="merge these into one poster",
        api_key="sk-ark",
        client=_capturing_client(captured),
        watermark=False,
    )

    assert captured["body"] == {
        "model": SEEDREAM_5_5,
        "prompt": "merge these into one poster",
        "watermark": False,
        "image": ["https://ark.example/style.png", PNG_DATA_URL],
    }


def test_edit_coerces_multipart_string_values_into_ark_types() -> None:
    captured: Final[dict[str, object]] = {}

    litellm.image_edit(
        model=f"volcengine/{SEEDREAM_5_5}",
        image=PNG_BYTES,
        prompt="split the lantern into layers",
        api_key="sk-ark",
        client=_capturing_client(captured),
        watermark="false",
        layer_decomposition="true",
        layer_decomposition_options='{"max_images": 5}',
    )

    assert captured["body"] == {
        "model": SEEDREAM_5_5,
        "prompt": "split the lantern into layers",
        "watermark": False,
        "layer_decomposition": True,
        "layer_decomposition_options": {"max_images": 5},
        "image": [PNG_DATA_URL],
    }


@pytest.mark.parametrize("n", [2, "2"])
def test_edit_with_n_above_one_is_refused_before_any_ark_request(n: int | str) -> None:
    captured: Final[dict[str, object]] = {}

    with pytest.raises(litellm.UnsupportedParamsError):
        litellm.image_edit(
            model=f"volcengine/{SEEDREAM_5_5}",
            image=PNG_BYTES,
            prompt="make the lantern red",
            api_key="sk-ark",
            client=_capturing_client(captured),
            n=n,
        )

    assert captured == {}


def test_edit_with_a_dropped_n_sends_one_ark_request_without_n() -> None:
    captured: Final[dict[str, object]] = {}

    response: Final = litellm.image_edit(
        model=f"volcengine/{SEEDREAM_5_5}",
        image=PNG_BYTES,
        prompt="make the lantern red",
        api_key="sk-ark",
        client=_capturing_client(captured),
        n=2,
        drop_params=True,
    )

    assert captured["body"] == {"model": SEEDREAM_5_5, "prompt": "make the lantern red", "image": [PNG_DATA_URL]}
    assert len(response.data) == 1


def test_edit_with_stream_and_n_dropped_sends_one_ark_request_without_either() -> None:
    bodies: Final[list[dict[str, object]]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={**ARK_RESPONSE, "data": [{"url": "https://ark.example/one.png"}]})

    response: Final = litellm.image_edit(
        model=f"volcengine/{SEEDREAM_5_5}",
        image=PNG_BYTES,
        prompt="make the lantern red",
        api_key="sk-ark",
        client=HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(respond))),
        n=3,
        stream=True,
        drop_params=True,
    )

    assert bodies == [{"model": SEEDREAM_5_5, "prompt": "make the lantern red", "image": [PNG_DATA_URL]}]
    assert len(response.data) == 1


def test_edit_with_a_multipart_stream_false_is_refused_for_n_and_not_for_stream() -> None:
    captured: Final[dict[str, object]] = {}

    with pytest.raises(litellm.UnsupportedParamsError) as exc_info:
        litellm.image_edit(
            model=f"volcengine/{SEEDREAM_5_5}",
            prompt="make it blue",
            image=io.BytesIO(PNG_BYTES),
            api_key="sk-ark",
            client=_capturing_client(captured),
            n="2",
            stream="false",
        )

    assert "n > 1 is not supported" in str(exc_info.value)
    assert captured == {}


def test_edit_forwards_extra_body_fields_to_ark() -> None:
    captured: Final[dict[str, object]] = {}

    _ = litellm.image_edit(
        model=f"volcengine/{SEEDREAM_5_5}",
        image=PNG_BYTES,
        prompt="make the lantern red",
        api_key="sk-ark",
        client=_capturing_client(captured),
        extra_body={"watermark": False, "optimize_prompt_options": {"mode": "fast"}},
    )

    assert captured["body"] == {
        "model": SEEDREAM_5_5,
        "prompt": "make the lantern red",
        "watermark": False,
        "optimize_prompt_options": {"mode": "fast"},
        "image": [PNG_DATA_URL],
    }


def test_edit_surfaces_the_ark_error_message() -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400, json={"error": {"code": "InvalidParameter", "message": "image 2 exceeds 10 reference images"}}
        )

    with pytest.raises(litellm.BadRequestError) as exc_info:
        litellm.image_edit(
            model=f"volcengine/{SEEDREAM_5_5}",
            image=PNG_BYTES,
            prompt="merge these",
            api_key="sk-ark",
            client=HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(respond))),
        )

    assert "exceeds 10 reference images" in str(exc_info.value)
