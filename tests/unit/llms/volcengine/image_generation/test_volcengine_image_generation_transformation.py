import json
from typing import Final

import httpx
import pytest

import litellm
from litellm.llms.custom_httpx.http_handler import HTTPHandler
from litellm.llms.volcengine.image_edit.transformation import VolcEngineImageEditConfig
from litellm.llms.volcengine.image_generation.transformation import VolcEngineImageGenerationConfig
from litellm.utils import ProviderConfigManager

API_BASE: Final = "https://ark.cn-beijing.volces.com"
IMAGES_URL: Final = f"{API_BASE}/api/v3/images/generations"
SEEDREAM_5_5: Final = "doubao-seedream-5-5-261028"

ARK_RESPONSE: Final[dict[str, object]] = {
    "created": 1784696685,
    "model": SEEDREAM_5_5,
    "data": [
        {
            "url": "https://ark.example/base.png",
            "size": "2048x2048",
            "output_format": "png",
            "background": "transparent",
            "z_index": 0,
        },
        {
            "url": "https://ark.example/parrot.png",
            "size": "2048x2048",
            "output_format": "png",
            "z_index": 1,
            "name": "parrot",
            "description": "a green parrot",
            "bounding_box": {"absolute": [347, 305, 642, 997], "normalized": [169, 305, 313, 997]},
        },
    ],
    "usage": {"generated_images": 2, "input_images": 1, "output_tokens": 32768, "total_tokens": 32768},
}


def _capturing_client(
    captured: dict[str, object],
    responses: list[httpx.Response] | None = None,
) -> HTTPHandler:
    def respond(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["authorization"] = request.headers.get("authorization")
        captured["body"] = json.loads(request.content)
        return responses[0] if responses else httpx.Response(200, json=ARK_RESPONSE)

    return HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(respond)))


def test_generation_sends_every_ark_param_and_maps_user_to_safety_identifier() -> None:
    captured: Final[dict[str, object]] = {}

    response: Final = litellm.image_generation(
        model=f"volcengine/{SEEDREAM_5_5}",
        prompt="split the parrot out of the base image",
        api_key="sk-ark",
        client=_capturing_client(captured),
        size="2K",
        response_format="url",
        image=["https://ark.example/input.png"],
        watermark=False,
        layer_decomposition=True,
        layer_decomposition_options={"max_images": 5},
        background="transparent",
        user="end-user-42",
        safety_identifier="explicit-id",
    )

    assert captured["url"] == IMAGES_URL
    assert captured["authorization"] == "Bearer sk-ark"
    assert captured["body"] == {
        "model": SEEDREAM_5_5,
        "prompt": "split the parrot out of the base image",
        "size": "2K",
        "response_format": "url",
        "image": ["https://ark.example/input.png"],
        "watermark": False,
        "layer_decomposition": True,
        "layer_decomposition_options": {"max_images": 5},
        "background": "transparent",
        "safety_identifier": "explicit-id",
    }
    assert response.data[0].url == "https://ark.example/base.png"
    assert response.data[1].provider_specific_fields == {
        "size": "2048x2048",
        "output_format": "png",
        "z_index": 1,
        "name": "parrot",
        "description": "a green parrot",
        "bounding_box": {"absolute": [347, 305, 642, 997], "normalized": [169, 305, 313, 997]},
    }
    assert (response.size, response.output_format, response.background) == ("2048x2048", "png", "transparent")
    assert response.usage is not None
    assert (response.usage.output_tokens, response.usage.total_tokens) == (32768, 32768)
    assert response.usage.model_dump()["input_images"] == 1
    assert response.usage.model_dump()["generated_images"] == 2


def test_generation_returns_base64_images() -> None:
    captured: Final[dict[str, object]] = {}
    body: Final[dict[str, object]] = {
        **ARK_RESPONSE,
        "data": [{"b64_json": "aW1n", "size": "1024x1024", "output_format": "jpeg"}],
    }

    response: Final = litellm.image_generation(
        model=f"volcengine/{SEEDREAM_5_5}",
        prompt="a cat",
        api_key="sk-ark",
        client=_capturing_client(captured, responses=[httpx.Response(200, json=body)]),
        response_format="b64_json",
    )

    assert captured["body"]["response_format"] == "b64_json"
    assert response.data[0].b64_json == "aW1n"
    assert response.data[0].url is None


def test_user_becomes_safety_identifier_when_no_explicit_id_is_given() -> None:
    captured: Final[dict[str, object]] = {}

    litellm.image_generation(
        model=f"volcengine/{SEEDREAM_5_5}",
        prompt="a cat",
        api_key="sk-ark",
        client=_capturing_client(captured),
        user="end-user-42",
    )

    assert captured["body"]["safety_identifier"] == "end-user-42"
    assert "user" not in captured["body"]


@pytest.mark.parametrize("n", [2, "3"])
def test_n_above_one_is_refused_before_any_ark_request(n: int | str) -> None:
    captured: Final[dict[str, object]] = {}

    with pytest.raises(litellm.UnsupportedParamsError):
        litellm.image_generation(
            model=f"volcengine/{SEEDREAM_5_5}",
            prompt="a cat",
            api_key="sk-ark",
            client=_capturing_client(captured),
            n=n,
        )

    assert captured == {}


@pytest.mark.parametrize(("n", "drop_params"), [(1, False), (3, True)])
def test_one_image_or_a_dropped_n_sends_one_ark_request_without_n(n: int, drop_params: bool) -> None:
    captured: Final[dict[str, object]] = {}

    response: Final = litellm.image_generation(
        model=f"volcengine/{SEEDREAM_5_5}",
        prompt="a cat",
        api_key="sk-ark",
        client=_capturing_client(
            captured,
            responses=[httpx.Response(200, json={**ARK_RESPONSE, "data": [{"url": "https://ark.example/one.png"}]})],
        ),
        n=n,
        drop_params=drop_params,
    )

    assert captured["body"] == {"model": SEEDREAM_5_5, "prompt": "a cat"}
    assert len(response.data) == 1


@pytest.mark.parametrize(
    ("api_base", "expected_url"),
    [
        (None, IMAGES_URL),
        ("https://ark.example", "https://ark.example/api/v3/images/generations"),
        ("https://ark.example/api/v3", "https://ark.example/api/v3/images/generations"),
        ("https://ark.example/api/v3/", "https://ark.example/api/v3/images/generations"),
        ("https://ark.example/api/v3/images/generations", "https://ark.example/api/v3/images/generations"),
    ],
)
def test_generation_resolves_an_api_base_to_the_images_route(api_base: str | None, expected_url: str) -> None:
    captured: Final[dict[str, object]] = {}

    litellm.image_generation(
        model=f"volcengine/{SEEDREAM_5_5}",
        prompt="a cat",
        api_key="sk-ark",
        api_base=api_base,
        client=_capturing_client(captured),
    )

    assert captured["url"] == expected_url


def test_generation_surfaces_the_ark_error_message() -> None:
    captured: Final[dict[str, object]] = {}

    with pytest.raises(litellm.BadRequestError) as exc_info:
        litellm.image_generation(
            model=f"volcengine/{SEEDREAM_5_5}",
            prompt="a cat",
            api_key="sk-ark",
            client=_capturing_client(
                captured,
                responses=[
                    httpx.Response(
                        400,
                        json={"error": {"code": "InvalidParameter", "message": "size 512x512 is below the minimum"}},
                    )
                ],
            ),
        )

    assert "size 512x512 is below the minimum" in str(exc_info.value)


def test_generation_without_an_ark_key_is_an_authentication_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ARK_API_KEY", raising=False)
    monkeypatch.delenv("VOLCENGINE_API_KEY", raising=False)
    captured: Final[dict[str, object]] = {}

    with pytest.raises(litellm.AuthenticationError) as exc_info:
        litellm.image_generation(
            model=f"volcengine/{SEEDREAM_5_5}",
            prompt="a cat",
            client=_capturing_client(captured),
        )

    assert "ARK_API_KEY" in str(exc_info.value)
    assert captured == {}


def test_generation_rejects_streaming_because_ark_streams_sse_outside_the_image_surface() -> None:
    captured: Final[dict[str, object]] = {}

    with pytest.raises(litellm.UnsupportedParamsError):
        litellm.image_generation(
            model=f"volcengine/{SEEDREAM_5_5}",
            prompt="a cat",
            api_key="sk-ark",
            client=_capturing_client(captured),
            stream=True,
        )

    assert captured == {}


def test_both_surfaces_are_registered_for_volcengine() -> None:
    assert isinstance(
        ProviderConfigManager.get_provider_image_generation_config(SEEDREAM_5_5, litellm.LlmProviders.VOLCENGINE),
        VolcEngineImageGenerationConfig,
    )
    assert isinstance(
        ProviderConfigManager.get_provider_image_edit_config(SEEDREAM_5_5, litellm.LlmProviders.VOLCENGINE),
        VolcEngineImageEditConfig,
    )
