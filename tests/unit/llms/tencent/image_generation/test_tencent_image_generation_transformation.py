import json

import httpx
import pytest

import litellm
from litellm.llms.custom_httpx.http_handler import HTTPHandler

TOKENHUB_RESPONSE = {
    "created": 1792551825,
    "data": [{"output_format": "jpeg", "size": "1248x832", "url": "https://cos.example/lantern.jpg"}],
    "model": "doubao-seedream-5-0-pro-260628",
    "tokenhub_usage": {"total_tokens": 28125},
    "request_id": "c57b0374-b5fe-47db-b4b2-751315f920aa",
}


def _client(captured: dict, status_code: int = 200, body: dict | None = None) -> HTTPHandler:
    def respond(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["authorization"] = request.headers.get("authorization")
        captured["body"] = json.loads(request.content)
        return httpx.Response(status_code, json=TOKENHUB_RESPONSE if body is None else body)

    return HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(respond)))


def test_generation_posts_openai_and_tokenhub_params_to_the_wand_route():
    captured = {}

    response = litellm.image_generation(
        model="tencent/seedream-image-v5.0-pro",
        prompt="a red paper lantern",
        size="2K",
        response_format="url",
        api_key="sk-tokenhub",
        client=_client(captured),
        images=["https://ref.example/lantern.png"],
        watermark=False,
    )

    assert captured["url"] == "https://tokenhub-intl.tencentcloudmaas.com/v1/wand/si-image/generation"
    assert captured["authorization"] == "Bearer sk-tokenhub"
    assert captured["body"] == {
        "model": "seedream-image-v5.0-pro",
        "prompt": "a red paper lantern",
        "size": "2K",
        "response_format": "url",
        "images": ["https://ref.example/lantern.png"],
        "watermark": False,
    }
    assert response.data[0].url == "https://cos.example/lantern.jpg"
    assert response.created == 1792551825
    assert response.usage.total_tokens == 28125
    assert response.usage.output_tokens == 28125


def test_generation_returns_base64_images():
    captured = {}
    body = {**TOKENHUB_RESPONSE, "data": [{"output_format": "png", "size": "1024x1024", "b64_json": "aW1n"}]}

    response = litellm.image_generation(
        model="tencent/seedream-image-v5.0-flash",
        prompt="a red paper lantern",
        response_format="b64_json",
        api_key="sk-tokenhub",
        client=_client(captured, body=body),
    )

    assert response.data[0].b64_json == "aW1n"
    assert response.data[0].url is None


@pytest.mark.parametrize(
    ("api_base", "expected_url"),
    [
        ("https://tokenhub.example", "https://tokenhub.example/v1/wand/si-image/generation"),
        ("https://tokenhub.example/v1", "https://tokenhub.example/v1/wand/si-image/generation"),
        ("https://tokenhub.example/v1/", "https://tokenhub.example/v1/wand/si-image/generation"),
    ],
)
def test_generation_resolves_a_configured_api_base_to_the_wand_route(api_base, expected_url):
    captured = {}

    litellm.image_generation(
        model="tencent/seedream-image-v5.0-lite",
        prompt="a red paper lantern",
        api_key="sk-tokenhub",
        api_base=api_base,
        client=_client(captured),
    )

    assert captured["url"] == expected_url


@pytest.mark.parametrize(
    ("params", "expected_body_params"),
    [
        ({"n": 1}, {}),
        ({"n": 3, "drop_params": True}, {}),
        ({"background": "auto"}, {}),
        ({"background": "transparent", "output_format": "png"}, {"background": "transparent", "output_format": "png"}),
    ],
)
def test_generation_sends_only_values_tokenhub_accepts(params, expected_body_params):
    captured = {}

    litellm.image_generation(
        model="tencent/seedream-image-v5.0-pro",
        prompt="a red paper lantern",
        api_key="sk-tokenhub",
        client=_client(captured),
        **params,
    )

    assert captured["body"] == {
        "model": "seedream-image-v5.0-pro",
        "prompt": "a red paper lantern",
        **expected_body_params,
    }


def test_generation_rejects_more_than_one_image_without_drop_params():
    captured = {}

    with pytest.raises(litellm.UnsupportedParamsError) as exc_info:
        litellm.image_generation(
            model="tencent/seedream-image-v5.0-pro",
            prompt="a red paper lantern",
            n=2,
            api_key="sk-tokenhub",
            client=_client(captured),
        )

    assert exc_info.value.status_code == 400
    assert captured == {}


@pytest.mark.parametrize(
    ("error", "expected_message"),
    [
        (
            {
                "error": {
                    "message": "The model or service ID 'seedream-image-v5.0-pro' does not exist.",
                    "code": "400004",
                    "type": "gateway_error",
                }
            },
            "does not exist",
        ),
        ({"Type": 2, "Code": 1001, "Msg": "Invalid param", "Desc": ""}, "Invalid param"),
    ],
)
def test_generation_surfaces_the_tokenhub_error_message(error, expected_message):
    with pytest.raises(litellm.BadRequestError) as exc_info:
        litellm.image_generation(
            model="tencent/seedream-image-v5.0-pro",
            prompt="a red paper lantern",
            api_key="sk-tokenhub",
            client=_client({}, status_code=400, body=error),
        )

    assert expected_message in str(exc_info.value)


def test_generation_without_a_tokenhub_key_is_an_authentication_error(monkeypatch):
    monkeypatch.delenv("TENCENT_API_KEY", raising=False)
    captured = {}

    with pytest.raises(litellm.AuthenticationError) as exc_info:
        litellm.image_generation(
            model="tencent/seedream-image-v5.0-pro",
            prompt="a red paper lantern",
            client=_client(captured),
        )

    assert exc_info.value.status_code == 401
    assert "TENCENT_API_KEY" in str(exc_info.value)
    assert captured == {}
