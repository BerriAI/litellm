import base64
import json
from typing import Final

import httpx
import pytest

import litellm
from litellm.llms.chatgpt.images import ChatGPTImageEditConfig, ChatGPTImageGenerationConfig
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.types.llms.openai import ImageGenerationRequestQuality
from litellm.types.router import GenericLiteLLMParams


@pytest.mark.parametrize(
    "model,quality",
    [
        ("gpt-image-2", ImageGenerationRequestQuality.AUTO),
        ("gpt-image-2.5-flare", ImageGenerationRequestQuality.XHIGH),
        ("gpt-image-2.5-flare", ImageGenerationRequestQuality.MAX),
        ("gpt-image-2.5-sunburst", ImageGenerationRequestQuality.XHIGH),
        ("gpt-image-2.5-sunburst", ImageGenerationRequestQuality.MAX),
    ],
)
@pytest.mark.parametrize("editing", [False, True])
def test_image_25_transmits_model_quality_and_transparency(model, quality, editing, chatgpt_tokens):
    expected: Final = {
        "model": model,
        "prompt": "a red circle with transparent surroundings",
        "quality": quality.value,
        "background": "transparent",
        "size": "2048x2048",
        **({"images": [{"image_url": "data:image/png;base64,aGVsbG8="}]} if editing else {}),
    }

    def respond(request):
        assert str(request.url) == "https://chatgpt.com/backend-api/codex/images/" + (
            "edits" if editing else "generations"
        )
        assert request.headers["content-type"] == "application/json"
        assert json.loads(request.content) == expected
        return httpx.Response(
            200,
            json={"created": 1, "data": [{"b64_json": "aGVsbG8="}], "quality": quality.value},
        )

    client: Final = HTTPHandler()
    client.client = httpx.Client(transport=httpx.MockTransport(respond))
    operation: Final = litellm.image_edit if editing else litellm.image_generation
    try:
        response: Final = operation(
            **{**expected, "model": "chatgpt/" + model, "quality": quality},
            client=client,
            chatgpt_token_dir=chatgpt_tokens,
        )
        assert response.data[0].b64_json == "aGVsbG8="
        assert response.quality == quality.value
    finally:
        client.client.close()


@pytest.mark.parametrize("model", ["gpt-image-2", "gpt-image-2.5-flare", "gpt-image-2.5-sunburst"])
def test_json_edit_preserves_provider_params_and_extra_body_precedence(model, chatgpt_tokens):
    references: Final = [{"image_url": "data:image/png;base64,aGVsbG8="}]

    def respond(request):
        assert request.headers["content-type"] == "application/json"
        assert json.loads(request.content) == {
            "model": model,
            "prompt": "red circle",
            "images": references,
            "seed": 7,
            "provider_options": {"steps": 30, "enabled": True},
            "output_compression": 90,
        }
        return httpx.Response(200, json={"created": 1, "data": [{"b64_json": "aGVsbG8="}]})

    with httpx.Client(transport=httpx.MockTransport(respond)) as http_client:
        response: Final = litellm.image_edit(
            model="chatgpt/" + model,
            prompt="red circle",
            images=references,
            client=HTTPHandler(client=http_client),
            chatgpt_token_dir=chatgpt_tokens,
            seed=42,
            output_compression=90,
            extra_body={"seed": 7, "provider_options": {"steps": 30, "enabled": True}},
        )
        assert response.data[0].b64_json == "aGVsbG8="


@pytest.mark.parametrize("api_base", [None, "https://image-gateway.test"])
def test_generation_routes_with_chatgpt_oauth(chatgpt_tokens, api_base):
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(200, json={"created": 1, "data": [{"b64_json": "aGVsbG8="}]})

    client = HTTPHandler()
    client.client = httpx.Client(transport=httpx.MockTransport(respond))
    result = litellm.image_generation(
        model="chatgpt/gpt-image-2",
        prompt="blue circle",
        api_base=api_base,
        client=client,
        quality="auto",
        size="auto",
        background="auto",
        extra_headers={"x-gateway-route": "images", "aUtHoRiZaTiOn": "Bearer wrong", "CHATGPT-ACCOUNT-ID": "wrong"},
    )
    assert requests[0].headers["x-gateway-route"] == "images"
    assert result.data[0].b64_json == "aGVsbG8="
    assert str(requests[0].url) == (api_base or "https://chatgpt.com/backend-api/codex") + "/images/generations"
    assert requests[0].headers["authorization"] == "Bearer test-token-" + "default"
    assert requests[0].headers["chatgpt-account-id"] == "test-account-" + "default"
    assert b'"model":"gpt-image-2"' in requests[0].content


def test_codex_json_edit_survives_sdk_dispatch(chatgpt_tokens):
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(200, json={"created": 1, "data": [{"b64_json": "aGVsbG8="}]})

    client = HTTPHandler()
    client.client = httpx.Client(transport=httpx.MockTransport(respond))
    references = [{"image_url": "data:image/png;base64,aGVsbG8="}]
    result = litellm.image_edit(
        model="chatgpt/gpt-image-2",
        prompt="red circle",
        extra_headers={"x-gateway-route": "images", "authorization": "Bearer wrong", "CHATGPT-ACCOUNT-ID": "wrong"},
        images=references,
        client=client,
        quality="auto",
        size="auto",
    )
    assert result.data[0].b64_json == "aGVsbG8="
    assert str(requests[0].url) == "https://chatgpt.com/backend-api/codex/images/edits"
    import json

    assert json.loads(requests[0].content)["images"] == references

    assert requests[0].headers["authorization"] == "Bearer test-token-default"
    assert requests[0].headers["chatgpt-account-id"] == "test-account-default"
    assert requests[0].headers["x-gateway-route"] == "images"


@pytest.mark.parametrize(
    "references", [[], [{"image_url": "file:///etc/passwd"}], [{}], [{"image_url": "https://example.com/a.png"}] * 6]
)
def test_edit_rejects_invalid_references(references):
    with pytest.raises(ValueError, match=r"images must contain|validation error"):
        ChatGPTImageEditConfig().transform_image_edit_request(
            "gpt-image-2", "edit", None, {}, GenericLiteLLMParams(images=references), {}
        )


def test_edit_converts_multipart_image_bytes():
    data, files = ChatGPTImageEditConfig().transform_image_edit_request(
        "gpt-image-2", "edit", b"example", {}, GenericLiteLLMParams(), {}
    )
    assert not files
    assert base64.b64decode(data["images"][0]["image_url"].split(",", 1)[1]) == b"example"


def test_image_auth_does_not_accept_inbound_override(chatgpt_tokens):
    headers = ChatGPTImageGenerationConfig().validate_environment(
        {"authorization": "Bearer wrong", "CHATGPT-ACCOUNT-ID": "wrong"}, "gpt-image-2", [], {}, {"chatgpt_token_dir": chatgpt_tokens}
    )
    assert httpx.Headers(headers)["authorization"] == "Bearer test-token-default"
    assert httpx.Headers(headers)["chatgpt-account-id"] == "test-account-default"


@pytest.mark.asyncio
async def test_async_codex_edit_without_multipart_image(chatgpt_tokens):
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(200, json={"created": 1, "data": [{"b64_json": "aGVsbG8="}]})

    client = AsyncHTTPHandler()
    client.client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    response = await litellm.aimage_edit(
        model="chatgpt/gpt-image-2",
        prompt="red circle",
        extra_headers={"x-gateway-route": "images", "authorization": "Bearer wrong", "CHATGPT-ACCOUNT-ID": "wrong"},
        client=client,
        images=[{"image_url": "data:image/png;base64,aGVsbG8="}],
        chatgpt_auth_profile="account3",
    )
    assert response.data[0].b64_json == "aGVsbG8="
    assert str(requests[0].url).endswith("/codex/images/edits")
    assert requests[0].headers["content-type"] == "application/json"
    await client.client.aclose()

    assert requests[0].headers["authorization"] == "Bearer test-token-default"
    assert requests[0].headers["chatgpt-account-id"] == "test-account-default"
    assert requests[0].headers["x-gateway-route"] == "images"


@pytest.mark.parametrize("as_tuple", [False, True])
def test_edit_accepts_filesystem_path(tmp_path, as_tuple):
    image = tmp_path / "reference.png"
    image.write_bytes(b"reference image bytes")
    data, files = ChatGPTImageEditConfig().transform_image_edit_request(
        "gpt-image-2", "edit", ("reference.png", image, "image/png") if as_tuple else image,
        {}, GenericLiteLLMParams(), {}
    )
    assert not files
    assert data["images"] == ({"image_url": "data:image/png;base64," + base64.b64encode(image.read_bytes()).decode()},)


@pytest.mark.parametrize("env_name", ["CHATGPT_API_BASE", "OPENAI_CHATGPT_API_BASE"])
@pytest.mark.parametrize("api_base", [None, "https://deployment.example/codex"])
def test_image_routes_use_configured_gateway(monkeypatch, env_name, api_base, tmp_path):
    token_path = tmp_path / "unavailable-token-directory"
    token_path.write_text("not a directory")
    monkeypatch.setenv("CHATGPT_TOKEN_DIR", str(token_path))
    monkeypatch.delenv("CHATGPT_API_BASE", raising=False)
    monkeypatch.delenv("OPENAI_CHATGPT_API_BASE", raising=False)
    monkeypatch.setenv(env_name, "https://gateway.example/codex/")
    expected = api_base or "https://gateway.example/codex"
    assert ChatGPTImageGenerationConfig().get_complete_url(api_base, None, "gpt-image-2", {}, {}) == (
        expected + "/images/generations"
    )
    assert ChatGPTImageEditConfig().get_complete_url("gpt-image-2", api_base, {}) == expected + "/images/edits"
