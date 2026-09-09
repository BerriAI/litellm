import base64

import httpx
import pytest

import litellm
from litellm.llms.chatgpt.images import ChatGPTImageEditConfig, ChatGPTImageGenerationConfig
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.types.router import GenericLiteLLMParams


def test_generation_routes_with_chatgpt_oauth(chatgpt_tokens):
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(200, json={"created": 1, "data": [{"b64_json": "aGVsbG8="}]})

    client = HTTPHandler()
    client.client = httpx.Client(transport=httpx.MockTransport(respond))
    result = litellm.image_generation(
        model="chatgpt/gpt-image-2",
        prompt="blue circle",
        client=client,
        quality="auto",
        size="auto",
        background="auto",
        extra_headers={"x-gateway-route": "images"},
    )
    assert requests[0].headers["x-gateway-route"] == "images"
    assert result.data[0].b64_json == "aGVsbG8="
    assert str(requests[0].url) == "https://chatgpt.com/backend-api/codex/images/generations"
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
        images=references,
        client=client,
        quality="auto",
        size="auto",
    )
    assert result.data[0].b64_json == "aGVsbG8="
    assert str(requests[0].url) == "https://chatgpt.com/backend-api/codex/images/edits"
    import json

    assert json.loads(requests[0].content)["images"] == references


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
        {"Authorization": "Bearer wrong"}, "gpt-image-2", [], {}, {"chatgpt_token_dir": chatgpt_tokens}
    )
    assert headers["Authorization"] == "Bearer test-token-default"


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
        client=client,
        images=[{"image_url": "data:image/png;base64,aGVsbG8="}],
        chatgpt_auth_profile="account3",
    )
    assert response.data[0].b64_json == "aGVsbG8="
    assert str(requests[0].url).endswith("/codex/images/edits")
    assert requests[0].headers["content-type"] == "application/json"
    await client.client.aclose()


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
