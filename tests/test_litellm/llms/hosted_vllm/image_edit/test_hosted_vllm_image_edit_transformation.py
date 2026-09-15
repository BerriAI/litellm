import httpx
import pytest

import litellm
from litellm.llms.custom_httpx.http_handler import HTTPHandler
from litellm.llms.hosted_vllm.image_edit import get_hosted_vllm_image_edit_config
from litellm.llms.hosted_vllm.image_edit.transformation import HostedVLLMImageEditConfig
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager

PNG_BYTES = b"\x89PNG\r\n\x1a\nfakepng"
MODEL = "Qwen/Qwen-Image-Edit-2511"


@pytest.fixture(autouse=True)
def _clear_hosted_vllm_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HOSTED_VLLM_API_KEY", raising=False)
    monkeypatch.delenv("HOSTED_VLLM_API_BASE", raising=False)


def test_provider_config_registration():
    config = ProviderConfigManager.get_provider_image_edit_config(
        model=f"hosted_vllm/{MODEL}",
        provider=LlmProviders.HOSTED_VLLM,
    )

    assert isinstance(config, HostedVLLMImageEditConfig)
    assert isinstance(get_hosted_vllm_image_edit_config(MODEL), HostedVLLMImageEditConfig)


@pytest.mark.parametrize(
    "api_base",
    ["http://localhost:8091", "http://localhost:8091/", "http://localhost:8091/v1", "http://localhost:8091/v1/"],
)
def test_get_complete_url_appends_images_edits(api_base: str):
    config = HostedVLLMImageEditConfig()

    assert (
        config.get_complete_url(model=MODEL, api_base=api_base, litellm_params={})
        == "http://localhost:8091/v1/images/edits"
    )


def test_get_complete_url_falls_back_to_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HOSTED_VLLM_API_BASE", "http://vllm-omni:8000/v1")
    config = HostedVLLMImageEditConfig()

    assert (
        config.get_complete_url(model=MODEL, api_base=None, litellm_params={})
        == "http://vllm-omni:8000/v1/images/edits"
    )


def test_get_complete_url_requires_api_base():
    config = HostedVLLMImageEditConfig()

    with pytest.raises(ValueError, match="api_base not set"):
        config.get_complete_url(model=MODEL, api_base=None, litellm_params={})


def test_validate_environment_defaults_to_fake_api_key():
    headers = HostedVLLMImageEditConfig().validate_environment(headers={}, model=MODEL)

    assert headers == {"Authorization": "Bearer fake-api-key"}


def test_validate_environment_uses_provided_api_key_and_keeps_headers():
    headers = HostedVLLMImageEditConfig().validate_environment(
        headers={"X-Test": "1"},
        model=MODEL,
        api_key="my-custom-key",
    )

    assert headers == {"X-Test": "1", "Authorization": "Bearer my-custom-key"}


def test_validate_environment_falls_back_to_env_api_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HOSTED_VLLM_API_KEY", "env-key")

    headers = HostedVLLMImageEditConfig().validate_environment(headers={}, model=MODEL)

    assert headers["Authorization"] == "Bearer env-key"


def test_image_edit_posts_multipart_to_vllm_omni():
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json={"created": 1712697600, "data": [{"b64_json": "aW1n"}]})

    response = litellm.image_edit(
        model=f"hosted_vllm/{MODEL}",
        image=PNG_BYTES,
        prompt="add a hat",
        api_base="http://localhost:8091",
        api_key="test-key",
        client=HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(handler))),
        seed=42,
    )

    assert response.data
    assert len(captured) == 1
    request = captured[0]
    assert str(request.url) == "http://localhost:8091/v1/images/edits"
    assert request.headers["authorization"] == "Bearer test-key"
    assert request.headers["content-type"].startswith("multipart/form-data")
    assert b'name="image[]"' in request.content
    assert PNG_BYTES in request.content
    assert f'name="model"\r\n\r\n{MODEL}'.encode() in request.content
    assert b'name="prompt"\r\n\r\nadd a hat' in request.content
    assert b'name="seed"\r\n\r\n42' in request.content


@pytest.mark.parametrize("param", ["mask", "quality", "input_fidelity"])
def test_params_vllm_omni_ignores_are_not_advertised(param: str):
    supported = HostedVLLMImageEditConfig().get_supported_openai_params(MODEL)

    assert param not in supported
    assert {"image", "prompt", "n", "size", "response_format", "background", "user"} <= set(supported)


def test_image_edit_rejects_quality_unless_dropped():
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json={"created": 1712697600, "data": [{"b64_json": "aW1n"}]})

    client = HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(handler)))

    with pytest.raises(litellm.UnsupportedParamsError, match="quality"):
        litellm.image_edit(
            model=f"hosted_vllm/{MODEL}",
            image=PNG_BYTES,
            prompt="add a hat",
            api_base="http://localhost:8091",
            client=client,
            quality="low",
        )
    assert captured == []

    litellm.image_edit(
        model=f"hosted_vllm/{MODEL}",
        image=PNG_BYTES,
        prompt="add a hat",
        api_base="http://localhost:8091",
        client=client,
        quality="low",
        drop_params=True,
    )

    assert len(captured) == 1
    assert b'name="quality"' not in captured[0].content
    assert b'name="prompt"\r\n\r\nadd a hat' in captured[0].content
