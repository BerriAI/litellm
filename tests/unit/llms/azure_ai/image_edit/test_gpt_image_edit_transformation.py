from typing import Final

import httpx
import pytest

import litellm
from litellm.llms.azure_ai.image_edit import get_azure_ai_image_edit_config
from litellm.llms.azure_ai.image_edit.gpt_transformation import (
    AzureFoundryGPTImageEditConfig,
)
from litellm.llms.custom_httpx.http_handler import HTTPHandler

FOUNDRY_API_BASE: Final = "https://example.services.ai.azure.com"
DEPLOYMENT_EDIT_URL: Final = (
    f"{FOUNDRY_API_BASE}/openai/deployments/gpt-image-2.5-flare/images/edits?api-version=2025-04-01-preview"
)


@pytest.mark.parametrize(
    "model",
    [
        "gpt-image-1",
        "gpt-image-2",
        "gpt-image-2.5-flare",
        "gpt-image-2.5-flare-2026-09-08",
        "gpt-image-2.5-sunburst",
        "gpt-image-2.5-sunburst-2026-09-08",
    ],
)
def test_gpt_image_models_get_gpt_image_edit_config(model: str):
    assert isinstance(get_azure_ai_image_edit_config(model), AzureFoundryGPTImageEditConfig)


def test_gpt_image_edit_url_uses_foundry_deployment_route():
    config: Final = AzureFoundryGPTImageEditConfig()

    url: Final = config.get_complete_url(
        model="gpt-image-2.5-flare",
        api_base=FOUNDRY_API_BASE,
        litellm_params={"api_version": "2025-04-01-preview"},
    )

    assert url == DEPLOYMENT_EDIT_URL


def test_gpt_image_edit_url_uses_azure_ai_api_base_and_version_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("AZURE_AI_API_BASE", FOUNDRY_API_BASE)
    monkeypatch.setenv("AZURE_AI_API_VERSION", "2025-04-01-preview")
    monkeypatch.delenv("AZURE_API_VERSION", raising=False)
    monkeypatch.setattr(litellm, "api_base", None)
    monkeypatch.setattr(litellm, "api_version", None)

    url: Final = AzureFoundryGPTImageEditConfig().get_complete_url(
        model="gpt-image-2.5-flare",
        api_base=None,
        litellm_params={},
    )

    assert url == DEPLOYMENT_EDIT_URL


def test_gpt_image_edit_url_uses_v1_surface_for_preview_version():
    url: Final = AzureFoundryGPTImageEditConfig().get_complete_url(
        model="gpt-image-2.5-flare",
        api_base=FOUNDRY_API_BASE,
        litellm_params={"api_version": "preview"},
    )

    assert url == f"{FOUNDRY_API_BASE}/openai/v1/images/edits?api-version=preview"


def test_gpt_image_edit_sends_api_key_header():
    headers: Final = AzureFoundryGPTImageEditConfig().validate_environment(
        {},
        "gpt-image-2.5-flare",
        api_key="test-key",
    )

    assert headers == {"api-key": "test-key"}


def test_gpt_image_edit_prefers_litellm_params_api_key():
    headers: Final = AzureFoundryGPTImageEditConfig().validate_environment(
        {},
        "gpt-image-2.5-flare",
        api_key="positional-key",
        litellm_params={"api_key": "params-key"},
    )

    assert headers["api-key"] == "params-key"


def test_gpt_image_edit_reads_azure_ai_api_key_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("AZURE_AI_API_KEY", "env-key")
    monkeypatch.setattr(litellm, "api_key", None)

    headers: Final = AzureFoundryGPTImageEditConfig().validate_environment(
        {},
        "gpt-image-2.5-flare",
    )

    assert headers["api-key"] == "env-key"


def test_gpt_image_edit_falls_back_to_entra_token(monkeypatch: pytest.MonkeyPatch, no_ambient_azure_credentials: None):
    monkeypatch.delenv("AZURE_AI_API_KEY", raising=False)
    monkeypatch.setattr(litellm, "api_key", None)

    headers: Final = AzureFoundryGPTImageEditConfig().validate_environment(
        {},
        "gpt-image-2.5-flare",
        litellm_params={"azure_ad_token": "entra-token"},
    )

    assert headers == {"Authorization": "Bearer entra-token"}


def test_gpt_image_edit_strips_model_from_deployment_multipart():
    data: Final = AzureFoundryGPTImageEditConfig().finalize_image_edit_request_data(
        {"model": "gpt-image-2.5-flare", "prompt": "Add a hat"},
        DEPLOYMENT_EDIT_URL,
    )

    assert data == {"prompt": "Add a hat"}


def test_gpt_image_edit_keeps_model_on_v1_multipart():
    data: Final = AzureFoundryGPTImageEditConfig().finalize_image_edit_request_data(
        {"model": "gpt-image-2.5-flare", "prompt": "Add a hat"},
        f"{FOUNDRY_API_BASE}/openai/v1/images/edits?api-version=preview",
    )

    assert data == {"model": "gpt-image-2.5-flare", "prompt": "Add a hat"}


def test_gpt_image_edit_end_to_end_posts_multipart_to_deployment_route(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("AZURE_AI_API_VERSION", "2025-04-01-preview")
    monkeypatch.delenv("AZURE_API_VERSION", raising=False)
    captured: Final[list[httpx.Request]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json={"created": 1, "data": [{"b64_json": "aW1n"}]})

    client: Final = HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(respond)))
    response: Final = litellm.image_edit(
        model="azure_ai/gpt-image-2.5-flare",
        image=b"image",
        prompt="Add a hat",
        api_key="test-key",
        api_base=FOUNDRY_API_BASE,
        client=client,
    )

    assert response.data[0]["b64_json"] == "aW1n"
    assert len(captured) == 1
    request: Final = captured[0]
    assert str(request.url) == DEPLOYMENT_EDIT_URL
    assert request.headers["api-key"] == "test-key"
    assert b'name="prompt"' in request.content
    assert b'name="model"' not in request.content
