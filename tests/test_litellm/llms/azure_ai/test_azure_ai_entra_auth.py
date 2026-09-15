"""
Entra ID / OAuth auth for Azure AI Foundry routes.

Every azure_ai route must authenticate with an Entra ID token when no API key is configured,
instead of requiring an API key.
"""

from unittest.mock import patch

import pytest

import litellm
from litellm.llms.azure_ai.common_utils import get_azure_ai_auth_headers
from litellm.llms.azure_ai.ocr.transformation import AzureAIOCRConfig

ENTRA_PARAMS = {"azure_ad_token": "entra-token"}


@pytest.fixture(autouse=True)
def clear_azure_env(monkeypatch):
    for env_var in (
        "AZURE_AI_API_KEY",
        "AZURE_API_KEY",
        "AZURE_AD_TOKEN",
        "AZURE_TENANT_ID",
        "AZURE_CLIENT_ID",
        "AZURE_CLIENT_SECRET",
        "AZURE_SCOPE",
        "OPENAI_API_KEY",
        "AZURE_DOCUMENT_INTELLIGENCE_API_KEY",
    ):
        monkeypatch.delenv(env_var, raising=False)
    monkeypatch.setattr(litellm, "api_key", None)
    monkeypatch.setattr(litellm, "openai_key", None)


def test_api_key_wins_over_entra_credentials():
    headers = get_azure_ai_auth_headers(api_key="my-key", litellm_params=ENTRA_PARAMS, api_key_header="Api-Key")

    assert headers == {"Api-Key": "my-key"}


def test_entra_token_used_when_no_api_key():
    headers = get_azure_ai_auth_headers(api_key=None, litellm_params=ENTRA_PARAMS, api_key_header="Api-Key")

    assert headers == {"Authorization": "Bearer entra-token"}


def test_service_principal_token_is_requested_with_the_configured_scope():
    with patch("litellm.llms.azure.common_utils.get_azure_ad_token_from_entra_id") as mock_entra_id:  # test-quality-ok: stubs the Entra token fetch to assert the SP credential+scope plumbing and the returned Bearer header; live SP path proven by the PR's Azure Foundry e2e QA
        mock_entra_id.return_value = lambda: "sp-token"

        headers = get_azure_ai_auth_headers(
            api_key=None,
            litellm_params={
                "tenant_id": "tenant",
                "client_id": "client",
                "client_secret": "secret",
                "azure_scope": "https://ai.azure.com/.default",
            },
        )

    mock_entra_id.assert_called_once_with(
        tenant_id="tenant",
        client_id="client",
        client_secret="secret",
        scope="https://ai.azure.com/.default",
    )
    assert headers == {"Authorization": "Bearer sp-token"}


def test_error_mentions_both_credential_types_when_nothing_is_configured():
    with pytest.raises(ValueError, match="AZURE_AI_API_KEY") as exc_info:
        get_azure_ai_auth_headers(api_key=None, litellm_params={})

    message = str(exc_info.value)
    assert "AZURE_AI_API_KEY" in message
    assert "client_secret" in message


def test_ocr_authenticates_with_entra_token():
    headers = AzureAIOCRConfig().validate_environment(
        headers={},
        model="azure_ai/mistral-ocr",
        api_base="https://my-resource.services.ai.azure.com",
        litellm_params=ENTRA_PARAMS,
    )

    assert headers["Authorization"] == "Bearer entra-token"


def test_embedding_falls_back_to_entra_token_instead_of_openai_key(monkeypatch):  # test-quality-ok: asserts the embedding handler is authed with the Entra token, not the OpenAI key fallback; live path proven by the PR's Azure Foundry e2e QA
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-key")

    with patch.object(litellm.main.azure_ai_embedding, "embedding") as mock_embedding:  # test-quality-ok: no injection seam for the embedding handler through the public embedding() API; live path proven by the PR's Azure Foundry e2e QA
        mock_embedding.return_value = litellm.EmbeddingResponse()

        litellm.embedding(
            model="azure_ai/cohere-embed-v3-english",
            input=["hello"],
            api_base="https://my-resource.services.ai.azure.com",
            azure_ad_token="entra-token",
        )

    assert mock_embedding.call_args.kwargs["api_key"] == "entra-token"


def test_image_generation_authenticates_with_entra_token():
    with patch.object(litellm.images.main.azure_chat_completions, "image_generation") as mock_image_generation:  # test-quality-ok: asserts image_generation forwards the computed Entra bearer header; no injection seam through the public API; live path proven by the PR's Azure Foundry e2e QA
        mock_image_generation.return_value = litellm.ImageResponse()

        litellm.image_generation(
            model="azure_ai/FLUX-1.1-pro",
            prompt="a red circle",
            api_base="https://my-resource.services.ai.azure.com",
            azure_ad_token="entra-token",
        )

    headers = mock_image_generation.call_args.kwargs["headers"]
    assert headers["Authorization"] == "Bearer entra-token"
    assert "api-key" not in headers


@pytest.mark.parametrize("header_name", ["Authorization", "authorization", "api-key", "API-KEY"])
def test_image_generation_keeps_caller_supplied_auth_header(header_name):
    with patch.object(litellm.images.main.azure_chat_completions, "image_generation") as mock_image_generation:  # test-quality-ok: asserts a caller-supplied auth header is preserved over Entra; no injection seam through the public API; live path proven by the PR's Azure Foundry e2e QA
        mock_image_generation.return_value = litellm.ImageResponse()

        litellm.image_generation(
            model="azure_ai/FLUX-1.1-pro",
            prompt="a red circle",
            api_base="https://my-resource.services.ai.azure.com",
            headers={header_name: "caller-credential"},
        )

    headers = mock_image_generation.call_args.kwargs["headers"]
    assert headers[header_name] == "caller-credential"
    assert len(headers) == 2


def test_image_generation_still_uses_api_key_header():
    with patch.object(litellm.images.main.azure_chat_completions, "image_generation") as mock_image_generation:  # test-quality-ok: asserts the api-key header path still works alongside Entra; no injection seam through the public API; live path proven by the PR's Azure Foundry e2e QA
        mock_image_generation.return_value = litellm.ImageResponse()

        litellm.image_generation(
            model="azure_ai/FLUX-1.1-pro",
            prompt="a red circle",
            api_base="https://my-resource.services.ai.azure.com",
            api_key="my-key",
        )

    headers = mock_image_generation.call_args.kwargs["headers"]
    assert headers["api-key"] == "my-key"
    assert "Authorization" not in headers
