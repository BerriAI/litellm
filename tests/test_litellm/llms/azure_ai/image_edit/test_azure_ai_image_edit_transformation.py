

import litellm
from litellm.llms.azure_ai.image_edit.flux2_transformation import (
    AzureFoundryFlux2ImageEditConfig,
)
from litellm.llms.azure_ai.image_edit.transformation import (
    AzureFoundryFluxImageEditConfig,
)


def test_azure_ai_validate_environment():
    """Test Azure AI environment validation"""
    config = AzureFoundryFluxImageEditConfig()

    headers = {}
    config.validate_environment(headers, "FLUX.1-Kontext-pro", api_key="test-key")
    assert "Api-Key" in headers
    assert headers["Api-Key"] == "test-key"


def test_azure_ai_url_generation():
    """Test Azure AI URL generation"""
    config = AzureFoundryFluxImageEditConfig()

    api_base = "https://test-endpoint.eastus2.inference.ai.azure.com"
    complete_url = config.get_complete_url(
        model="FLUX.1-Kontext-pro",
        api_base=api_base,
        litellm_params={"api_version": "2025-04-01-preview"},
    )
    expected_url = f"{api_base}/openai/deployments/FLUX.1-Kontext-pro/images/edits?api-version=2025-04-01-preview"
    assert complete_url == expected_url


def test_azure_ai_validate_environment_with_entra_token(monkeypatch):
    monkeypatch.delenv("AZURE_AI_API_KEY", raising=False)
    monkeypatch.setattr(litellm, "api_key", None)
    config = AzureFoundryFluxImageEditConfig()

    headers = config.validate_environment(
        {},
        "FLUX.1-Kontext-pro",
        litellm_params={"azure_ad_token": "entra-token"},
    )

    assert headers == {"Authorization": "Bearer entra-token"}


def test_flux2_validate_environment_with_entra_token(monkeypatch):
    monkeypatch.delenv("AZURE_AI_API_KEY", raising=False)
    monkeypatch.setattr(litellm, "api_key", None)
    config = AzureFoundryFlux2ImageEditConfig()

    headers = config.validate_environment(
        {},
        "flux.2-pro",
        litellm_params={"azure_ad_token": "entra-token"},
    )

    assert headers["Authorization"] == "Bearer entra-token"
    assert headers["Content-Type"] == "application/json"
