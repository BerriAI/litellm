import os
import sys

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path

from litellm.llms.azure_ai.image_edit.transformation import AzureFoundryFluxImageEditConfig


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
        litellm_params={"api_version": "2025-04-01-preview"}
    )
    expected_url = f"{api_base}/openai/deployments/FLUX.1-Kontext-pro/images/edits?api-version=2025-04-01-preview"
    assert complete_url == expected_url
