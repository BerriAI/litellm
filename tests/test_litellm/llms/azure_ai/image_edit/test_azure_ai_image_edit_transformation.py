

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


def test_flux2_size_param_translates_to_width_height():
    config = AzureFoundryFlux2ImageEditConfig()
    result = config.map_openai_params(
        image_edit_optional_params={"size": "896x1184"},
        model="flux.2-pro",
        drop_params=False,
    )
    assert result == {"width": 896, "height": 1184}
    assert "size" not in result


def test_flux2_explicit_width_height_pass_through():
    config = AzureFoundryFlux2ImageEditConfig()
    result = config.map_openai_params(
        image_edit_optional_params={"width": 512, "height": 512},
        model="flux.2-pro",
        drop_params=False,
    )
    assert result == {"width": 512, "height": 512}
