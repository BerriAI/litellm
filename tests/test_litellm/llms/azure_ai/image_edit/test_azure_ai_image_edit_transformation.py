
import base64

import pytest

import litellm
from litellm.images.utils import ImageEditRequestUtils
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


def test_flux2_image_edit_maps_openai_and_provider_parameters():
    config = AzureFoundryFlux2ImageEditConfig()
    requested_params = ImageEditRequestUtils.get_requested_image_edit_optional_param(
        {
            "n": 2,
            "size": "1536x1024",
            "guidance": 4.5,
            "steps": 32,
            "unrelated": "discarded",
        },
        provider_supported_params=config.get_supported_openai_params("FLUX.2-flex"),
    )
    mapped_params = config.map_openai_params(
        image_edit_optional_params=requested_params,
        model="FLUX.2-flex",
        drop_params=False,
    )

    assert mapped_params == {
        "num_images": 2,
        "width": 1536,
        "height": 1024,
        "guidance": 4.5,
        "steps": 32,
    }


@pytest.mark.parametrize(
    ("model", "max_reference_images"),
    [
        ("FLUX.2-flex", 10),
        ("FLUX.2-pro", 8),
    ],
)
def test_flux2_image_edit_uses_all_reference_fields(model: str, max_reference_images: int):
    images = [f"image-{index}".encode() for index in range(1, max_reference_images + 1)]
    request, files = AzureFoundryFlux2ImageEditConfig().transform_image_edit_request(
        model=model,
        prompt="Blend every reference",
        image=images,
        image_edit_optional_request_params={"guidance": 4.5, "steps": 20},
        litellm_params={},
        headers={},
    )

    assert files == []
    assert request["input_image"] == base64.b64encode(images[0]).decode()
    assert request[f"input_image_{max_reference_images}"] == base64.b64encode(images[-1]).decode()
    assert "input_image_1" not in request
    assert "image" not in request
    assert len([key for key in request if key.startswith("input_image")]) == max_reference_images
    assert request["guidance"] == 4.5
    assert request["steps"] == 20


@pytest.mark.parametrize(
    ("model", "reference_images"),
    [
        ("FLUX.2-flex", 11),
        ("FLUX.2-pro", 9),
    ],
)
def test_flux2_image_edit_rejects_too_many_references(model: str, reference_images: int):
    with pytest.raises(ValueError, match=f"at most {reference_images - 1} reference images"):
        AzureFoundryFlux2ImageEditConfig().transform_image_edit_request(
            model=model,
            prompt="Blend every reference",
            image=[b"image"] * reference_images,
            image_edit_optional_request_params={},
            litellm_params={},
            headers={},
        )
