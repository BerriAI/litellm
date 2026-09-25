import base64
import json
import struct
import zlib
from collections.abc import Mapping
from io import BytesIO
from typing import Final

import httpx
import pytest

import litellm
from litellm.images.utils import ImageEditRequestUtils
from litellm.llms.azure_ai.image_edit.flux2_transformation import (
    AzureFoundryFlux2ImageEditConfig,
)
from litellm.llms.azure_ai.image_edit.transformation import (
    AzureFoundryFluxImageEditConfig,
)
from litellm.llms.custom_httpx.http_handler import HTTPHandler


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


def test_azure_ai_validate_environment_with_entra_token(monkeypatch, no_ambient_azure_credentials):
    monkeypatch.delenv("AZURE_AI_API_KEY", raising=False)
    monkeypatch.setattr(litellm, "api_key", None)
    config = AzureFoundryFluxImageEditConfig()

    headers = config.validate_environment(
        {},
        "FLUX.1-Kontext-pro",
        litellm_params={"azure_ad_token": "entra-token"},
    )

    assert headers == {"Authorization": "Bearer entra-token"}


def test_flux2_validate_environment_with_entra_token(monkeypatch, no_ambient_azure_credentials):
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


@pytest.mark.parametrize("dimensions", ({"size": "2048x1024"}, {"width": 2048, "height": 1024}, {"width": "2048", "height": "1024"}))
@pytest.mark.usefixtures("local_model_cost_map")
def test_flux2_image_edit_preserves_controls_and_pixel_cost(dimensions: Mapping[str, int | str]):
    def respond(request: httpx.Request) -> httpx.Response:
        body: Final = json.loads(request.content)
        assert body == {
            "model": "FLUX.2-flex",
            "prompt": "Add a hat",
            "input_image": base64.b64encode(b"image").decode(),
            "num_images": 2,
            "width": 2048,
            "height": 1024,
            "guidance": 4.5,
            "steps": 32,
        }
        return httpx.Response(200, json={"data": [{"b64_json": "aW1n"}, {"b64_json": "aW1n"}]})

    client: Final = HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(respond)))
    response: Final = litellm.image_edit(
        model="azure_ai/FLUX.2-flex",
        image=b"image",
        prompt="Add a hat",
        api_key="test-key",
        api_base="https://example.services.ai.azure.com",
        client=client,
        n=2,
        guidance="4.5",
        steps="32",
        **dimensions,
    )

    catalog_rate: Final = litellm.get_model_info(model="azure_ai/FLUX.2-flex", custom_llm_provider="azure_ai")[
        "input_cost_per_pixel"
    ]
    # the reference b"image" decodes to non-image content and is not metered; generated pixels only
    assert response._hidden_params["response_cost"] == pytest.approx(catalog_rate * 2048 * 1024 * 2)


def test_flux2_image_edit_encodes_a_mid_position_stream_from_the_start():
    stream: Final = BytesIO(b"prefix" + b"image")
    stream.seek(6)

    request, files = AzureFoundryFlux2ImageEditConfig().transform_image_edit_request(
        model="FLUX.2-flex",
        prompt="Blend every reference",
        image=[stream],
        image_edit_optional_request_params={},
        litellm_params={},
        headers={},
    )

    assert request["input_image"] == base64.b64encode(b"prefiximage").decode()


def _png_bytes(width: int, height: int) -> bytes:
    """Smallest well-formed PNG carrying real IHDR dimensions."""

    def chunk(tag: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + tag
            + payload
            + struct.pack(">I", zlib.crc32(tag + payload))
        )

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(b"\x00"))
        + chunk(b"IEND", b"")
    )


@pytest.mark.usefixtures("local_model_cost_map")
def test_flux2_image_edit_bills_every_uploaded_reference():
    """Each input_image field the transform builds and posts is metered at input_cost_per_pixel."""
    catalog_rate: Final = litellm.get_model_info(model="azure_ai/FLUX.2-flex", custom_llm_provider="azure_ai")[
        "input_cost_per_pixel"
    ]

    client: Final = HTTPHandler(
        client=httpx.Client(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, json={"data": [{"b64_json": "aW1n"}]})
            )
        )
    )
    response: Final = litellm.image_edit(
        model="azure_ai/FLUX.2-flex",
        image=[_png_bytes(1024, 1024), _png_bytes(1024, 1024)],
        prompt="Blend the references",
        api_key="test-key",
        api_base="https://example.services.ai.azure.com",
        client=client,
        size="1024x1024",
    )

    assert response.reference_pixels == 2 * 1024 * 1024
    assert response._hidden_params["response_cost"] == pytest.approx(catalog_rate * 3 * 1024 * 1024)


@pytest.mark.usefixtures("local_model_cost_map")
def test_mai_image_edit_bills_only_the_reference_it_uploads():
    """MAI-Image uploads only the first caller image, so billing must follow the uploaded set,
    not the requested set: two passed images bill one uploaded reference."""
    deployment_rate: Final = 5e-08

    client: Final = HTTPHandler(
        client=httpx.Client(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, json={"data": [{"b64_json": "aW1n"}]})
            )
        )
    )
    response: Final = litellm.image_edit(
        model="azure_ai/MAI-Image-2.5",
        image=[_png_bytes(1024, 1024), _png_bytes(1024, 1024)],
        prompt="Add a hat",
        api_key="test-key",
        api_base="https://example.services.ai.azure.com",
        client=client,
        input_cost_per_pixel=deployment_rate,
    )

    catalog_price_per_image: Final = litellm.get_model_info(
        model="azure_ai/MAI-Image-2.5", custom_llm_provider="azure_ai"
    )["output_cost_per_image"]
    assert response.reference_pixels == 1024 * 1024
    assert response._hidden_params["response_cost"] == pytest.approx(
        catalog_price_per_image + deployment_rate * 1024 * 1024
    )


def test_flux2_image_edit_accepts_and_drops_openai_only_parameters():
    optional_params: Final = ImageEditRequestUtils.get_optional_params_image_edit(
        model="FLUX.2-pro",
        image_edit_provider_config=AzureFoundryFlux2ImageEditConfig(),
        image_edit_optional_params={"n": 1, "size": "auto", "quality": "high", "user": "end-user-1"},
        drop_params=False,
    )

    assert optional_params == {"num_images": 1}
