"""Unit tests for Bedrock Amazon Nova Canvas image edit (issue #24267)."""

import io

import httpx

from litellm.llms.bedrock.image_edit.amazon_nova_canvas_image_edit_transformation import (
    BedrockAmazonNovaCanvasImageEditConfig,
    get_bedrock_image_edit_config_for_model,
)
from litellm.llms.bedrock.image_edit.handler import BedrockImageEdit
from litellm.llms.bedrock.image_edit.stability_transformation import (
    BedrockStabilityImageEditConfig,
)


def test_get_config_class_nova_canvas():
    """Nova Canvas model resolves to BedrockAmazonNovaCanvasImageEditConfig."""
    cls = BedrockImageEdit.get_config_class("amazon.nova-canvas-v1:0")
    assert cls is BedrockAmazonNovaCanvasImageEditConfig


def test_get_config_class_stability_unchanged():
    """Stability edit models still use stability config."""
    cls = BedrockImageEdit.get_config_class(
        "stability.stable-image-inpaint-v1:0",
    )
    assert cls is BedrockStabilityImageEditConfig


def test_provider_config_router_returns_nova_for_canvas():
    """ProviderConfigManager routes Nova Canvas to Nova image-edit config."""
    cfg = get_bedrock_image_edit_config_for_model("amazon.nova-canvas-v1:0")
    assert isinstance(cfg, BedrockAmazonNovaCanvasImageEditConfig)


def test_provider_config_router_returns_stability_for_sd():
    """Non-Nova Bedrock image edit still uses Stability config."""
    cfg = get_bedrock_image_edit_config_for_model(
        "stability.stable-image-inpaint-v1:0",
    )
    assert isinstance(cfg, BedrockStabilityImageEditConfig)


def test_transform_request_image_variation_without_mask():
    """No mask -> IMAGE_VARIATION with images + text."""
    config = BedrockAmazonNovaCanvasImageEditConfig()
    img = io.BytesIO(b"fake-png")
    body, _ = config.transform_image_edit_request(
        model="amazon.nova-canvas-v1:0",
        prompt="make it warmer",
        image=img,
        image_edit_optional_request_params={},
        litellm_params={},  # type: ignore[arg-type]
        headers={},
    )
    assert body["taskType"] == "IMAGE_VARIATION"
    assert body["imageVariationParams"]["text"] == "make it warmer"
    assert len(body["imageVariationParams"]["images"]) == 1


def test_transform_request_inpainting_with_mask():
    """Mask present -> INPAINTING with inPaintingParams (AWS field names)."""
    config = BedrockAmazonNovaCanvasImageEditConfig()
    main = io.BytesIO(b"img-bytes")
    mask = io.BytesIO(b"mask-bytes")
    body, _ = config.transform_image_edit_request(
        model="amazon.nova-canvas-v1:0",
        prompt="add a hat",
        image=main,
        image_edit_optional_request_params={"mask": mask},
        litellm_params={},  # type: ignore[arg-type]
        headers={},
    )
    assert body["taskType"] == "INPAINTING"
    ip = body["inPaintingParams"]
    assert ip["text"] == "add a hat"
    assert "maskImage" in ip
    assert ip["image"]  # base64


def test_transform_request_background_removal():
    """taskType BACKGROUND_REMOVAL builds minimal body."""
    config = BedrockAmazonNovaCanvasImageEditConfig()
    img = io.BytesIO(b"x")
    body, _ = config.transform_image_edit_request(
        model="amazon.nova-canvas-v1:0",
        prompt="ignored",
        image=img,
        image_edit_optional_request_params={"taskType": "BACKGROUND_REMOVAL"},
        litellm_params={},  # type: ignore[arg-type]
        headers={},
    )
    assert body["taskType"] == "BACKGROUND_REMOVAL"
    assert "image" in body["backgroundRemovalParams"]


def test_transform_response_to_openai_format():
    """Response maps images[] to ImageResponse.data b64_json."""
    config = BedrockAmazonNovaCanvasImageEditConfig()
    resp = httpx.Response(
        200,
        json={"images": ["YmFzZTY0X2E=", "YmFzZTY0X2I="]},
    )
    model_response = config.transform_image_edit_response(
        model="amazon.nova-canvas-v1:0",
        raw_response=resp,
        logging_obj=None,  # type: ignore[arg-type]
    )
    assert model_response.data is not None
    assert len(model_response.data) == 2
    assert model_response.data[0].b64_json == "YmFzZTY0X2E="
