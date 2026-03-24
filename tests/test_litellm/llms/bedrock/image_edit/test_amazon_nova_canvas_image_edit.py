"""Unit tests for Bedrock Amazon Nova Canvas image edit (issue #24267)."""

import io

import httpx
import pytest

import litellm
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


def test_get_config_class_us_cross_region_nova_canvas():
    """Cross-region inference id us.amazon.nova-canvas-v1:0 maps via model_prices."""
    cls = BedrockImageEdit.get_config_class("us.amazon.nova-canvas-v1:0")
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


def test_get_bedrock_image_edit_config_unknown_model_falls_back_to_stability():
    """Unknown Bedrock image-edit models keep legacy Stability adapter (with warning)."""
    cfg = get_bedrock_image_edit_config_for_model("amazon.titan-image-generator-v1")
    assert isinstance(cfg, BedrockStabilityImageEditConfig)


def test_get_bedrock_helper_matches_handler_get_config_class():
    """Handler and get_bedrock_image_edit_config_for_model must agree (no silent PCM-only fallback)."""
    for model in (
        "amazon.nova-canvas-v1:0",
        "stability.stable-image-inpaint-v1:0",
    ):
        handler_cls = BedrockImageEdit.get_config_class(model)
        helper_cfg = get_bedrock_image_edit_config_for_model(model)
        assert isinstance(helper_cfg, handler_cls)


def test_handler_and_helper_agree_for_unknown_bedrock_image_model():
    """Handler and helper both fall back to Stability for unrecognized Bedrock models."""
    model = "amazon.titan-image-generator-v1"
    assert BedrockImageEdit.get_config_class(model) is BedrockStabilityImageEditConfig
    assert isinstance(
        get_bedrock_image_edit_config_for_model(model),
        BedrockStabilityImageEditConfig,
    )


def test_provider_config_manager_bedrock_nova_canvas():
    """ProviderConfigManager.get_provider_image_edit_config matches handler routing."""
    from litellm.utils import ProviderConfigManager

    cfg = ProviderConfigManager.get_provider_image_edit_config(
        "amazon.nova-canvas-v1:0",
        litellm.LlmProviders.BEDROCK,
    )
    assert isinstance(cfg, BedrockAmazonNovaCanvasImageEditConfig)


def test_provider_config_manager_bedrock_stability_inpaint():
    """ProviderConfigManager returns Stability config for Stability edit models."""
    from litellm.utils import ProviderConfigManager

    cfg = ProviderConfigManager.get_provider_image_edit_config(
        "stability.stable-image-inpaint-v1:0",
        litellm.LlmProviders.BEDROCK,
    )
    assert isinstance(cfg, BedrockStabilityImageEditConfig)


def test_provider_config_manager_bedrock_unknown_falls_back_to_stability():
    """Provider path keeps legacy Stability adapter for unrecognized Bedrock models."""
    from litellm.utils import ProviderConfigManager

    cfg = ProviderConfigManager.get_provider_image_edit_config(
        "amazon.titan-image-generator-v1",
        litellm.LlmProviders.BEDROCK,
    )
    assert isinstance(cfg, BedrockStabilityImageEditConfig)


def test_provider_config_manager_bedrock_dispatches_to_nova_transform_outpainting():
    """
    Full dispatch: utils.ProviderConfigManager -> get_bedrock_image_edit_config_for_model
    -> Nova config.transform_image_edit_request (not only direct helper calls).
    """
    from litellm.utils import ProviderConfigManager

    cfg = ProviderConfigManager.get_provider_image_edit_config(
        "amazon.nova-canvas-v1:0",
        litellm.LlmProviders.BEDROCK,
    )
    assert cfg is not None
    img = io.BytesIO(b"scene")
    mask = io.BytesIO(b"mask-bytes")
    body, _ = cfg.transform_image_edit_request(
        model="amazon.nova-canvas-v1:0",
        prompt="expand left",
        image=img,
        image_edit_optional_request_params={
            "taskType": "OUTPAINTING",
            "mask": mask,
        },
        litellm_params={},  # type: ignore[arg-type]
        headers={},
    )
    assert body["taskType"] == "OUTPAINTING"
    assert "maskImage" in body["outPaintingParams"]


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


def test_transform_request_explicit_image_variation_with_mask_honors_task_type():
    """Explicit taskType=IMAGE_VARIATION must not be overridden by mask presence."""
    config = BedrockAmazonNovaCanvasImageEditConfig()
    main = io.BytesIO(b"img-bytes")
    mask = io.BytesIO(b"mask-bytes")
    body, _ = config.transform_image_edit_request(
        model="amazon.nova-canvas-v1:0",
        prompt="vary style",
        image=main,
        image_edit_optional_request_params={
            "taskType": "IMAGE_VARIATION",
            "mask": mask,
        },
        litellm_params={},  # type: ignore[arg-type]
        headers={},
    )
    assert body["taskType"] == "IMAGE_VARIATION"
    assert "imageVariationParams" in body
    assert body["imageVariationParams"]["text"] == "vary style"
    assert len(body["imageVariationParams"]["images"]) == 1


def test_transform_request_outpainting_with_mask():
    """OUTPAINTING with OpenAI mask -> outPaintingParams.maskImage."""
    config = BedrockAmazonNovaCanvasImageEditConfig()
    main = io.BytesIO(b"img")
    mask = io.BytesIO(b"mask")
    body, _ = config.transform_image_edit_request(
        model="amazon.nova-canvas-v1:0",
        prompt="extend the sky",
        image=main,
        image_edit_optional_request_params={
            "taskType": "OUTPAINTING",
            "mask": mask,
        },
        litellm_params={},  # type: ignore[arg-type]
        headers={},
    )
    assert body["taskType"] == "OUTPAINTING"
    assert "maskImage" in body["outPaintingParams"]
    assert body["outPaintingParams"]["text"] == "extend the sky"


def test_transform_request_outpainting_with_mask_prompt():
    """OUTPAINTING with maskPrompt only (no binary mask)."""
    config = BedrockAmazonNovaCanvasImageEditConfig()
    img = io.BytesIO(b"img")
    body, _ = config.transform_image_edit_request(
        model="amazon.nova-canvas-v1:0",
        prompt="new background",
        image=img,
        image_edit_optional_request_params={
            "taskType": "OUTPAINTING",
            "maskPrompt": "the area behind the subject",
        },
        litellm_params={},  # type: ignore[arg-type]
        headers={},
    )
    assert body["taskType"] == "OUTPAINTING"
    assert body["outPaintingParams"]["maskPrompt"] == "the area behind the subject"


def test_transform_request_outpainting_prefers_mask_prompt_over_binary_mask():
    """OUTPAINTING chooses maskPrompt over maskImage when both are set (_nova_canvas_task_body)."""
    config = BedrockAmazonNovaCanvasImageEditConfig()
    main = io.BytesIO(b"img")
    mask = io.BytesIO(b"mask")
    body, _ = config.transform_image_edit_request(
        model="amazon.nova-canvas-v1:0",
        prompt="extend scene",
        image=main,
        image_edit_optional_request_params={
            "taskType": "OUTPAINTING",
            "mask": mask,
            "maskPrompt": "sky region",
        },
        litellm_params={},  # type: ignore[arg-type]
        headers={},
    )
    assert body["taskType"] == "OUTPAINTING"
    op = body["outPaintingParams"]
    assert op["maskPrompt"] == "sky region"
    assert "maskImage" not in op


def test_transform_request_outpainting_with_out_painting_mode():
    """OUTPAINTING forwards outPaintingMode into outPaintingParams."""
    config = BedrockAmazonNovaCanvasImageEditConfig()
    img = io.BytesIO(b"img")
    mask = io.BytesIO(b"m")
    body, _ = config.transform_image_edit_request(
        model="amazon.nova-canvas-v1:0",
        prompt="widen",
        image=img,
        image_edit_optional_request_params={
            "taskType": "OUTPAINTING",
            "mask": mask,
            "outPaintingMode": "PRECISE",
        },
        litellm_params={},  # type: ignore[arg-type]
        headers={},
    )
    assert body["taskType"] == "OUTPAINTING"
    assert body["outPaintingParams"]["outPaintingMode"] == "PRECISE"


def test_get_supported_openai_params_includes_outpainting_fields():
    """Documented OUTPAINTING-related optional params are advertised for routing/UI."""
    config = BedrockAmazonNovaCanvasImageEditConfig()
    supported = config.get_supported_openai_params("amazon.nova-canvas-v1:0")
    assert "taskType" in supported
    assert "maskPrompt" in supported
    assert "outPaintingMode" in supported
    assert "mask" in supported


def test_transform_request_outpainting_without_mask_raises():
    """OUTPAINTING without maskPrompt or maskImage must fail fast with a clear error."""
    config = BedrockAmazonNovaCanvasImageEditConfig()
    img = io.BytesIO(b"img")
    with pytest.raises(
        ValueError,
        match="OUTPAINTING requires either a mask image or a mask prompt",
    ):
        config.transform_image_edit_request(
            model="amazon.nova-canvas-v1:0",
            prompt="extend",
            image=img,
            image_edit_optional_request_params={"taskType": "OUTPAINTING"},
            litellm_params={},  # type: ignore[arg-type]
            headers={},
        )


def test_transform_request_inpainting_explicit_task_without_mask_raises():
    """INPAINTING taskType without mask or maskPrompt must fail fast."""
    config = BedrockAmazonNovaCanvasImageEditConfig()
    img = io.BytesIO(b"img")
    with pytest.raises(
        ValueError, match="INPAINTING requires either maskPrompt or maskImage"
    ):
        config.transform_image_edit_request(
            model="amazon.nova-canvas-v1:0",
            prompt="fix it",
            image=img,
            image_edit_optional_request_params={"taskType": "INPAINTING"},
            litellm_params={},  # type: ignore[arg-type]
            headers={},
        )


def test_transform_request_unknown_task_type_raises():
    """Unknown taskType must not silently map to IMAGE_VARIATION or INPAINTING."""
    config = BedrockAmazonNovaCanvasImageEditConfig()
    img = io.BytesIO(b"img")
    with pytest.raises(ValueError, match="Unsupported Amazon Nova Canvas taskType"):
        config.transform_image_edit_request(
            model="amazon.nova-canvas-v1:0",
            prompt="x",
            image=img,
            image_edit_optional_request_params={"taskType": "TEXT_IMAGE"},
            litellm_params={},  # type: ignore[arg-type]
            headers={},
        )


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


def test_transform_request_background_removal_omits_image_generation_config():
    """AWS Nova Canvas does not allow imageGenerationConfig on BACKGROUND_REMOVAL."""
    config = BedrockAmazonNovaCanvasImageEditConfig()
    img = io.BytesIO(b"x")
    body, _ = config.transform_image_edit_request(
        model="amazon.nova-canvas-v1:0",
        prompt="ignored",
        image=img,
        image_edit_optional_request_params={
            "taskType": "BACKGROUND_REMOVAL",
            "size": "512x512",
            "seed": 42,
            "quality": "standard",
            "cfgScale": 7.5,
            "n": 2,
        },
        litellm_params={},  # type: ignore[arg-type]
        headers={},
    )
    assert body["taskType"] == "BACKGROUND_REMOVAL"
    assert "imageGenerationConfig" not in body


def test_transform_request_image_variation_includes_image_generation_config():
    """Non-BACKGROUND_REMOVAL tasks may include imageGenerationConfig when params are set."""
    config = BedrockAmazonNovaCanvasImageEditConfig()
    img = io.BytesIO(b"x")
    body, _ = config.transform_image_edit_request(
        model="amazon.nova-canvas-v1:0",
        prompt="warm",
        image=img,
        image_edit_optional_request_params={"size": "1024x1024", "seed": 1},
        litellm_params={},  # type: ignore[arg-type]
        headers={},
    )
    assert body["taskType"] == "IMAGE_VARIATION"
    assert "imageGenerationConfig" in body
    assert body["imageGenerationConfig"]["width"] == 1024
    assert body["imageGenerationConfig"]["height"] == 1024
    assert body["imageGenerationConfig"]["seed"] == 1


def test_is_nova_canvas_image_edit_model_uses_model_cost_flag(monkeypatch):
    """Routing uses supports_nova_canvas_image_edit in model_cost, not a hardcoded name substring."""
    fake_id = "amazon.custom-bedrock-image-edit-v99:0"
    monkeypatch.setitem(
        litellm.model_cost,
        fake_id,
        {
            "litellm_provider": "bedrock",
            "mode": "image_generation",
            "supports_nova_canvas_image_edit": True,
        },
    )
    assert (
        BedrockAmazonNovaCanvasImageEditConfig._is_nova_canvas_image_edit_model(fake_id)
        is True
    )

    monkeypatch.setitem(
        litellm.model_cost,
        "amazon.not-nova-canvas-v1:0",
        {
            "litellm_provider": "bedrock",
            "mode": "image_generation",
        },
    )
    assert (
        BedrockAmazonNovaCanvasImageEditConfig._is_nova_canvas_image_edit_model(
            "amazon.not-nova-canvas-v1:0"
        )
        is False
    )


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


def test_transform_response_non_200_raises():
    """HTTP 4xx/5xx with JSON body surfaces a structured error."""
    config = BedrockAmazonNovaCanvasImageEditConfig()
    resp = httpx.Response(
        400,
        json={"message": "ValidationException: invalid input"},
    )
    with pytest.raises(Exception, match="Nova Canvas image edit error"):
        config.transform_image_edit_response(
            model="amazon.nova-canvas-v1:0",
            raw_response=resp,
            logging_obj=None,  # type: ignore[arg-type]
        )


def test_transform_response_errors_field_raises():
    """Align with Bedrock Stability: top-level ``errors`` in body."""
    config = BedrockAmazonNovaCanvasImageEditConfig()
    resp = httpx.Response(
        200,
        json={"errors": ["upstream failure"]},
    )
    with pytest.raises(Exception, match="Nova Canvas image edit error"):
        config.transform_image_edit_response(
            model="amazon.nova-canvas-v1:0",
            raw_response=resp,
            logging_obj=None,  # type: ignore[arg-type]
        )


def test_transform_response_message_or_error_field_raises():
    """API-level error payload in JSON body is raised even when status is 200."""
    config = BedrockAmazonNovaCanvasImageEditConfig()
    resp = httpx.Response(
        200,
        json={"message": "ValidationException: task rejected"},
    )
    with pytest.raises(Exception, match="Nova Canvas image edit error"):
        config.transform_image_edit_response(
            model="amazon.nova-canvas-v1:0",
            raw_response=resp,
            logging_obj=None,  # type: ignore[arg-type]
        )


def test_transform_response_non_null_finish_reason_raises():
    """Align with Bedrock Stability: non-null first finish_reason indicates failure."""
    config = BedrockAmazonNovaCanvasImageEditConfig()
    resp = httpx.Response(
        200,
        json={"images": [], "finish_reasons": ["content_filtered"]},
    )
    with pytest.raises(Exception, match="Nova Canvas image edit error"):
        config.transform_image_edit_response(
            model="amazon.nova-canvas-v1:0",
            raw_response=resp,
            logging_obj=None,  # type: ignore[arg-type]
        )
