"""Tests for FAL AI image generation configuration and request handling.

Validates:
- URL construction for generic and model-specific configs
- Parameter passthrough (FAL-specific params are not silently dropped)
- extra_body flattening (image_urls, loras, etc. sent via extra_body)
- End-to-end request body construction via aimage_generation()
- Response transformation
"""

import asyncio
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../.."))

import litellm
from litellm import aimage_generation
from litellm.llms.fal_ai.image_generation import get_fal_ai_image_generation_config
from litellm.llms.fal_ai.image_generation.transformation import (
    FalAIBaseConfig,
    FalAIImageGenerationConfig,
)


# ============================================================================
# URL Construction Tests
# ============================================================================


class TestURLConstruction:
    """Verify URL is built correctly for all FAL model types."""

    def test_generic_model_includes_model_in_url(self):
        """Generic models (no specific config) must include model name in URL."""
        cfg = FalAIImageGenerationConfig()
        url = cfg.get_complete_url(None, None, "nano-banana-2", {}, {})
        assert url == "https://fal.run/fal-ai/nano-banana-2"

    def test_generic_edit_model_includes_full_path(self):
        """Edit models must include the full path including /edit."""
        cfg = FalAIImageGenerationConfig()
        url = cfg.get_complete_url(None, None, "nano-banana-2/edit", {}, {})
        assert url == "https://fal.run/fal-ai/nano-banana-2/edit"

    def test_deep_nested_model_path(self):
        """Models with deep paths (flux-2/klein/9b/base/edit/lora)."""
        cfg = FalAIImageGenerationConfig()
        url = cfg.get_complete_url(
            None, None, "flux-2/klein/9b/base/edit/lora", {}, {}
        )
        assert url == "https://fal.run/fal-ai/flux-2/klein/9b/base/edit/lora"

    def test_gemini_image_preview_edit(self):
        """Gemini image preview edit model."""
        cfg = FalAIImageGenerationConfig()
        url = cfg.get_complete_url(
            None, None, "gemini-3-pro-image-preview/edit", {}, {}
        )
        assert url == "https://fal.run/fal-ai/gemini-3-pro-image-preview/edit"

    def test_custom_api_base_overrides_default(self):
        """Custom api_base should be used instead of default."""
        cfg = FalAIImageGenerationConfig()
        url = cfg.get_complete_url(
            "https://custom-fal.example.com", None, "nano-banana-2", {}, {}
        )
        assert url == "https://custom-fal.example.com/fal-ai/nano-banana-2"

    def test_api_base_trailing_slash_stripped(self):
        """Trailing slashes on api_base should be stripped."""
        cfg = FalAIImageGenerationConfig()
        url = cfg.get_complete_url(
            "https://fal.run/", None, "nano-banana-2", {}, {}
        )
        assert url == "https://fal.run/fal-ai/nano-banana-2"

    def test_config_with_explicit_endpoint_uses_endpoint(self):
        """Subclasses with IMAGE_GENERATION_ENDPOINT use that instead of model."""
        # Use a concrete subclass since FalAIBaseConfig is abstract
        cfg = FalAIImageGenerationConfig()
        cfg.IMAGE_GENERATION_ENDPOINT = "fal-ai/flux-pro/v1.1-ultra"
        url = cfg.get_complete_url(None, None, "ignored-model", {}, {})
        assert url == "https://fal.run/fal-ai/flux-pro/v1.1-ultra"

    def test_kling_image_model(self):
        """kling-image/omni model path."""
        cfg = FalAIImageGenerationConfig()
        url = cfg.get_complete_url(None, None, "kling-image/omni", {}, {})
        assert url == "https://fal.run/fal-ai/kling-image/omni"

    def test_bytedance_seedream_model(self):
        """ByteDance SeedDream model with deep path."""
        cfg = FalAIImageGenerationConfig()
        url = cfg.get_complete_url(
            None, None, "bytedance/seedream/v5/lite/text-to-image", {}, {}
        )
        assert (
            url == "https://fal.run/fal-ai/bytedance/seedream/v5/lite/text-to-image"
        )


# ============================================================================
# Config Selection Tests
# ============================================================================


class TestConfigSelection:
    """Verify get_fal_ai_image_generation_config returns the right config."""

    def test_generic_model_returns_default_config(self):
        cfg = get_fal_ai_image_generation_config("nano-banana-2/edit")
        assert isinstance(cfg, FalAIImageGenerationConfig)

    def test_flux_klein_returns_default_config(self):
        cfg = get_fal_ai_image_generation_config("flux-2/klein/9b/base/edit/lora")
        assert isinstance(cfg, FalAIImageGenerationConfig)

    def test_kling_returns_default_config(self):
        cfg = get_fal_ai_image_generation_config("kling-image/omni")
        assert isinstance(cfg, FalAIImageGenerationConfig)


# ============================================================================
# Parameter Passthrough Tests
# ============================================================================


class TestParamPassthrough:
    """Verify FAL-specific params pass through map_openai_params."""

    def _cfg(self):
        return FalAIImageGenerationConfig()

    def test_loras_passed_through(self):
        result = self._cfg().map_openai_params(
            {"loras": [{"path": "https://example.com/lora.safetensors", "scale": 1}]},
            {},
            "flux-2/klein/9b/base/edit/lora",
            True,
        )
        assert "loras" in result
        assert result["loras"][0]["path"] == "https://example.com/lora.safetensors"

    def test_image_urls_passed_through(self):
        result = self._cfg().map_openai_params(
            {"image_urls": ["data:image/png;base64,abc123"]},
            {},
            "nano-banana-2/edit",
            True,
        )
        assert "image_urls" in result
        assert result["image_urls"][0] == "data:image/png;base64,abc123"

    def test_inference_params_passed_through(self):
        result = self._cfg().map_openai_params(
            {
                "num_inference_steps": 28,
                "guidance_scale": 5.0,
                "seed": 42,
                "strength": 0.8,
            },
            {},
            "flux-2/klein/9b",
            True,
        )
        assert result["num_inference_steps"] == 28
        assert result["guidance_scale"] == 5.0
        assert result["seed"] == 42
        assert result["strength"] == 0.8

    def test_output_format_params_passed_through(self):
        result = self._cfg().map_openai_params(
            {
                "output_format": "png",
                "image_size": "landscape_4_3",
                "aspect_ratio": "16:9",
                "num_images": 2,
            },
            {},
            "nano-banana-2",
            True,
        )
        assert result["output_format"] == "png"
        assert result["image_size"] == "landscape_4_3"
        assert result["aspect_ratio"] == "16:9"
        assert result["num_images"] == 2

    def test_safety_params_passed_through(self):
        result = self._cfg().map_openai_params(
            {"enable_safety_checker": False, "safety_tolerance": "5"},
            {},
            "flux-2/klein/9b",
            True,
        )
        assert result["enable_safety_checker"] is False
        assert result["safety_tolerance"] == "5"

    def test_unknown_model_specific_params_still_pass_through(self):
        """Even params not in the explicit list should pass through."""
        result = self._cfg().map_openai_params(
            {"some_future_param": "value", "another_param": 42},
            {},
            "some-new-model",
            True,
        )
        assert result["some_future_param"] == "value"
        assert result["another_param"] == 42

    def test_existing_optional_params_not_overwritten(self):
        """Params already in optional_params should not be overwritten."""
        result = self._cfg().map_openai_params(
            {"seed": 99},
            {"seed": 42},
            "nano-banana-2",
            True,
        )
        assert result["seed"] == 42  # original preserved


# ============================================================================
# extra_body Flattening Tests
# ============================================================================


class TestExtraBodyFlattening:
    """Verify extra_body dict is flattened into top-level params.

    When LiteLLM proxy receives extra_body from the OpenAI client, it ends up
    as a nested dict in non_default_params. FAL-specific params like image_urls
    must be flattened to the top level for the FAL API.
    """

    def _cfg(self):
        return FalAIImageGenerationConfig()

    def test_image_urls_from_extra_body_flattened(self):
        """image_urls inside extra_body must appear at top level."""
        result = self._cfg().map_openai_params(
            {
                "extra_body": {
                    "image_urls": ["data:image/png;base64,abc123"],
                    "enable_safety_checker": False,
                }
            },
            {},
            "gemini-3-pro-image-preview/edit",
            True,
        )
        assert "image_urls" in result
        assert result["image_urls"] == ["data:image/png;base64,abc123"]
        assert result["enable_safety_checker"] is False
        assert "extra_body" not in result

    def test_extra_body_with_loras_and_inference_params(self):
        """Complex extra_body with loras and inference params."""
        result = self._cfg().map_openai_params(
            {
                "extra_body": {
                    "loras": [
                        {"path": "https://example.com/swap.safetensors", "scale": 1}
                    ],
                    "num_inference_steps": 28,
                    "guidance_scale": 5,
                    "output_format": "png",
                    "image_urls": ["data:image/png;base64,img1", "data:image/png;base64,img2"],
                }
            },
            {},
            "flux-2/klein/9b/base/edit/lora",
            True,
        )
        assert result["loras"][0]["path"] == "https://example.com/swap.safetensors"
        assert result["num_inference_steps"] == 28
        assert result["guidance_scale"] == 5
        assert result["output_format"] == "png"
        assert len(result["image_urls"]) == 2
        assert "extra_body" not in result

    def test_extra_body_with_metadata_and_labels(self):
        """Metadata and labels from extra_body should be flattened too."""
        result = self._cfg().map_openai_params(
            {
                "extra_body": {
                    "image_urls": ["data:image/png;base64,test"],
                    "metadata": {"team_id": "FRAMEO", "operation": "swap"},
                    "labels": {"environment": "dev"},
                    "resolution": "2K",
                }
            },
            {},
            "gemini-3-pro-image-preview/edit",
            True,
        )
        assert result["image_urls"] == ["data:image/png;base64,test"]
        assert result["metadata"]["team_id"] == "FRAMEO"
        assert result["labels"]["environment"] == "dev"
        assert result["resolution"] == "2K"
        assert "extra_body" not in result

    def test_extra_body_does_not_overwrite_existing_params(self):
        """Params already in optional_params should not be overwritten by extra_body."""
        result = self._cfg().map_openai_params(
            {"extra_body": {"seed": 99, "image_urls": ["data:..."]}},
            {"seed": 42},
            "nano-banana-2/edit",
            True,
        )
        assert result["seed"] == 42  # original preserved
        assert result["image_urls"] == ["data:..."]  # new param added

    def test_mixed_extra_body_and_direct_params(self):
        """Both extra_body params and direct params should be merged."""
        result = self._cfg().map_openai_params(
            {
                "extra_body": {"image_urls": ["data:..."], "enable_safety_checker": False},
                "num_inference_steps": 28,
                "guidance_scale": 5,
            },
            {},
            "flux-2/klein/9b/base/edit/lora",
            True,
        )
        assert result["image_urls"] == ["data:..."]
        assert result["enable_safety_checker"] is False
        assert result["num_inference_steps"] == 28
        assert result["guidance_scale"] == 5
        assert "extra_body" not in result

    def test_empty_extra_body_is_harmless(self):
        """Empty extra_body dict should not break anything."""
        result = self._cfg().map_openai_params(
            {"extra_body": {}, "seed": 42},
            {},
            "nano-banana-2",
            True,
        )
        assert result["seed"] == 42
        assert "extra_body" not in result

    def test_non_dict_extra_body_passed_through(self):
        """If extra_body is not a dict (edge case), pass it through as-is."""
        result = self._cfg().map_openai_params(
            {"extra_body": "unexpected_string"},
            {},
            "nano-banana-2",
            True,
        )
        # Non-dict extra_body is treated as a regular param
        assert result["extra_body"] == "unexpected_string"


# ============================================================================
# Request Transform Tests
# ============================================================================


class TestRequestTransform:
    """Verify the final request body is correct."""

    def test_basic_text_to_image_request(self):
        cfg = FalAIImageGenerationConfig()
        body = cfg.transform_image_generation_request(
            model="nano-banana-2",
            prompt="A cute cat",
            optional_params={},
            litellm_params={},
            headers={},
        )
        assert body == {"prompt": "A cute cat"}

    def test_request_with_all_params(self):
        cfg = FalAIImageGenerationConfig()
        body = cfg.transform_image_generation_request(
            model="flux-2/klein/9b/base/edit/lora",
            prompt="Swap character",
            optional_params={
                "image_urls": ["data:image/png;base64,abc"],
                "loras": [{"path": "https://example.com/lora.safetensors", "scale": 1}],
                "num_inference_steps": 28,
                "guidance_scale": 5,
                "enable_safety_checker": False,
                "output_format": "png",
            },
            litellm_params={},
            headers={},
        )
        assert body["prompt"] == "Swap character"
        assert body["image_urls"] == ["data:image/png;base64,abc"]
        assert body["loras"][0]["scale"] == 1
        assert body["num_inference_steps"] == 28
        assert body["guidance_scale"] == 5
        assert body["enable_safety_checker"] is False
        assert body["output_format"] == "png"

    def test_request_flattens_extra_body(self):
        """extra_body should be flattened into the request body by transform."""
        cfg = FalAIImageGenerationConfig()
        body = cfg.transform_image_generation_request(
            model="gemini-3-pro-image-preview/edit",
            prompt="Edit the character",
            optional_params={
                "extra_body": {
                    "image_urls": ["data:image/png;base64,abc"],
                    "enable_safety_checker": False,
                    "resolution": "2K",
                },
            },
            litellm_params={},
            headers={},
        )
        assert body["prompt"] == "Edit the character"
        assert body["image_urls"] == ["data:image/png;base64,abc"]
        assert body["enable_safety_checker"] is False
        assert body["resolution"] == "2K"
        assert "extra_body" not in body

    def test_request_flattens_extra_body_without_overwriting(self):
        """Existing params should not be overwritten by extra_body."""
        cfg = FalAIImageGenerationConfig()
        body = cfg.transform_image_generation_request(
            model="nano-banana-2/edit",
            prompt="Test",
            optional_params={
                "seed": 42,
                "extra_body": {"seed": 99, "image_urls": ["data:..."]},
            },
            litellm_params={},
            headers={},
        )
        assert body["seed"] == 42  # original preserved
        assert body["image_urls"] == ["data:..."]  # new param added
        assert "extra_body" not in body


# ============================================================================
# End-to-End Tests (mocked HTTP)
# ============================================================================


@pytest.mark.parametrize(
    "model,expected_endpoint",
    [
        ("fal_ai/fal-ai/flux-pro/v1.1-ultra", "fal-ai/flux-pro/v1.1-ultra"),
        (
            "fal_ai/fal-ai/stable-diffusion-v35-medium",
            "fal-ai/stable-diffusion-v35-medium",
        ),
    ],
)
@pytest.mark.asyncio
async def test_fal_ai_image_generation_basic(model, expected_endpoint):
    """End-to-end: basic text-to-image with correct URL and headers."""
    captured_url = None
    captured_json_data = None
    captured_headers = None

    def capture_post_call(*args, **kwargs):
        nonlocal captured_url, captured_json_data, captured_headers

        captured_url = args[0] if args else kwargs.get("url")
        captured_json_data = kwargs.get("json")
        captured_headers = kwargs.get("headers")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json.return_value = {
            "images": [
                {
                    "url": "https://example.com/generated-image.png",
                    "width": 1024,
                    "height": 768,
                    "content_type": "image/jpeg",
                }
            ],
            "seed": 42,
        }

        return mock_response

    with patch(
        "litellm.llms.custom_httpx.http_handler.HTTPHandler.post"
    ) as mock_post:
        mock_post.side_effect = capture_post_call

        response = await aimage_generation(
            model=model,
            prompt="A cute baby sea otter",
            api_key="test-fal-ai-key-12345",
        )

        assert response is not None
        assert len(response.data) > 0

        assert captured_url is not None
        assert "fal.run" in captured_url
        assert expected_endpoint in captured_url

        assert captured_headers["Authorization"] == "Key test-fal-ai-key-12345"
        assert captured_json_data["prompt"] == "A cute baby sea otter"


@pytest.mark.asyncio
async def test_fal_ai_generic_model_url_and_params():
    """End-to-end: generic model (nano-banana-2) with FAL-specific params via extra_body.

    FAL-specific params must be sent via extra_body (LiteLLM drops non-OpenAI
    kwargs that are passed directly). This matches how dash-inference sends them.
    """
    captured_url = None
    captured_json_data = None

    def capture_post_call(*args, **kwargs):
        nonlocal captured_url, captured_json_data
        captured_url = args[0] if args else kwargs.get("url")
        captured_json_data = kwargs.get("json")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json.return_value = {
            "images": [{"url": "https://fal.media/result.png"}]
        }
        return mock_response

    with patch(
        "litellm.llms.custom_httpx.http_handler.HTTPHandler.post"
    ) as mock_post:
        mock_post.side_effect = capture_post_call

        response = await aimage_generation(
            model="fal_ai/nano-banana-2",
            prompt="A test image",
            api_key="test-key",
            extra_body={
                "num_inference_steps": 28,
                "guidance_scale": 5,
                "seed": 42,
            },
        )

        assert response is not None
        assert len(response.data) == 1

        # URL must include model path
        assert captured_url is not None
        assert "fal-ai/nano-banana-2" in captured_url

        # FAL params from extra_body must be in request body (flattened, not nested)
        assert captured_json_data["prompt"] == "A test image"
        assert captured_json_data.get("num_inference_steps") == 28
        assert captured_json_data.get("guidance_scale") == 5
        assert captured_json_data.get("seed") == 42
        assert "extra_body" not in captured_json_data


@pytest.mark.asyncio
async def test_fal_ai_edit_model_with_extra_body():
    """End-to-end: edit model with image_urls in extra_body (the bug scenario).

    This simulates what dash-inference does:
    client.images.generate(model='fal_ai/...', extra_body={'image_urls': [...]})
    """
    captured_json_data = None

    def capture_post_call(*args, **kwargs):
        nonlocal captured_json_data
        captured_json_data = kwargs.get("json")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json.return_value = {
            "images": [{"url": "https://fal.media/edited.png"}]
        }
        return mock_response

    with patch(
        "litellm.llms.custom_httpx.http_handler.HTTPHandler.post"
    ) as mock_post:
        mock_post.side_effect = capture_post_call

        response = await aimage_generation(
            model="fal_ai/gemini-3-pro-image-preview/edit",
            prompt="Edit the character appearance",
            api_key="test-key",
            extra_body={
                "image_urls": ["data:image/png;base64,abc123"],
                "enable_safety_checker": False,
                "resolution": "2K",
            },
        )

        assert response is not None
        assert len(response.data) == 1

        # image_urls must be at top level, NOT nested under extra_body
        assert captured_json_data is not None
        assert "image_urls" in captured_json_data, (
            f"image_urls missing from request body. Got keys: {list(captured_json_data.keys())}"
        )
        assert captured_json_data["image_urls"] == [
            "data:image/png;base64,abc123"
        ]
        assert captured_json_data.get("enable_safety_checker") is False
        assert "extra_body" not in captured_json_data, (
            "extra_body should be flattened, not nested"
        )


@pytest.mark.asyncio
async def test_fal_ai_klein_lora_with_extra_body():
    """End-to-end: Klein LoRA edit with loras + image_urls in extra_body.

    This simulates the dash-inference fal_generate() path for character swap.
    """
    captured_json_data = None
    captured_url = None

    def capture_post_call(*args, **kwargs):
        nonlocal captured_json_data, captured_url
        captured_url = args[0] if args else kwargs.get("url")
        captured_json_data = kwargs.get("json")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json.return_value = {
            "images": [{"url": "https://fal.media/swapped.png"}]
        }
        return mock_response

    with patch(
        "litellm.llms.custom_httpx.http_handler.HTTPHandler.post"
    ) as mock_post:
        mock_post.side_effect = capture_post_call

        response = await aimage_generation(
            model="fal_ai/flux-2/klein/9b/base/edit/lora",
            prompt="Swap the character in Image 1 with Image 2",
            api_key="test-key",
            extra_body={
                "loras": [
                    {
                        "path": "https://storage.googleapis.com/loras/swap_v5.safetensors",
                        "scale": 1,
                    }
                ],
                "image_urls": [
                    "data:image/png;base64,base_image",
                    "data:image/png;base64,char_image",
                ],
                "num_inference_steps": 28,
                "guidance_scale": 5,
                "output_format": "png",
                "enable_safety_checker": False,
            },
        )

        assert response is not None

        # URL must include full model path
        assert "flux-2/klein/9b/base/edit/lora" in captured_url

        # All params from extra_body must be at top level
        assert captured_json_data["loras"][0]["scale"] == 1
        assert len(captured_json_data["image_urls"]) == 2
        assert captured_json_data["num_inference_steps"] == 28
        assert captured_json_data["guidance_scale"] == 5
        assert captured_json_data["output_format"] == "png"
        assert captured_json_data["enable_safety_checker"] is False
        assert "extra_body" not in captured_json_data
