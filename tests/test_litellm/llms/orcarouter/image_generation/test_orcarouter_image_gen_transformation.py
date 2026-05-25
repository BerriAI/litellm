"""Unit tests for OrcaRouter image generation transformation logic."""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))

from litellm.llms.orcarouter.image_generation import (
    OrcaRouterDallE2ImageGenerationConfig,
    OrcaRouterDallE3ImageGenerationConfig,
    OrcaRouterGPTImageGenerationConfig,
    get_orcarouter_image_generation_config,
)


class TestFactoryDispatch:
    def test_factory_dall_e_2_returns_dalle2_config(self):
        config = get_orcarouter_image_generation_config("orcarouter/openai/dall-e-2")
        assert isinstance(config, OrcaRouterDallE2ImageGenerationConfig)

    def test_factory_dall_e_3_returns_dalle3_config(self):
        config = get_orcarouter_image_generation_config("orcarouter/openai/dall-e-3")
        assert isinstance(config, OrcaRouterDallE3ImageGenerationConfig)

    def test_factory_gpt_image_returns_gpt_image_config(self):
        config = get_orcarouter_image_generation_config("orcarouter/openai/gpt-image-1")
        assert isinstance(config, OrcaRouterGPTImageGenerationConfig)

    def test_factory_non_openai_vendor_returns_gpt_image_config(self):
        # Imagen / grok-imagine / gemini-image — generic fallback
        for model in [
            "orcarouter/google/imagen-3.0-generate-002",
            "orcarouter/xai/grok-imagine-1",
            "orcarouter/google/gemini-2.5-flash-image",
        ]:
            config = get_orcarouter_image_generation_config(model)
            assert isinstance(
                config, OrcaRouterGPTImageGenerationConfig
            ), f"failed for {model}"

    def test_factory_strips_orcarouter_prefix_for_matching(self):
        config = get_orcarouter_image_generation_config("openai/dall-e-3")  # no prefix
        assert isinstance(config, OrcaRouterDallE3ImageGenerationConfig)


class TestUrlConstruction:
    def test_url_default(self):
        url = OrcaRouterDallE3ImageGenerationConfig().get_complete_url(
            api_base=None,
            api_key=None,
            model="orcarouter/openai/dall-e-3",
            optional_params={},
            litellm_params={},
        )
        assert url == "https://api.orcarouter.ai/v1/images/generations"

    def test_url_custom_base_strips_trailing_slash(self):
        url = OrcaRouterDallE3ImageGenerationConfig().get_complete_url(
            api_base="https://custom.orcarouter.ai/v1/",
            api_key=None,
            model="x",
            optional_params={},
            litellm_params={},
        )
        assert url == "https://custom.orcarouter.ai/v1/images/generations"

    def test_url_consistent_across_three_configs(self):
        kwargs = dict(
            api_base=None,
            api_key=None,
            model="x",
            optional_params={},
            litellm_params={},
        )
        assert (
            OrcaRouterDallE2ImageGenerationConfig().get_complete_url(**kwargs)
            == OrcaRouterDallE3ImageGenerationConfig().get_complete_url(**kwargs)
            == OrcaRouterGPTImageGenerationConfig().get_complete_url(**kwargs)
            == "https://api.orcarouter.ai/v1/images/generations"
        )


class TestValidateEnvironment:
    def test_sets_authorization_header(self):
        headers = OrcaRouterDallE3ImageGenerationConfig().validate_environment(
            headers={},
            model="x",
            messages=[],
            optional_params={},
            litellm_params={},
            api_key="sk-test",
        )
        assert headers["Authorization"] == "Bearer sk-test"
        assert headers["HTTP-Referer"] == "https://www.orcarouter.ai/"
        assert headers["X-Title"] == "liteLLM"

    def test_raises_without_api_key(self, monkeypatch):
        import litellm

        monkeypatch.delenv("ORCAROUTER_API_KEY", raising=False)
        original_key = litellm.api_key
        litellm.api_key = None
        try:
            with pytest.raises(ValueError, match="OrcaRouter API key is required"):
                OrcaRouterDallE3ImageGenerationConfig().validate_environment(
                    headers={},
                    model="x",
                    messages=[],
                    optional_params={},
                    litellm_params={},
                    api_key=None,
                )
        finally:
            litellm.api_key = original_key

    def test_user_headers_preserved(self):
        headers = OrcaRouterDallE3ImageGenerationConfig().validate_environment(
            headers={"X-Custom": "v"},
            model="x",
            messages=[],
            optional_params={},
            litellm_params={},
            api_key="sk-test",
        )
        # User custom header is preserved. Authorization overrides any user value.
        assert headers["X-Custom"] == "v"
        assert headers["Authorization"] == "Bearer sk-test"


class TestTransformRequest:
    def test_strips_orcarouter_prefix_from_model(self):
        body = (
            OrcaRouterDallE3ImageGenerationConfig().transform_image_generation_request(
                model="orcarouter/openai/dall-e-3",
                prompt="a cat",
                optional_params={"size": "1024x1024", "quality": "hd"},
                litellm_params={},
                headers={},
            )
        )
        assert body["model"] == "openai/dall-e-3"
        assert body["prompt"] == "a cat"
        assert body["size"] == "1024x1024"
        assert body["quality"] == "hd"

    def test_preserves_unprefixed_model(self):
        body = (
            OrcaRouterDallE2ImageGenerationConfig().transform_image_generation_request(
                model="openai/dall-e-2",
                prompt="x",
                optional_params={},
                litellm_params={},
                headers={},
            )
        )
        assert body["model"] == "openai/dall-e-2"


class TestInheritedSupportedParams:
    def test_dall_e_2_supports_n_size_quality(self):
        params = OrcaRouterDallE2ImageGenerationConfig().get_supported_openai_params(
            "orcarouter/openai/dall-e-2"
        )
        assert "n" in params
        assert "size" in params
        assert "quality" in params

    def test_dall_e_3_supports_style(self):
        params = OrcaRouterDallE3ImageGenerationConfig().get_supported_openai_params(
            "orcarouter/openai/dall-e-3"
        )
        assert "style" in params

    def test_gpt_image_supports_background_and_output_format(self):
        params = OrcaRouterGPTImageGenerationConfig().get_supported_openai_params(
            "orcarouter/openai/gpt-image-1"
        )
        assert "background" in params
        assert "output_format" in params
