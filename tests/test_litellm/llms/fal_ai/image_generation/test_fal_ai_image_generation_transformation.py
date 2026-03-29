import os
import sys

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path

from litellm.llms.fal_ai.image_generation.transformation import (
    FalAIImageGenerationConfig,
)


class TestFalAIImageGenerationTransformation:
    def setup_method(self):
        self.config = FalAIImageGenerationConfig()

    def test_get_complete_url_uses_model_path_for_generic_models(self):
        result = self.config.get_complete_url(
            api_base=None,
            api_key="test_key",
            model="fal_ai/fal-ai/flux-2",
            optional_params={},
            litellm_params={},
        )

        assert result == "https://fal.run/fal-ai/flux-2"

    def test_get_complete_url_uses_custom_base_for_generic_models(self):
        result = self.config.get_complete_url(
            api_base="https://custom.fal.run/",
            api_key="test_key",
            model="fal_ai/fal-ai/flux-2",
            optional_params={},
            litellm_params={},
        )

        assert result == "https://custom.fal.run/fal-ai/flux-2"

    def test_map_openai_params_generic_model_maps_supported_fields(self):
        result = self.config.map_openai_params(
            non_default_params={
                "n": 2,
                "response_format": "url",
                "size": "1024x1024",
            },
            optional_params={},
            model="fal_ai/fal-ai/flux-2",
            drop_params=False,
        )

        assert result == {
            "num_images": 2,
            "output_format": "png",
            "image_size": "square_hd",
        }

    def test_map_openai_params_gpt_image_keeps_raw_dimension_size(self):
        result = self.config.map_openai_params(
            non_default_params={
                "size": "1024x1024",
            },
            optional_params={},
            model="fal_ai/fal-ai/gpt-image-1.5",
            drop_params=False,
        )

        assert result == {
            "image_size": "1024x1024",
        }
