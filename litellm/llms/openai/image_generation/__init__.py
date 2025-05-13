from litellm.llms.base_llm.image_generation.transformation import (
    BaseImageGenerationConfig,
)

from .dall_e_2_transformation import DallE2ImageGenerationConfig
from .dall_e_3_transformation import DallE3ImageGenerationConfig
from .gpt_transformation import GPTImageGenerationConfig

__all__ = [
    "DallE2ImageGenerationConfig",
    "DallE3ImageGenerationConfig",
    "GPTImageGenerationConfig",
]


def get_openai_image_generation_config(model: str) -> BaseImageGenerationConfig:
    if model.startswith("dall-e-2") or model == "":  # empty model is dall-e-2
        return DallE2ImageGenerationConfig()
    elif model.startswith("dall-e-3"):
        return DallE3ImageGenerationConfig()
    else:
        return GPTImageGenerationConfig()
