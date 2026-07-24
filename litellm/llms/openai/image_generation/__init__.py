from litellm.llms.base_llm.image_generation.transformation import (
    BaseImageGenerationConfig,
)

from .dall_e_2_transformation import DallE2ImageGenerationConfig
from .dall_e_3_transformation import DallE3ImageGenerationConfig
from .gpt_transformation import GPTImageGenerationConfig
from .guardrail_translation import (
    OpenAIImageGenerationHandler,
    guardrail_translation_mappings,
)
from .openai_compatible_transformation import (
    OpenAICompatibleImageGenerationConfig,
)

__all__ = [
    "DallE2ImageGenerationConfig",
    "DallE3ImageGenerationConfig",
    "GPTImageGenerationConfig",
    "OpenAICompatibleImageGenerationConfig",
    "OpenAIImageGenerationHandler",
    "guardrail_translation_mappings",
]


def get_openai_image_generation_config(model: str) -> BaseImageGenerationConfig:
    """
    Return the OpenAI image-generation transformation config for the given model.

    - ``dall-e-2`` (and empty string) → :class:`DallE2ImageGenerationConfig`
    - ``dall-e-3*``                    → :class:`DallE3ImageGenerationConfig`
    - ``gpt-image-1*``                 → :class:`GPTImageGenerationConfig`
    - everything else routed through the ``openai/`` provider →
      :class:`OpenAICompatibleImageGenerationConfig` (generic fallback for
      community OpenAI-compatible image endpoints, e.g. third-party
      aggregators and services that expose an OpenAI-shaped
      ``/v1/images/generations``).
    """
    if model.startswith("dall-e-2") or model == "":  # empty model is dall-e-2
        return DallE2ImageGenerationConfig()
    elif model.startswith("dall-e-3"):
        return DallE3ImageGenerationConfig()
    elif model.startswith("gpt-image"):
        return GPTImageGenerationConfig()
    else:
        return OpenAICompatibleImageGenerationConfig()
