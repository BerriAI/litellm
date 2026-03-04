from litellm.llms.base_llm.image_generation.transformation import (
    BaseImageGenerationConfig,
)
from .transformation import (
    ModelsLabImageGenerationConfig,
    _redact_sensitive_fields,
)

__all__ = [
    "ModelsLabImageGenerationConfig",
    "get_modelslab_image_generation_config",
    "_redact_sensitive_fields",
]


def get_modelslab_image_generation_config(model: str) -> BaseImageGenerationConfig:
    return ModelsLabImageGenerationConfig()
