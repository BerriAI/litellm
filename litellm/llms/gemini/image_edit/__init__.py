from litellm.llms.base_llm.image_edit.transformation import BaseImageEditConfig

from .cost_calculator import cost_calculator
from .transformation import GeminiImageEditConfig

__all__ = ["GeminiImageEditConfig", "cost_calculator", "get_gemini_image_edit_config"]


def get_gemini_image_edit_config(model: str) -> BaseImageEditConfig:
    return GeminiImageEditConfig()
