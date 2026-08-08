"""MiniMax image generation transformation."""

from litellm.llms.base_llm.image_generation.transformation import (
    BaseImageGenerationConfig,
)

from .transformation import MinimaxImageGenerationConfig

__all__ = ["MinimaxImageGenerationConfig", "get_minimax_image_generation_config"]


def get_minimax_image_generation_config(model: str) -> BaseImageGenerationConfig:
    """Get the MiniMax image generation config for the given model."""
    return MinimaxImageGenerationConfig()
