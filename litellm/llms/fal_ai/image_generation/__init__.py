from litellm.llms.base_llm.image_generation.transformation import (
    BaseImageGenerationConfig,
)

from .imagen4_transformation import FalAIImagen4Config
from .transformation import FalAIBaseConfig, FalAIImageGenerationConfig

__all__ = [
    "FalAIBaseConfig",
    "FalAIImageGenerationConfig",
    "FalAIImagen4Config",
]


def get_fal_ai_image_generation_config(model: str) -> BaseImageGenerationConfig:
    """
    Get the appropriate Fal AI image generation configuration based on the model.
    
    Args:
        model: The Fal AI model name (e.g., "fal-ai/imagen4/preview")
        
    Returns:
        The appropriate configuration class for the specified model
    """
    # Map model names to their corresponding configuration classes
    if "imagen4" in model.lower() or "imagen-4" in model.lower():
        return FalAIImagen4Config()
    
    # Default to generic Fal AI configuration
    return FalAIImageGenerationConfig()

