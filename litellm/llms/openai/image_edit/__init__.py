from litellm.llms.base_llm.image_edit.transformation import BaseImageEditConfig

from .dalle2_transformation import DallE2ImageEditConfig
from .transformation import OpenAIImageEditConfig

__all__ = ["OpenAIImageEditConfig", "DallE2ImageEditConfig", "get_openai_image_edit_config"]


def get_openai_image_edit_config(model: str) -> BaseImageEditConfig:
    """
    Get the appropriate OpenAI image edit config based on the model.
    
    Args:
        model: The model name (e.g., "dall-e-2", "gpt-image-1")
        
    Returns:
        The appropriate config instance for the model
    """
    model_normalized = model.lower().replace("-", "").replace("_", "")
    
    if model_normalized == "dalle2":
        return DallE2ImageEditConfig()
    else:
        # Default to standard OpenAI config for gpt-image-1 and other models
        return OpenAIImageEditConfig()

