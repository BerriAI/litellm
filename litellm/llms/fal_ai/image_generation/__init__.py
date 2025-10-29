from litellm.llms.base_llm.image_generation.transformation import (
    BaseImageGenerationConfig,
)

from .bria_transformation import FalAIBriaConfig
from .flux_pro_v11_ultra_transformation import FalAIFluxProV11UltraConfig
from .imagen4_transformation import FalAIImagen4Config
from .recraft_v3_transformation import FalAIRecraftV3Config
from .stable_diffusion_transformation import FalAIStableDiffusionConfig
from .transformation import FalAIBaseConfig, FalAIImageGenerationConfig

__all__ = [
    "FalAIBaseConfig",
    "FalAIImageGenerationConfig",
    "FalAIImagen4Config",
    "FalAIRecraftV3Config",
    "FalAIBriaConfig",
    "FalAIFluxProV11UltraConfig",
    "FalAIStableDiffusionConfig",
]


def get_fal_ai_image_generation_config(model: str) -> BaseImageGenerationConfig:
    """
    Get the appropriate Fal AI image generation configuration based on the model.
    
    Args:
        model: The Fal AI model name (e.g., "fal-ai/imagen4/preview", "fal-ai/recraft/v3/text-to-image")
        
    Returns:
        The appropriate configuration class for the specified model
    """
    model_lower = model.lower()
    
    # Map model names to their corresponding configuration classes
    if "imagen4" in model_lower or "imagen-4" in model_lower:
        return FalAIImagen4Config()
    elif "recraft" in model_lower:
        return FalAIRecraftV3Config()
    elif "bria" in model_lower:
        return FalAIBriaConfig()
    elif "flux-pro" in model_lower and "ultra" in model_lower:
        return FalAIFluxProV11UltraConfig()
    elif "stable-diffusion" in model_lower:
        return FalAIStableDiffusionConfig()
    
    # Default to generic Fal AI configuration
    return FalAIImageGenerationConfig()

