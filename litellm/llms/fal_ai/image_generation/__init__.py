from litellm.llms.base_llm.image_generation.transformation import (
    BaseImageGenerationConfig,
)

from .bria_transformation import FalAIBriaConfig
from .flux_pro_v11_transformation import FalAIFluxProV11Config
from .flux_pro_v11_ultra_transformation import FalAIFluxProV11UltraConfig
from .flux_schnell_transformation import FalAIFluxSchnellConfig
from .imagen4_transformation import FalAIImagen4Config
from .recraft_v3_transformation import FalAIRecraftV3Config
from .ideogram_v3_transformation import FalAIIdeogramV3Config
from .stable_diffusion_transformation import FalAIStableDiffusionConfig
from .transformation import FalAIBaseConfig, FalAIImageGenerationConfig
from .bytedance_transformation import (
    FalAIBytedanceSeedreamV3Config,
    FalAIBytedanceDreaminaV31Config,
)

__all__ = [
    "FalAIBaseConfig",
    "FalAIImageGenerationConfig",
    "FalAIImagen4Config",
    "FalAIRecraftV3Config",
    "FalAIBriaConfig",
    "FalAIFluxProV11Config",
    "FalAIFluxProV11UltraConfig",
    "FalAIFluxSchnellConfig",
    "FalAIStableDiffusionConfig",
    "FalAIBytedanceSeedreamV3Config",
    "FalAIBytedanceDreaminaV31Config",
    "FalAIIdeogramV3Config",
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
    elif "flux-pro" in model_lower:
        if "ultra" in model_lower:
            return FalAIFluxProV11UltraConfig()
        return FalAIFluxProV11Config()
    elif "flux/schnell" in model_lower or "flux-schnell" in model_lower or "schnell" in model_lower:
        return FalAIFluxSchnellConfig()
    elif "bytedance/seedream" in model_lower:
        return FalAIBytedanceSeedreamV3Config()
    elif "bytedance/dreamina" in model_lower:
        return FalAIBytedanceDreaminaV31Config()
    elif "ideogram" in model_lower:
        return FalAIIdeogramV3Config()
    elif "stable-diffusion" in model_lower:
        return FalAIStableDiffusionConfig()
    
    # Default to generic Fal AI configuration
    return FalAIImageGenerationConfig()

