from typing import Final

from litellm.llms.base_llm.image_generation.transformation import (
    BaseImageGenerationConfig,
)

from .bria_transformation import FalAIBriaConfig
from .bytedance_transformation import (
    FalAIBytedanceDreaminaV31Config,
    FalAIBytedanceSeedreamV3Config,
)
from .flux_pro_v11_transformation import FalAIFluxProV11Config
from .flux_pro_v11_ultra_transformation import FalAIFluxProV11UltraConfig
from .flux_schnell_transformation import FalAIFluxSchnellConfig
from .ideogram_v3_transformation import FalAIIdeogramV3Config
from .imagen4_transformation import FalAIImagen4Config
from .nano_banana_transformation import FalAINanoBananaConfig
from .recraft_v3_transformation import FalAIRecraftV3Config
from .stable_diffusion_transformation import FalAIStableDiffusionConfig
from .transformation import FalAIBaseConfig, FalAIImageGenerationConfig

__all__ = [
    "FalAIBaseConfig",
    "FalAIBriaConfig",
    "FalAIBytedanceDreaminaV31Config",
    "FalAIBytedanceSeedreamV3Config",
    "FalAIFluxProV11Config",
    "FalAIFluxProV11UltraConfig",
    "FalAIFluxSchnellConfig",
    "FalAIIdeogramV3Config",
    "FalAIImageGenerationConfig",
    "FalAIImagen4Config",
    "FalAINanoBananaConfig",
    "FalAIRecraftV3Config",
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
    model_lower: Final = model.lower()

    # Map model names to their corresponding configuration classes
    if "nano-banana" in model_lower or "gemini-25-flash-image" in model_lower:
        return FalAINanoBananaConfig()
    elif "imagen4" in model_lower or "imagen-4" in model_lower:
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
