from litellm.llms.base_llm.image_generation.transformation import (
    BaseImageGenerationConfig,
)
from litellm.llms.vertex_ai.common_utils import (
    VertexAIModelRoute,
    get_vertex_ai_model_route,
)

from .vertex_gemini_transformation import VertexAIGeminiImageGenerationConfig
from .vertex_imagen_transformation import VertexAIImagenImageGenerationConfig

__all__ = [
    "VertexAIGeminiImageGenerationConfig", 
    "VertexAIImagenImageGenerationConfig",
    "get_vertex_ai_image_generation_config", 
]


def get_vertex_ai_image_generation_config(model: str) -> BaseImageGenerationConfig:
    """
    Get the appropriate image generation config for a Vertex AI model.
    
    Routes to the correct transformation class based on the model type:
    - Gemini image generation models use generateContent API (VertexAIGeminiImageGenerationConfig)
    - Imagen models use predict API (VertexAIImagenImageGenerationConfig)
    
    Args:
        model: The model name (e.g., "gemini-2.5-flash-image", "imagegeneration@006")
        
    Returns:
        BaseImageGenerationConfig: The appropriate configuration class
    """
    # Determine the model route
    model_route = get_vertex_ai_model_route(model)
    
    if model_route == VertexAIModelRoute.GEMINI:
        # Gemini models use generateContent API
        return VertexAIGeminiImageGenerationConfig()
    else:
        # Default to Imagen for other models (imagegeneration, etc.)
        # This includes NON_GEMINI models like imagegeneration@006
        return VertexAIImagenImageGenerationConfig()

