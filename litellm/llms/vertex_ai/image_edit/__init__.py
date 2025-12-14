from litellm.llms.base_llm.image_edit.transformation import BaseImageEditConfig
from litellm.llms.vertex_ai.common_utils import VertexAIModelRoute, get_vertex_ai_model_route

from .cost_calculator import cost_calculator
from .vertex_gemini_transformation import VertexAIGeminiImageEditConfig
from .vertex_imagen_transformation import VertexAIImagenImageEditConfig

__all__ = [
    "VertexAIGeminiImageEditConfig", 
    "VertexAIImagenImageEditConfig",
    "get_vertex_ai_image_edit_config", 
    "cost_calculator"
]


def get_vertex_ai_image_edit_config(model: str) -> BaseImageEditConfig:
    """
    Get the appropriate image edit config for a Vertex AI model.
    
    Routes to the correct transformation class based on the model type:
    - Gemini models use generateContent API (VertexAIGeminiImageEditConfig)
    - Imagen models use predict API (VertexAIImagenImageEditConfig)
    
    Args:
        model: The model name (e.g., "gemini-2.5-flash", "imagegeneration@006")
        
    Returns:
        BaseImageEditConfig: The appropriate configuration class
    """
    # Determine the model route
    model_route = get_vertex_ai_model_route(model)
    
    if model_route == VertexAIModelRoute.GEMINI:
        # Gemini models use generateContent API
        return VertexAIGeminiImageEditConfig()
    else:
        # Default to Imagen for other models (imagegeneration, etc.)
        # This includes NON_GEMINI models like imagegeneration@006
        return VertexAIImagenImageEditConfig()
