from .bge import VertexBGEConfig
from .embed_transformation import (
    VertexGeminiEmbeddingConfig as EmbedVertexGeminiEmbeddingConfig,
)
from .transformation import VertexAITextEmbeddingConfig

__all__ = [
    "VertexAITextEmbeddingConfig",
    "VertexBGEConfig",
    "EmbedVertexGeminiEmbeddingConfig",
    "get_vertex_ai_embedding_config",
]


def get_vertex_ai_embedding_config(model: str) -> VertexAITextEmbeddingConfig:
    """
    Get the appropriate embedding config for a Vertex AI model.

    Routes to the correct transformation class based on the model type:
    - Models with "embed/" prefix use embedContent API (EmbedVertexGeminiEmbeddingConfig)
    - BGE models use their own transformation (VertexBGEConfig)
    - Other models use predict API (VertexAITextEmbeddingConfig)

    Args:
        model: The model name (e.g., "embed/gemini-embedding-2-exp-11-2025", "textembedding-gecko")

    Returns:
        VertexAITextEmbeddingConfig: The appropriate configuration class
    """
    # Check if model has "embed/" prefix
    if model.startswith("embed/"):
        return EmbedVertexGeminiEmbeddingConfig()
    
    # For other models, return the standard config
    # The config itself handles routing to Gemini/BGE when needed
    return VertexAITextEmbeddingConfig()
