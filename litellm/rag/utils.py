"""
RAG utility functions.

Provides provider configuration utilities similar to ProviderConfigManager.
"""

from typing import TYPE_CHECKING, Type

if TYPE_CHECKING:
    from litellm.rag.ingestion.base_ingestion import BaseRAGIngestion


def get_rag_ingestion_class(custom_llm_provider: str) -> Type["BaseRAGIngestion"]:
    """
    Get the appropriate RAG ingestion class for a provider.

    Args:
        custom_llm_provider: The LLM provider name (e.g., "openai", "bedrock", "vertex_ai")

    Returns:
        The ingestion class for the provider

    Raises:
        ValueError: If the provider is not supported
    """
    from litellm.llms.vertex_ai.rag_engine.ingestion import VertexAIRAGIngestion
    from litellm.rag.ingestion.bedrock_ingestion import BedrockRAGIngestion
    from litellm.rag.ingestion.openai_ingestion import OpenAIRAGIngestion

    provider_map = {
        "openai": OpenAIRAGIngestion,
        "bedrock": BedrockRAGIngestion,
        "vertex_ai": VertexAIRAGIngestion,
    }

    ingestion_class = provider_map.get(custom_llm_provider)
    if ingestion_class is None:
        raise ValueError(
            f"RAG ingestion not supported for provider: {custom_llm_provider}. "
            f"Supported providers: {list(provider_map.keys())}"
        )

    return ingestion_class


def get_rag_transformation_class(custom_llm_provider: str):
    """
    Get the appropriate RAG transformation class for a provider.

    Args:
        custom_llm_provider: The LLM provider name

    Returns:
        The transformation class for the provider, or None if not needed
    """
    if custom_llm_provider == "vertex_ai":
        from litellm.llms.vertex_ai.rag_engine.transformation import (
            VertexAIRAGTransformation,
        )

        return VertexAIRAGTransformation

    # OpenAI and Bedrock don't need special transformations
    return None

