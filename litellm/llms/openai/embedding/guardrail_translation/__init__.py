"""OpenAI Embedding handler for Unified Guardrails."""

from litellm.llms.openai.embedding.guardrail_translation.handler import (
    OpenAIEmbeddingHandler,
)
from litellm.types.utils import CallTypes

guardrail_translation_mappings = {
    CallTypes.embedding: OpenAIEmbeddingHandler,
    CallTypes.aembedding: OpenAIEmbeddingHandler,
}

__all__ = ["guardrail_translation_mappings", "OpenAIEmbeddingHandler"]
