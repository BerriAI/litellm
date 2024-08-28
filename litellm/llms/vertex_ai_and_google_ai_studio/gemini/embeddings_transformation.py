"""
Transformation logic from OpenAI /v1/embeddings format to Google AI Studio /embedContent format. 

Why separate file? Make it easy to see how transformation works
"""

from typing import List

from litellm.types.llms.openai import EmbeddingInput
from litellm.types.llms.vertex_ai import ContentType, PartType

from ..common_utils import VertexAIError


def transform_openai_input_gemini_content(input: EmbeddingInput) -> ContentType:
    """
    The content to embed. Only the parts.text fields will be counted.
    """
    if isinstance(input, str):
        return ContentType(parts=[PartType(text=input)])
    elif isinstance(input, list) and len(input) == 1:
        return ContentType(parts=[PartType(text=input[0])])
    else:
        raise VertexAIError(
            status_code=422,
            message="/embedContent only generates a single text embedding vector. File an issue, to add support for /batchEmbedContent - https://github.com/BerriAI/litellm/issues",
        )
