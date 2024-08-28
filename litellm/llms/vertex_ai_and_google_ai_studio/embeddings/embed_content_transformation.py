"""
Transformation logic from OpenAI /v1/embeddings format to Google AI Studio /embedContent format. 

Why separate file? Make it easy to see how transformation works
"""

from litellm import EmbeddingResponse
from litellm.types.llms.openai import EmbeddingInput
from litellm.types.llms.vertex_ai import (
    ContentType,
    PartType,
    VertexAITextEmbeddingsResponseObject,
)
from litellm.types.utils import Embedding, Usage
from litellm.utils import get_formatted_prompt, token_counter

from ..common_utils import VertexAIError


def transform_openai_input_gemini_content(input: str) -> ContentType:
    """
    The content to embed. Only the parts.text fields will be counted.
    """
    return ContentType(parts=[PartType(text=input)])


def process_response(
    input: EmbeddingInput,
    model_response: EmbeddingResponse,
    model: str,
    _predictions: VertexAITextEmbeddingsResponseObject,
) -> EmbeddingResponse:
    model_response.data = [
        Embedding(
            embedding=_predictions["embedding"]["values"],
            index=0,
            object="embedding",
        )
    ]

    model_response.model = model

    input_text = get_formatted_prompt(data={"input": input}, call_type="embedding")
    prompt_tokens = token_counter(model=model, text=input_text)
    model_response.usage = Usage(
        prompt_tokens=prompt_tokens, total_tokens=prompt_tokens
    )

    return model_response
