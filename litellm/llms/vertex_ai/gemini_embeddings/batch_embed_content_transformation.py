"""
Transformation logic from OpenAI /v1/embeddings format to Google AI Studio /batchEmbedContents format. 

Why separate file? Make it easy to see how transformation works
"""

from typing import List

from litellm import EmbeddingResponse
from litellm.types.llms.openai import EmbeddingInput
from litellm.types.llms.vertex_ai import (
    ContentType,
    EmbedContentRequest,
    PartType,
    VertexAIBatchEmbeddingsRequestBody,
    VertexAIBatchEmbeddingsResponseObject,
)
from litellm.types.utils import Embedding, Usage
from litellm.utils import get_formatted_prompt, token_counter


def transform_openai_input_gemini_content(
    input: EmbeddingInput, model: str, optional_params: dict
) -> VertexAIBatchEmbeddingsRequestBody:
    """
    The content to embed. Only the parts.text fields will be counted.
    """
    gemini_model_name = "models/{}".format(model)
    requests: List[EmbedContentRequest] = []
    if isinstance(input, str):
        request = EmbedContentRequest(
            model=gemini_model_name,
            content=ContentType(parts=[PartType(text=input)]),
            **optional_params
        )
        requests.append(request)
    else:
        for i in input:
            request = EmbedContentRequest(
                model=gemini_model_name,
                content=ContentType(parts=[PartType(text=i)]),
                **optional_params
            )
            requests.append(request)

    return VertexAIBatchEmbeddingsRequestBody(requests=requests)


def process_response(
    input: EmbeddingInput,
    model_response: EmbeddingResponse,
    model: str,
    _predictions: VertexAIBatchEmbeddingsResponseObject,
) -> EmbeddingResponse:

    openai_embeddings: List[Embedding] = []
    for embedding in _predictions["embeddings"]:
        openai_embedding = Embedding(
            embedding=embedding["values"],
            index=0,
            object="embedding",
        )
        openai_embeddings.append(openai_embedding)

    model_response.data = openai_embeddings
    model_response.model = model

    input_text = get_formatted_prompt(data={"input": input}, call_type="embedding")
    prompt_tokens = token_counter(model=model, text=input_text)
    model_response.usage = Usage(
        prompt_tokens=prompt_tokens, total_tokens=prompt_tokens
    )

    return model_response
