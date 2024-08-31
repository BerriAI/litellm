"""
Transformation logic from OpenAI /v1/embeddings format to Bedrock Cohere /invoke format. 

Why separate file? Make it easy to see how transformation works
"""

from typing import List

import litellm
from litellm.types.llms.bedrock import CohereEmbeddingRequest, CohereEmbeddingResponse
from litellm.types.utils import Embedding, EmbeddingResponse


def _transform_request(
    input: List[str], inference_params: dict
) -> CohereEmbeddingRequest:
    transformed_request = CohereEmbeddingRequest(
        texts=input,
        input_type=litellm.COHERE_DEFAULT_EMBEDDING_INPUT_TYPE,  # default value to prevent failed requests
    )

    for k, v in inference_params.items():
        transformed_request[k] = v

    return transformed_request
