"""
Transformation logic from OpenAI /v1/embeddings format to Cohere's /v1/embed format.

Why separate file? Make it easy to see how transformation works

Convers
- v3 embedding models
- v2 embedding models

Docs - https://docs.cohere.com/v2/reference/embed
"""

import types
from typing import List, Optional

from litellm import COHERE_DEFAULT_EMBEDDING_INPUT_TYPE
from litellm.types.llms.bedrock import (
    COHERE_EMBEDDING_INPUT_TYPES,
    CohereEmbeddingRequest,
    CohereEmbeddingRequestWithModel,
)
from litellm.types.utils import Embedding, EmbeddingResponse, Usage
from litellm.utils import is_base64_encoded


class CohereEmbeddingConfig:
    """
    Reference: https://docs.cohere.com/v2/reference/embed
    """

    def __init__(self) -> None:
        pass

    def get_supported_openai_params(self) -> List[str]:
        return ["encoding_format"]

    def map_openai_params(
        self, non_default_params: dict, optional_params: dict
    ) -> dict:
        for k, v in non_default_params.items():
            if k == "encoding_format":
                optional_params["embedding_types"] = v
        return optional_params

    def _is_v3_model(self, model: str) -> bool:
        return "3" in model

    def _transform_request(
        self, model: str, input: List[str], inference_params: dict
    ) -> CohereEmbeddingRequestWithModel:
        is_encoded = False
        for input_str in input:
            is_encoded = is_base64_encoded(input_str)

        if is_encoded:  # check if string is b64 encoded image or not
            transformed_request = CohereEmbeddingRequestWithModel(
                model=model,
                images=input,
                input_type="image",
            )
        else:
            transformed_request = CohereEmbeddingRequestWithModel(
                model=model,
                texts=input,
                input_type=COHERE_DEFAULT_EMBEDDING_INPUT_TYPE,
            )

        for k, v in inference_params.items():
            transformed_request[k] = v  # type: ignore

        return transformed_request
