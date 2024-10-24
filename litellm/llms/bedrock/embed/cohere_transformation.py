"""
Transformation logic from OpenAI /v1/embeddings format to Bedrock Cohere /invoke format. 

Why separate file? Make it easy to see how transformation works
"""

from typing import List

import litellm
from litellm.types.llms.bedrock import CohereEmbeddingRequest, CohereEmbeddingResponse
from litellm.types.utils import Embedding, EmbeddingResponse


class BedrockCohereEmbeddingConfig:
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

    def _transform_request(
        self, input: List[str], inference_params: dict
    ) -> CohereEmbeddingRequest:
        transformed_request = CohereEmbeddingRequest(
            texts=input,
            input_type=litellm.COHERE_DEFAULT_EMBEDDING_INPUT_TYPE,  # type: ignore
        )

        for k, v in inference_params.items():
            transformed_request[k] = v  # type: ignore

        return transformed_request
