"""
Transformation logic from OpenAI /v1/embeddings format to Bedrock Cohere /invoke format. 

Why separate file? Make it easy to see how transformation works
"""

from typing import List

from litellm.llms.cohere.embed.transformation import CohereEmbeddingConfig
from litellm.types.llms.bedrock import CohereEmbeddingRequest


class BedrockCohereEmbeddingConfig:
    def __init__(self) -> None:
        pass

    def get_supported_openai_params(self) -> List[str]:
        return ["encoding_format", "dimensions"]

    def map_openai_params(
        self, non_default_params: dict, optional_params: dict
    ) -> dict:
        for k, v in non_default_params.items():
            if k == "encoding_format":
                optional_params["embedding_types"] = v
            elif k == "dimensions":
                optional_params["output_dimension"] = v
        return optional_params

    def _is_v3_model(self, model: str) -> bool:
        return "3" in model

    def _transform_request(
        self, model: str, input: List[str], inference_params: dict
    ) -> CohereEmbeddingRequest:
        transformed_request = CohereEmbeddingConfig()._transform_request(
            model, input, inference_params
        )

        new_transformed_request = CohereEmbeddingRequest(
            input_type=transformed_request["input_type"],
        )
        for k in CohereEmbeddingRequest.__annotations__.keys():
            if k in transformed_request:
                new_transformed_request[k] = transformed_request[k]  # type: ignore

        return new_transformed_request
