"""
Transformation logic from OpenAI /v1/embeddings format to Bedrock Amazon Titan G1 /invoke format. 

Why separate file? Make it easy to see how transformation works

Convers
- G1 request format

Docs - https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-titan-embed-text.html
"""

import types
from typing import List

from litellm.types.llms.bedrock import (
    AmazonTitanG1EmbeddingRequest,
    AmazonTitanG1EmbeddingResponse,
)
from litellm.types.utils import Embedding, EmbeddingResponse, Usage


class AmazonTitanG1Config:
    """
    Reference: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-titan-embed-text.html
    """

    def __init__(
        self,
    ) -> None:
        locals_ = locals()
        for key, value in locals_.items():
            if key != "self" and value is not None:
                setattr(self.__class__, key, value)

    @classmethod
    def get_config(cls):
        return {
            k: v
            for k, v in cls.__dict__.items()
            if not k.startswith("__")
            and not isinstance(
                v,
                (
                    types.FunctionType,
                    types.BuiltinFunctionType,
                    classmethod,
                    staticmethod,
                ),
            )
            and v is not None
        }

    def get_supported_openai_params(self) -> List[str]:
        return []

    def map_openai_params(
        self, non_default_params: dict, optional_params: dict
    ) -> dict:
        return optional_params

    def _transform_request(
        self, input: str, inference_params: dict
    ) -> AmazonTitanG1EmbeddingRequest:
        return AmazonTitanG1EmbeddingRequest(inputText=input)

    def _transform_response(
        self, response_list: List[dict], model: str
    ) -> EmbeddingResponse:
        total_prompt_tokens = 0

        transformed_responses: List[Embedding] = []
        for index, response in enumerate(response_list):
            _parsed_response = AmazonTitanG1EmbeddingResponse(**response)  # type: ignore
            transformed_responses.append(
                Embedding(
                    embedding=_parsed_response["embedding"],
                    index=index,
                    object="embedding",
                )
            )
            total_prompt_tokens += _parsed_response["inputTextTokenCount"]

        usage = Usage(
            prompt_tokens=total_prompt_tokens,
            completion_tokens=0,
            total_tokens=total_prompt_tokens,
        )
        return EmbeddingResponse(model=model, usage=usage, data=transformed_responses)
