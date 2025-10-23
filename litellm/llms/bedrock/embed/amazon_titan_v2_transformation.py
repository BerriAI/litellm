"""
Transformation logic from OpenAI /v1/embeddings format to Bedrock Amazon Titan V2 /invoke format.

Why separate file? Make it easy to see how transformation works

Convers
- v2 request format

Docs - https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-titan-embed-text.html
"""

import types
from typing import List, Optional, Union

from litellm.types.llms.bedrock import (
    AmazonTitanV2EmbeddingRequest,
    AmazonTitanV2EmbeddingResponse,
)
from litellm.types.utils import Embedding, EmbeddingResponse, Usage


class AmazonTitanV2Config:
    """
    Reference: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-titan-embed-text.html

    normalize: boolean - flag indicating whether or not to normalize the output embeddings. Defaults to true
    dimensions: int - The number of dimensions the output embeddings should have. The following values are accepted: 1024 (default), 512, 256.
    """

    normalize: Optional[bool] = None
    dimensions: Optional[int] = None

    def __init__(self, normalize: Optional[bool] = None, dimensions: Optional[int] = None) -> None:
        locals_ = locals().copy()
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
        return ["dimensions", "encoding_format"]

    def map_openai_params(self, non_default_params: dict, optional_params: dict) -> dict:
        for k, v in non_default_params.items():
            if k == "dimensions":
                optional_params["dimensions"] = v
            elif k == "encoding_format":
                # Map OpenAI encoding_format to AWS embeddingTypes
                if v == "float":
                    optional_params["embeddingTypes"] = ["float"]
                elif v == "base64":
                    # base64 maps to binary format in AWS
                    optional_params["embeddingTypes"] = ["binary"]
                else:
                    # For any other encoding format, default to float
                    optional_params["embeddingTypes"] = ["float"]
        return optional_params

    def _transform_request(self, input: str, inference_params: dict) -> AmazonTitanV2EmbeddingRequest:
        return AmazonTitanV2EmbeddingRequest(inputText=input, **inference_params)  # type: ignore

    def _transform_response(self, response_list: List[dict], model: str) -> EmbeddingResponse:
        total_prompt_tokens = 0

        transformed_responses: List[Embedding] = []
        for index, response in enumerate(response_list):
            _parsed_response = AmazonTitanV2EmbeddingResponse(**response)  # type: ignore

            # According to AWS docs, embeddingsByType is always present
            # If binary was requested (encoding_format="base64"), use binary data
            # Otherwise, use float data from embeddingsByType or fallback to embedding field
            embedding_data: Union[List[float], List[int]]

            if ("embeddingsByType" in _parsed_response and
                "binary" in _parsed_response["embeddingsByType"]):
                # Use binary data if available (for encoding_format="base64")
                embedding_data = _parsed_response["embeddingsByType"]["binary"]
            elif ("embeddingsByType" in _parsed_response and
                  "float" in _parsed_response["embeddingsByType"]):
                # Use float data from embeddingsByType
                embedding_data = _parsed_response["embeddingsByType"]["float"]
            elif "embedding" in _parsed_response:
                # Fallback to legacy embedding field
                embedding_data = _parsed_response["embedding"]
            else:
                raise ValueError(f"No embedding data found in response: {response}")

            transformed_responses.append(
                Embedding(
                    embedding=embedding_data,
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
