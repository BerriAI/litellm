"""
Transformation logic from OpenAI /v1/embeddings format to Bedrock Amazon Titan multimodal /invoke format.

Why separate file? Make it easy to see how transformation works

Docs - https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-titan-embed-mm.html
"""

from typing import List

from litellm.types.llms.bedrock import (
    AmazonTitanMultimodalEmbeddingConfig,
    AmazonTitanMultimodalEmbeddingRequest,
    AmazonTitanMultimodalEmbeddingResponse,
)
from litellm.types.utils import Embedding, EmbeddingResponse, Usage
from litellm.utils import get_base64_str, is_base64_encoded


class AmazonTitanMultimodalEmbeddingG1Config:
    """
    Reference - https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-titan-embed-mm.html
    """

    def __init__(self) -> None:
        pass

    def get_supported_openai_params(self) -> List[str]:
        return ["dimensions"]

    def map_openai_params(
        self, non_default_params: dict, optional_params: dict
    ) -> dict:
        for k, v in non_default_params.items():
            if k == "dimensions":
                optional_params[
                    "embeddingConfig"
                ] = AmazonTitanMultimodalEmbeddingConfig(outputEmbeddingLength=v)
        return optional_params

    def _transform_request(
        self, input: str, inference_params: dict
    ) -> AmazonTitanMultimodalEmbeddingRequest:
        ## check if b64 encoded str or not ##
        is_encoded = is_base64_encoded(input)
        if is_encoded:  # check if string is b64 encoded image or not
            b64_str = get_base64_str(input)
            transformed_request = AmazonTitanMultimodalEmbeddingRequest(
                inputImage=b64_str
            )
        else:
            transformed_request = AmazonTitanMultimodalEmbeddingRequest(inputText=input)

        for k, v in inference_params.items():
            transformed_request[k] = v  # type: ignore
        return transformed_request

    def _transform_response(
        self, response_list: List[dict], model: str
    ) -> EmbeddingResponse:
        total_prompt_tokens = 0
        transformed_responses: List[Embedding] = []
        for index, response in enumerate(response_list):
            _parsed_response = AmazonTitanMultimodalEmbeddingResponse(**response)  # type: ignore
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
