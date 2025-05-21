"""
Legacy /v1/embedding transformation logic for Bedrock Cohere. 
"""

from typing import Any, List, Optional, Union

import httpx

from litellm import COHERE_DEFAULT_EMBEDDING_INPUT_TYPE
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.types.llms.bedrock import (
    CohereEmbeddingRequest,
    CohereEmbeddingRequestWithModel,
)
from litellm.types.utils import EmbeddingResponse, PromptTokensDetailsWrapper, Usage
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

    def _calculate_usage(self, input: List[str], encoding: Any, meta: dict) -> Usage:
        input_tokens = 0

        text_tokens: Optional[int] = meta.get("billed_units", {}).get("input_tokens")

        image_tokens: Optional[int] = meta.get("billed_units", {}).get("images")

        prompt_tokens_details: Optional[PromptTokensDetailsWrapper] = None
        if image_tokens is None and text_tokens is None:
            for text in input:
                input_tokens += len(encoding.encode(text))
        else:
            prompt_tokens_details = PromptTokensDetailsWrapper(
                image_tokens=image_tokens,
                text_tokens=text_tokens,
            )
            if image_tokens:
                input_tokens += image_tokens
            if text_tokens:
                input_tokens += text_tokens

        return Usage(
            prompt_tokens=input_tokens,
            completion_tokens=0,
            total_tokens=input_tokens,
            prompt_tokens_details=prompt_tokens_details,
        )

    def _transform_response(
        self,
        response: httpx.Response,
        api_key: Optional[str],
        logging_obj: LiteLLMLoggingObj,
        data: Union[dict, CohereEmbeddingRequest],
        model_response: EmbeddingResponse,
        model: str,
        encoding: Any,
        input: list,
    ) -> EmbeddingResponse:
        response_json = response.json()
        ## LOGGING
        logging_obj.post_call(
            input=input,
            api_key=api_key,
            additional_args={"complete_input_dict": data},
            original_response=response_json,
        )
        """
            response 
            {
                'object': "list",
                'data': [
                
                ]
                'model', 
                'usage'
            }
        """
        embeddings = response_json["embeddings"]
        output_data = []
        for idx, embedding in enumerate(embeddings):
            output_data.append(
                {"object": "embedding", "index": idx, "embedding": embedding}
            )
        model_response.object = "list"
        model_response.data = output_data
        model_response.model = model
        input_tokens = 0
        for text in input:
            input_tokens += len(encoding.encode(text))

        setattr(
            model_response,
            "usage",
            self._calculate_usage(input, encoding, response_json.get("meta", {})),
        )

        return model_response
