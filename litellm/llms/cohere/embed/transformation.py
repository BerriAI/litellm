"""
Transformation logic from OpenAI /v1/embeddings format to Cohere's /v1/embed format.

Why separate file? Make it easy to see how transformation works

Convers
- v3 embedding models
- v2 embedding models

Docs - https://docs.cohere.com/v2/reference/embed
"""

from typing import Any, List, Optional, Union, cast

import httpx

import litellm
from litellm import COHERE_DEFAULT_EMBEDDING_INPUT_TYPE
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.base_llm import BaseEmbeddingConfig
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.types.llms.bedrock import (
    CohereEmbeddingRequest,
    CohereEmbeddingRequestWithModel,
)
from litellm.types.llms.openai import AllEmbeddingInputValues, AllMessageValues
from litellm.types.utils import EmbeddingResponse, PromptTokensDetailsWrapper, Usage
from litellm.utils import is_base64_encoded

from ..common_utils import CohereError


class CohereEmbeddingConfig(BaseEmbeddingConfig):
    """
    Reference: https://docs.cohere.com/v2/reference/embed
    """

    def __init__(self) -> None:
        pass

    def get_supported_openai_params(self, model: str) -> List[str]:
        return ["encoding_format", "dimensions"]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool = False,
    ) -> dict:
        for k, v in non_default_params.items():
            if k == "encoding_format":
                if isinstance(v, list):
                    optional_params["embedding_types"] = v
                else:
                    optional_params["embedding_types"] = [v]
            elif k == "dimensions":
                optional_params["output_dimension"] = v
        return optional_params

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> dict:
        default_headers = {
            "Content-Type": "application/json",
        }
        if api_key:
            default_headers["Authorization"] = f"Bearer {api_key}"
        headers = {**default_headers, **headers}
        return headers

    def _is_v3_model(self, model: str) -> bool:
        return "3" in model

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        return api_base or "https://api.cohere.ai/v2/embed"

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

    def transform_embedding_request(
        self,
        model: str,
        input: AllEmbeddingInputValues,
        optional_params: dict,
        headers: dict,
    ) -> dict:
        if isinstance(input, list) and (
            isinstance(input[0], list) or isinstance(input[0], int)
        ):
            raise ValueError("Input must be a list of strings")
        return cast(
            dict,
            self._transform_request(
                model=model,
                input=cast(List[str], input) if isinstance(input, List) else [input],
                inference_params=optional_params,
            ),
        )

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
        for k, embedding_list in embeddings.items():
            for idx, embedding in enumerate(embedding_list):
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

    def transform_embedding_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: EmbeddingResponse,
        logging_obj: LiteLLMLoggingObj,
        api_key: Optional[str],
        request_data: dict,
        optional_params: dict,
        litellm_params: dict,
    ) -> EmbeddingResponse:
        return self._transform_response(
            response=raw_response,
            api_key=api_key,
            logging_obj=logging_obj,
            data=request_data,
            model_response=model_response,
            model=model,
            encoding=litellm.encoding,
            input=logging_obj.model_call_details["input"],
        )

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, httpx.Headers]
    ) -> BaseLLMException:
        return CohereError(
            status_code=status_code,
            message=error_message,
        )
