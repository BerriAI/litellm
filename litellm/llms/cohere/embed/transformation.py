"""
Transformation logic from OpenAI /v1/embeddings format to Cohere's /v1/embed format.

Why separate file? Make it easy to see how transformation works

Convers
- v3 embedding models
- v2 embedding models

Docs - https://docs.cohere.com/v2/reference/embed
"""

from typing import Any, Final, cast

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
from litellm.types.llms.cohere import CohereEmbeddingInputList
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

    def get_supported_openai_params(self, model: str) -> list[str]:
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
        messages: list[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict:
        default_headers: Final = {
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
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: bool | None = None,
    ) -> str:
        return api_base or "https://api.cohere.ai/v2/embed"

    def _transform_request(
        self,
        model: str,
        input: list[str] | CohereEmbeddingInputList,
        inference_params: dict,
    ) -> CohereEmbeddingRequestWithModel:
        is_structured_input: Final = bool(input) and isinstance(input[0], dict)
        transformed_request: Final = (
            CohereEmbeddingRequestWithModel(
                model=model,
                inputs=cast(  # cast-ok: the first-item check narrows this homogeneous input list
                    CohereEmbeddingInputList, input
                ),
                input_type=COHERE_DEFAULT_EMBEDDING_INPUT_TYPE,
            )
            if is_structured_input
            else self._transform_string_request(
                model=model,
                input=cast(  # cast-ok: the structured-input branch was excluded above
                    list[str], input
                ),
            )
        )

        for k, v in inference_params.items():
            transformed_request[k] = v

        return transformed_request

    def _transform_string_request(
        self,
        model: str,
        input: list[str],  # mutable-ok: Cohere's JSON request schema requires an array
    ) -> CohereEmbeddingRequestWithModel:
        is_encoded: Final = bool(input) and is_base64_encoded(input[-1])
        return (
            CohereEmbeddingRequestWithModel(
                model=model,
                images=input,
                input_type="image",
            )
            if is_encoded
            else CohereEmbeddingRequestWithModel(
                model=model,
                texts=input,
                input_type=COHERE_DEFAULT_EMBEDDING_INPUT_TYPE,
            )
        )

    def _normalize_embedding_input(
        self,
        input: AllEmbeddingInputValues | CohereEmbeddingInputList,
    ) -> list[str] | CohereEmbeddingInputList:  # mutable-ok: provider transformation consumes JSON arrays
        if isinstance(input, str):
            return [input]
        if not input:
            raise ValueError("Input must not be empty")
        if isinstance(input[0], dict):
            return cast(  # cast-ok: the first-item check narrows this homogeneous input list
                CohereEmbeddingInputList, input
            )
        if isinstance(input[0], list) or isinstance(input[0], int):
            raise ValueError("Input must be a list of strings")
        return cast(list[str], input)

    def transform_embedding_request(
        self,
        model: str,
        input: AllEmbeddingInputValues | CohereEmbeddingInputList,
        optional_params: dict,
        headers: dict,
    ) -> dict:
        return cast(
            dict,
            self._transform_request(
                model=model,
                input=self._normalize_embedding_input(input),
                inference_params=optional_params,
            ),
        )

    def _calculate_usage(
        self,
        input: list[str] | CohereEmbeddingInputList,
        encoding: Any,
        meta: dict,
    ) -> Usage:
        text_tokens: Final[int | None] = meta.get("billed_units", {}).get("input_tokens")
        image_tokens: Final[int | None] = meta.get("billed_units", {}).get("images")
        fallback_texts: Final = tuple(
            text
            for item in input
            for text in (
                (item,)
                if isinstance(item, str)
                else tuple(content["text"] for content in item["content"] if content["type"] == "text")
            )
        )
        input_tokens: Final = (
            sum(len(encoding.encode(text)) for text in fallback_texts)
            if image_tokens is None and text_tokens is None
            else (image_tokens or 0) + (text_tokens or 0)
        )
        prompt_tokens_details: Final = (
            None
            if image_tokens is None and text_tokens is None
            else PromptTokensDetailsWrapper(
                image_tokens=image_tokens,
                text_tokens=text_tokens,
            )
        )

        return Usage(
            prompt_tokens=input_tokens,
            completion_tokens=0,
            total_tokens=input_tokens,
            prompt_tokens_details=prompt_tokens_details,
        )

    def _transform_response(
        self,
        response: httpx.Response,
        api_key: str | None,
        logging_obj: LiteLLMLoggingObj,
        data: dict | CohereEmbeddingRequest,
        model_response: EmbeddingResponse,
        model: str,
        encoding: Any,
        input: list[str] | CohereEmbeddingInputList,
    ) -> EmbeddingResponse:
        response_json: Final = response.json()
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
        embeddings: Final = response_json["embeddings"]
        output_data: Final = []
        for k, embedding_list in embeddings.items():
            for idx, embedding in enumerate(embedding_list):
                output_data.append({"object": "embedding", "index": idx, "embedding": embedding})
        model_response.object = "list"
        model_response.data = output_data
        model_response.model = model

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
        api_key: str | None,
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
            input=cast(  # cast-ok: logging preserves the original provider input without a precise static type
                list[str] | CohereEmbeddingInputList,
                logging_obj.model_call_details["input"],
            ),
        )

    def get_error_class(self, error_message: str, status_code: int, headers: dict | httpx.Headers) -> BaseLLMException:
        return CohereError(
            status_code=status_code,
            message=error_message,
        )
