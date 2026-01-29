
import json
from typing import TYPE_CHECKING, Any, Optional, Union

import httpx

from litellm.llms.base_llm.embedding.transformation import BaseEmbeddingConfig
from litellm.types.llms.openai import AllEmbeddingInputValues
from litellm.types.utils import EmbeddingResponse

from ..common_utils import NLPCloudError

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


class NLPCloudEmbeddingConfig(BaseEmbeddingConfig):
    def get_supported_openai_params(self, model: str) -> list:
        return []

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        return optional_params

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: Any,
        optional_params: dict,
        litellm_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> dict:
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
        }
        if api_key:
            headers["Authorization"] = f"Token {api_key}"
        return headers

    def transform_embedding_request(
        self,
        model: str,
        input: AllEmbeddingInputValues,
        optional_params: dict,
        headers: dict,
    ) -> dict:
        data = {
            "sentences": input,
            **optional_params,
        }
        return data

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
        ## LOGGING
        logging_obj.post_call(
            input=None,
            api_key=api_key,
            original_response=raw_response.text,
            additional_args={"complete_input_dict": request_data},
        )

        ## RESPONSE OBJECT
        try:
            completion_response = raw_response.json()
        except Exception:
            raise NLPCloudError(
                message=raw_response.text, status_code=raw_response.status_code
            )

        if "error" in completion_response:
            raise NLPCloudError(
                message=completion_response["error"],
                status_code=raw_response.status_code,
            )

        embeddings = completion_response.get("embeddings", [])
        output_data = []
        for idx, embedding_val in enumerate(embeddings):
            output_data.append(
                {
                    "object": "embedding",
                    "index": idx,
                    "embedding": embedding_val,
                }
            )

        model_response.data = output_data  # type: ignore
        model_response.model = model

        return model_response

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, httpx.Headers]
    ) -> NLPCloudError:
        return NLPCloudError(
            status_code=status_code, message=error_message, headers=headers
        )

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        if api_base is None:
            api_base = "https://api.nlpcloud.io/v1/"
        return api_base + model + "/embeddings"
