"""
CometAPI Embedding API support - OpenAI compatible
"""

from typing import List, Optional, Union

import httpx

from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.embedding.transformation import BaseEmbeddingConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllEmbeddingInputValues, AllMessageValues
from litellm.types.utils import EmbeddingResponse, Usage

from ..common_utils import CometAPIException


class CometAPIEmbeddingConfig(BaseEmbeddingConfig):
    """
    Configuration class for CometAPI Embedding API.
    
    Since CometAPI is OpenAI-compatible, this class provides OpenAI-standard
    embedding functionality with CometAPI-specific authentication and endpoints.
    """

    def __init__(self) -> None:
        pass

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        """
        Get the complete URL for the CometAPI embedding endpoint.
        """
        api_base = (
            "https://api.cometapi.com/v1" if api_base is None else api_base.rstrip("/")
        )
        complete_url = f"{api_base}/embeddings"
        return complete_url

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
        """
        Validate and set up authentication headers for CometAPI.
        """
        if api_key is None:
            api_key = get_secret_str("COMETAPI_KEY")

        default_headers = {
            "Authorization": f"Bearer {api_key}",
            "accept": "application/json",
            "Content-Type": "application/json",
        }

        if "Authorization" in headers:
            default_headers["Authorization"] = headers["Authorization"]

        return {**default_headers, **headers}

    def get_supported_openai_params(self, model: str) -> List[str]:
        """
        Get the supported OpenAI parameters for embedding requests.
        CometAPI supports standard OpenAI embedding parameters.
        """
        return [
            "dimensions",
            "encoding_format",
            "user",
        ]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        """
        Map OpenAI parameters to CometAPI format.
        """
        supported_openai_params = self.get_supported_openai_params(model)
        for param, value in non_default_params.items():
            if param in supported_openai_params:
                optional_params[param] = value
        return optional_params

    def transform_embedding_request(
        self,
        model: str,
        input: AllEmbeddingInputValues,
        optional_params: dict,
        headers: dict,
    ) -> dict:
        """
        Transform the embedding request into CometAPI format.
        """
        return {"input": input, "model": model, **optional_params}

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
        """
        Transform CometAPI response into standard EmbeddingResponse format.
        """
        try:
            raw_response_json = raw_response.json()
        except Exception:
            raise CometAPIException(
                message=raw_response.text,
                status_code=raw_response.status_code,
                headers=raw_response.headers,
            )

        model_response.model = raw_response_json.get("model")
        model_response.data = raw_response_json.get("data")
        model_response.object = raw_response_json.get("object")

        usage = Usage(
            prompt_tokens=raw_response_json.get("usage", {}).get("prompt_tokens", 0),
            total_tokens=raw_response_json.get("usage", {}).get("total_tokens", 0),
        )

        model_response.usage = usage
        return model_response

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, httpx.Headers]
    ) -> BaseLLMException:
        """
        Get the appropriate error class for CometAPI exceptions.
        """
        return CometAPIException(
            message=error_message, status_code=status_code, headers=headers
        )
