import re
from typing import List, Optional, Union
from urllib.parse import urljoin

import httpx
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.embedding.transformation import (
    BaseEmbeddingConfig,
    LiteLLMLoggingObj,
)
from litellm.llms.gigachat.common_utils import BaseGigaChat
from litellm.types.llms.openai import AllMessageValues, AllEmbeddingInputValues
from litellm.types.utils import EmbeddingResponse


class GigaChatEmbeddingConfig(BaseEmbeddingConfig, BaseGigaChat):
    """
    Transformations for gigachat /embeddings endpoint
    """

    def __init__(self):
        BaseEmbeddingConfig.__init__(self)
        BaseGigaChat.__init__(self)

    @property
    def custom_llm_provider(self) -> Optional[str]:
        return "gigachat"

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
        Build complete URL for embeddings endpoint.

        `api_base` is annotated as Optional[str] in the base interface, but this
        implementation requires a concrete string value. Raise an explicit
        error if it's missing so that both runtime and mypy are satisfied.
        """
        if api_base is None:
            raise ValueError("api_base must be provided for GigaChat embeddings")

        match = re.search(r"/v(\d+)/", api_base)
        if not match:
            api_base = urljoin(api_base, "v1/embeddings")
        return api_base

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
        Validate and prepare environment for GigaChat embedding API calls
        """
        import litellm

        if api_key is None:
            api_key = litellm.get_secret_str("GIGACHAT_API_KEY")

        # If no API key, try to get one via OAuth
        if api_key is None:
            api_key = self._get_oauth_token()

        if api_key is None:
            raise ValueError("GIGACHAT_API_KEY not found and OAuth credentials not provided")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "GigaChat-python-lib",
            **headers,
        }

        return headers

    def transform_embedding_request(
        self,
        model: str,
        input: AllEmbeddingInputValues,
        optional_params: dict,
        headers: dict,
    ) -> dict:
        """
        Transform OpenAI-style embedding request to GigaChat format
        """
        if isinstance(input, list) and (isinstance(input[0], list) or isinstance(input[0], int)):
            raise ValueError("Input must be a list of strings")
        request_body = {
            "model": model,
            "input": input,
        }

        return request_body

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
        Transform GigaChat embedding response to OpenAI format
        """
        try:
            response_json = raw_response.json()
        except Exception as e:
            raise ValueError(
                f"Failed to parse GigaChat embedding response as JSON: {raw_response.text}, Error: {str(e)}"
            )
        return EmbeddingResponse(**response_json)

    def get_supported_openai_params(self, model: str) -> list:
        """
        Get supported OpenAI parameters for GigaChat embeddings
        """
        return ["encoding_format"]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        """
        Map OpenAI parameters to GigaChat format
        """
        return {}

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, httpx.Headers]
    ) -> BaseLLMException:
        from ..common_utils import GigaChatError

        return GigaChatError(status_code=status_code, message=error_message, headers=headers)
