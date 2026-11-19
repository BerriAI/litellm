from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import httpx

from litellm.types.rerank import RerankResponse

from ..chat.transformation import BaseLLMException

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


class BaseRerankConfig(ABC):
    @abstractmethod
    def validate_environment(
        self,
        headers: dict,
        model: str,
        api_key: str | None = None,
        optional_params: dict | None = None,
        litellm_params: Mapping[str, object] | None = None,
    ) -> dict:
        pass

    @abstractmethod
    def transform_rerank_request(
        self,
        model: str,
        optional_rerank_params: dict,
        headers: dict,
        litellm_params: dict | None = None,
    ) -> dict:
        return {}

    @abstractmethod
    def transform_rerank_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: RerankResponse,
        logging_obj: LiteLLMLoggingObj,
        api_key: str | None = None,
        request_data: dict = {},
        optional_params: dict = {},
        litellm_params: dict = {},
    ) -> RerankResponse:
        return model_response

    @abstractmethod
    def get_complete_url(
        self,
        api_base: str | None,
        model: str,
        optional_params: dict | None = None,
    ) -> str:
        """
        OPTIONAL

        Get the complete url for the request

        Some providers need `model` in `api_base`
        """
        return api_base or ""

    @abstractmethod
    def get_supported_cohere_rerank_params(self, model: str) -> list:
        pass

    @abstractmethod
    def map_cohere_rerank_params(
        self,
        non_default_params: dict,
        model: str,
        drop_params: bool,
        query: str,
        documents: list[str | dict[str, object]],
        custom_llm_provider: str | None = None,
        top_n: int | None = None,
        rank_fields: list[str] | None = None,
        return_documents: bool | None = True,
        max_chunks_per_doc: int | None = None,
        max_tokens_per_doc: int | None = None,
        instruction: str | None = None,
    ) -> dict:
        pass

    def get_error_class(self, error_message: str, status_code: int, headers: dict | httpx.Headers) -> BaseLLMException:
        raise BaseLLMException(
            status_code=status_code,
            message=error_message,
            headers=headers,
        )
