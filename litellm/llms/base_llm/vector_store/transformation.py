from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import httpx

from litellm.types.router import GenericLiteLLMParams
from litellm.types.vector_stores import (
    VectorStoreCreateOptionalRequestParams,
    VectorStoreCreateResponse,
    VectorStoreSearchOptionalRequestParams,
    VectorStoreSearchResponse,
)

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    from ..chat.transformation import BaseLLMException as _BaseLLMException

    LiteLLMLoggingObj = _LiteLLMLoggingObj
    BaseLLMException = _BaseLLMException
else:
    LiteLLMLoggingObj = Any
    BaseLLMException = Any

class BaseVectorStoreConfig:
    @abstractmethod
    def transform_search_vector_store_request(
        self,
        vector_store_id: str,
        query: Union[str, List[str]],
        vector_store_search_optional_params: VectorStoreSearchOptionalRequestParams,
        api_base: str,
    ) -> Tuple[str, Dict]:
        pass

    @abstractmethod
    def transform_search_vector_store_response(self, response: httpx.Response) -> VectorStoreSearchResponse:
        pass

    @abstractmethod
    def transform_create_vector_store_request(
        self,
        vector_store_create_optional_params: VectorStoreCreateOptionalRequestParams,
        api_base: str,
    ) -> Tuple[str, Dict]:
        pass

    @abstractmethod
    def transform_create_vector_store_response(self, response: httpx.Response) -> VectorStoreCreateResponse:
        pass

    @abstractmethod
    def validate_environment(
        self, headers: dict, litellm_params: Optional[GenericLiteLLMParams]
    ) -> dict:
        return {}

    @abstractmethod
    def get_complete_url(
        self,
        api_base: Optional[str],
        litellm_params: dict,
    ) -> str:
        """
        OPTIONAL

        Get the complete url for the request

        Some providers need `model` in `api_base`
        """
        if api_base is None:
            raise ValueError("api_base is required")
        return api_base
    

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, httpx.Headers]
    ) -> BaseLLMException:
        from ..chat.transformation import BaseLLMException

        raise BaseLLMException(
            status_code=status_code,
            message=error_message,
            headers=headers,
        )

