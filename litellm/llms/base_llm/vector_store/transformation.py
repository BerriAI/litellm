from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import httpx

from litellm.types.router import GenericLiteLLMParams
from litellm.types.vector_stores import (
    BaseVectorStoreAuthCredentials,
    VECTOR_STORE_OPENAI_PARAMS,
    VectorStoreCreateOptionalRequestParams,
    VectorStoreCreateResponse,
    VectorStoreIndexEndpoints,
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

    def get_supported_openai_params(
        self, model: str
    ) -> List[VECTOR_STORE_OPENAI_PARAMS]:
        return []

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        drop_params: bool,
    ) -> dict:
        return optional_params

    @abstractmethod
    def get_auth_credentials(
        self, litellm_params: dict
    ) -> BaseVectorStoreAuthCredentials:
        pass

    @abstractmethod
    def get_vector_store_endpoints_by_type(self) -> VectorStoreIndexEndpoints:
        pass

    @abstractmethod
    def transform_search_vector_store_request(
        self,
        vector_store_id: str,
        query: Union[str, List[str]],
        vector_store_search_optional_params: VectorStoreSearchOptionalRequestParams,
        api_base: str,
        litellm_logging_obj: LiteLLMLoggingObj,
        litellm_params: dict,
    ) -> Tuple[str, Dict]:

        pass

    @abstractmethod
    def transform_search_vector_store_response(
        self, response: httpx.Response, litellm_logging_obj: LiteLLMLoggingObj
    ) -> VectorStoreSearchResponse:
        pass

    @abstractmethod
    def transform_create_vector_store_request(
        self,
        vector_store_create_optional_params: VectorStoreCreateOptionalRequestParams,
        api_base: str,
    ) -> Tuple[str, Dict]:
        pass

    @abstractmethod
    def transform_create_vector_store_response(
        self, response: httpx.Response
    ) -> VectorStoreCreateResponse:
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

    def sign_request(
        self,
        headers: dict,
        optional_params: Dict,
        request_data: Dict,
        api_base: str,
        api_key: Optional[str] = None,
    ) -> Tuple[dict, Optional[bytes]]:
        """Optionally sign or modify the request before sending.

        Providers like AWS Bedrock require SigV4 signing. Providers that don't
        require any signing can simply return the headers unchanged and ``None``
        for the signed body.
        """
        return headers, None

    def calculate_vector_store_cost(
        self,
        response: VectorStoreSearchResponse,
    ) -> Tuple[float, float]:
        return 0.0, 0.0
