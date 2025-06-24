from abc import abstractmethod
from typing import Dict, List, Optional, Tuple, Union

from litellm.types.router import GenericLiteLLMParams
from litellm.types.vector_stores import (
    VectorStoreSearchOptionalRequestParams,
    VectorStoreSearchResponse,
)


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
    def transform_search_vector_store_response(self) -> VectorStoreSearchResponse:
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

