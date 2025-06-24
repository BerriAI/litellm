from abc import abstractmethod
from typing import Dict, Optional, Tuple

from litellm.types.router import GenericLiteLLMParams


class BaseVectorStoreConfig:
    @abstractmethod
    def transform_search_vector_store_request(
        self,
        vector_store_id: str,
        query: str,
        api_base: str,
    ) -> Tuple[str, Dict]:
        pass

    @abstractmethod
    def transform_query_vector_store_response(self):
        pass


    @abstractmethod
    def validate_environment(
        self, headers: dict, model: str, litellm_params: Optional[GenericLiteLLMParams]
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

