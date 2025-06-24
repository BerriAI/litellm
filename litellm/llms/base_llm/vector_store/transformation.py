from abc import abstractmethod
from typing import Optional

from litellm.types.router import GenericLiteLLMParams


class BaseVectorStoreTransformation:
    @abstractmethod
    def transform_query_vector_store_request(self):
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

