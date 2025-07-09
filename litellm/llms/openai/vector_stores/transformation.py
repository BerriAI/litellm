from typing import Dict, List, Optional, Tuple, Union, cast

import httpx

import litellm
from litellm.llms.base_llm.vector_store.transformation import BaseVectorStoreConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.router import GenericLiteLLMParams
from litellm.types.vector_stores import (
    VectorStoreCreateOptionalRequestParams,
    VectorStoreCreateRequest,
    VectorStoreCreateResponse,
    VectorStoreSearchOptionalRequestParams,
    VectorStoreSearchRequest,
    VectorStoreSearchResponse,
)


class OpenAIVectorStoreConfig(BaseVectorStoreConfig):
    ASSISTANTS_HEADER_KEY = "OpenAI-Beta"
    ASSISTANTS_HEADER_VALUE = "assistants=v2"

    def validate_environment(
        self, headers: dict, litellm_params: Optional[GenericLiteLLMParams]
    ) -> dict:
        litellm_params = litellm_params or GenericLiteLLMParams()
        api_key = (
            litellm_params.api_key
            or litellm.api_key
            or litellm.openai_key
            or get_secret_str("OPENAI_API_KEY")
        )
        headers.update(
            {
                "Authorization": f"Bearer {api_key}",
            }
        )

        #########################################################
        # Ensure OpenAI Assistants header is includes
        #########################################################
        if self.ASSISTANTS_HEADER_KEY not in headers:
            headers.update(
                {
                    self.ASSISTANTS_HEADER_KEY: self.ASSISTANTS_HEADER_VALUE,
            }
        )

        return headers

    def get_complete_url(
        self,
        api_base: Optional[str],
        litellm_params: dict,
    ) -> str:
        """
        Get the Base endpoint for OpenAI Vector Stores API
        """
        api_base = (
            api_base
            or litellm.api_base
            or get_secret_str("OPENAI_BASE_URL")
            or get_secret_str("OPENAI_API_BASE")
            or "https://api.openai.com/v1"
        )

        # Remove trailing slashes
        api_base = api_base.rstrip("/")

        return f"{api_base}/vector_stores"
    

    def transform_search_vector_store_request(
        self,
        vector_store_id: str,
        query: Union[str, List[str]],
        vector_store_search_optional_params: VectorStoreSearchOptionalRequestParams,
        api_base: str,
    ) -> Tuple[str, Dict]:
        url = f"{api_base}/{vector_store_id}/search"
        typed_request_body = VectorStoreSearchRequest(
            query=query,
            filters=vector_store_search_optional_params.get("filters", None),
            max_num_results=vector_store_search_optional_params.get("max_num_results", None),
            ranking_options=vector_store_search_optional_params.get("ranking_options", None),
            rewrite_query=vector_store_search_optional_params.get("rewrite_query", None),
        )

        dict_request_body = cast(dict, typed_request_body)
        return url, dict_request_body
    


    def transform_search_vector_store_response(self, response: httpx.Response) -> VectorStoreSearchResponse:
        try:
            response_json = response.json()
            return VectorStoreSearchResponse(
                **response_json
            )
        except Exception as e:
            raise self.get_error_class(
                error_message=str(e), 
                status_code=response.status_code, 
                headers=response.headers
            )

    def transform_create_vector_store_request(
        self,
        vector_store_create_optional_params: VectorStoreCreateOptionalRequestParams,
        api_base: str,
    ) -> Tuple[str, Dict]:
        url = api_base  # Base URL for creating vector stores
        typed_request_body = VectorStoreCreateRequest(
            name=vector_store_create_optional_params.get("name", None),
            file_ids=vector_store_create_optional_params.get("file_ids", None),
            expires_after=vector_store_create_optional_params.get("expires_after", None),
            chunking_strategy=vector_store_create_optional_params.get("chunking_strategy", None),
            metadata=vector_store_create_optional_params.get("metadata", None),
        )

        dict_request_body = cast(dict, typed_request_body)
        return url, dict_request_body

    def transform_create_vector_store_response(self, response: httpx.Response) -> VectorStoreCreateResponse:
        try:
            response_json = response.json()
            return VectorStoreCreateResponse(
                **response_json
            )
        except Exception as e:
            raise self.get_error_class(
                error_message=str(e), 
                status_code=response.status_code, 
                headers=response.headers
            )

    

    
    