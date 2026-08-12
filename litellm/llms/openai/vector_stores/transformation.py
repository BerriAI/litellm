from typing import TYPE_CHECKING, Any, Final, cast

import httpx

import litellm
from litellm.litellm_core_utils.url_utils import encode_url_path_segment
from litellm.llms.base_llm.vector_store.transformation import BaseVectorStoreConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.router import GenericLiteLLMParams
from litellm.types.vector_stores import (
    BaseVectorStoreAuthCredentials,
    VectorStoreCreateOptionalRequestParams,
    VectorStoreCreateRequest,
    VectorStoreCreateResponse,
    VectorStoreIndexEndpoints,
    VectorStoreSearchOptionalRequestParams,
    VectorStoreSearchRequest,
    VectorStoreSearchResponse,
)
from litellm.utils import add_openai_metadata

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


class OpenAIVectorStoreConfig(BaseVectorStoreConfig):
    ASSISTANTS_HEADER_KEY = "OpenAI-Beta"
    ASSISTANTS_HEADER_VALUE = "assistants=v2"

    def get_auth_credentials(self, litellm_params: dict) -> BaseVectorStoreAuthCredentials:
        api_key: Final = litellm_params.get("api_key")
        if api_key is None:
            raise ValueError("api_key is required")
        return {
            "headers": {
                "Authorization": f"Bearer {api_key}",
            },
        }

    def get_vector_store_endpoints_by_type(self) -> VectorStoreIndexEndpoints:
        return {
            "read": [("GET", "/vector_stores/{index_name}/search")],
            "write": [("POST", "/vector_stores")],
        }

    def validate_environment(self, headers: dict, litellm_params: GenericLiteLLMParams | None) -> dict:
        litellm_params = litellm_params or GenericLiteLLMParams()
        api_key = litellm_params.api_key or litellm.api_key or litellm.openai_key or get_secret_str("OPENAI_API_KEY")
        headers.update(
            {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
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
        api_base: str | None,
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
        query: str | list[str],
        vector_store_search_optional_params: VectorStoreSearchOptionalRequestParams,
        api_base: str,
        litellm_logging_obj: LiteLLMLoggingObj,
        litellm_params: dict,
        extra_body: dict[str, Any] | None = None,
    ) -> tuple[str, dict]:
        encoded_vector_store_id: Final = encode_url_path_segment(vector_store_id, field_name="vector_store_id")
        url: Final = f"{api_base}/{encoded_vector_store_id}/search"
        typed_request_body: Final = VectorStoreSearchRequest(
            query=query,
            filters=vector_store_search_optional_params.get("filters", None),
            max_num_results=vector_store_search_optional_params.get("max_num_results", None),
            ranking_options=vector_store_search_optional_params.get("ranking_options", None),
            rewrite_query=vector_store_search_optional_params.get("rewrite_query", None),
        )

        dict_request_body: Final = cast(dict, typed_request_body)
        return url, dict_request_body

    def transform_search_vector_store_response(
        self, response: httpx.Response, litellm_logging_obj: LiteLLMLoggingObj
    ) -> VectorStoreSearchResponse:
        try:
            response_json: Final = response.json()
            return VectorStoreSearchResponse(**response_json)
        except Exception as e:
            raise self.get_error_class(
                error_message=str(e),
                status_code=response.status_code,
                headers=response.headers,
            )

    def transform_create_vector_store_request(
        self,
        vector_store_create_optional_params: VectorStoreCreateOptionalRequestParams,
        api_base: str,
    ) -> tuple[str, dict]:
        url: Final = api_base  # Base URL for creating vector stores
        metadata: Final = vector_store_create_optional_params.get("metadata", None)
        metadata_payload: Final = add_openai_metadata(metadata)

        typed_request_body: Final = VectorStoreCreateRequest(
            name=vector_store_create_optional_params.get("name", None),
            file_ids=vector_store_create_optional_params.get("file_ids", None),
            expires_after=vector_store_create_optional_params.get("expires_after", None),
            chunking_strategy=vector_store_create_optional_params.get("chunking_strategy", None),
            metadata=metadata_payload,
        )

        dict_request_body: Final = cast(dict, typed_request_body)
        return url, dict_request_body

    def transform_create_vector_store_response(self, response: httpx.Response) -> VectorStoreCreateResponse:
        try:
            response_json: Final = response.json()
            return VectorStoreCreateResponse(**response_json)
        except Exception as e:
            raise self.get_error_class(
                error_message=str(e),
                status_code=response.status_code,
                headers=response.headers,
            )
