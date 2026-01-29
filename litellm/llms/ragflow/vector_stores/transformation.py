from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import httpx

from litellm.llms.base_llm.vector_store.transformation import BaseVectorStoreConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.router import GenericLiteLLMParams
from litellm.types.vector_stores import (
    BaseVectorStoreAuthCredentials,
    VectorStoreCreateOptionalRequestParams,
    VectorStoreCreateResponse,
    VectorStoreFileCounts,
    VectorStoreIndexEndpoints,
    VectorStoreSearchOptionalRequestParams,
    VectorStoreSearchResponse,
)

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


class RAGFlowVectorStoreConfig(BaseVectorStoreConfig):
    """Vector store configuration for RAGFlow datasets."""

    def get_auth_credentials(
        self, litellm_params: dict
    ) -> BaseVectorStoreAuthCredentials:
        api_key = litellm_params.get("api_key")
        if api_key is None:
            # Try to get from environment variable
            api_key = get_secret_str("RAGFLOW_API_KEY")
        if api_key is None:
            raise ValueError("api_key is required (set RAGFLOW_API_KEY env var or pass in litellm_params)")
        return {
            "headers": {
                "Authorization": f"Bearer {api_key}",
            },
        }

    def get_vector_store_endpoints_by_type(self) -> VectorStoreIndexEndpoints:
        """RAGFlow vector stores are management-only, no search support."""
        return {
            "read": [],
            "write": [],
        }

    def validate_environment(
        self, headers: dict, litellm_params: Optional[GenericLiteLLMParams]
    ) -> dict:
        """Validate environment and set headers for RAGFlow API."""
        litellm_params = litellm_params or GenericLiteLLMParams()
        api_key = (
            litellm_params.api_key
            or get_secret_str("RAGFLOW_API_KEY")
        )
        
        if api_key is None:
            raise ValueError("RAGFLOW_API_KEY is required (set env var or pass in litellm_params)")
        
        headers.update(
            {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
        )
        return headers

    def get_complete_url(
        self,
        api_base: Optional[str],
        litellm_params: dict,
    ) -> str:
        """
        Get the complete URL for RAGFlow datasets API.
        
        Supports:
        - RAGFLOW_API_BASE env var
        - api_base in litellm_params
        - Default: http://localhost:9380
        """
        api_base = (
            api_base
            or litellm_params.get("api_base")
            or get_secret_str("RAGFLOW_API_BASE")
            or "http://localhost:9380"
        )

        # Remove trailing slashes
        api_base = api_base.rstrip("/")

        # RAGFlow datasets API endpoint
        return f"{api_base}/api/v1/datasets"

    def transform_search_vector_store_request(
        self,
        vector_store_id: str,
        query: Union[str, List[str]],
        vector_store_search_optional_params: VectorStoreSearchOptionalRequestParams,
        api_base: str,
        litellm_logging_obj: LiteLLMLoggingObj,
        litellm_params: dict,
    ) -> Tuple[str, Dict]:
        """RAGFlow vector stores are management-only, search is not supported."""
        raise NotImplementedError(
            "RAGFlow vector stores support dataset management only, not search/retrieval"
        )

    def transform_search_vector_store_response(
        self, response: httpx.Response, litellm_logging_obj: LiteLLMLoggingObj
    ) -> VectorStoreSearchResponse:
        """RAGFlow vector stores are management-only, search is not supported."""
        raise NotImplementedError(
            "RAGFlow vector stores support dataset management only, not search/retrieval"
        )

    def transform_create_vector_store_request(
        self,
        vector_store_create_optional_params: VectorStoreCreateOptionalRequestParams,
        api_base: str,
    ) -> Tuple[str, Dict]:
        """
        Transform create request to RAGFlow POST /api/v1/datasets format.
        
        Maps LiteLLM params to RAGFlow dataset creation parameters.
        RAGFlow-specific fields can be passed via metadata.
        """
        url = api_base  # Already includes /api/v1/datasets from get_complete_url
        
        # Extract name (required by RAGFlow)
        name = vector_store_create_optional_params.get("name")
        if not name:
            raise ValueError("name is required for RAGFlow dataset creation")
        
        # Build request body
        request_body: Dict[str, Any] = {
            "name": name,
        }
        
        # Extract RAGFlow-specific fields from metadata
        metadata = vector_store_create_optional_params.get("metadata")
        if metadata:
            # RAGFlow-specific fields that can be in metadata
            ragflow_fields = [
                "avatar",
                "description",
                "embedding_model",
                "permission",
                "chunk_method",
                "parser_config",
                "parse_type",
                "pipeline_id",
            ]
            
            for field in ragflow_fields:
                if field in metadata:
                    request_body[field] = metadata[field]
        
        # Validate: chunk_method and pipeline_id are mutually exclusive
        if "chunk_method" in request_body and "pipeline_id" in request_body:
            raise ValueError(
                "chunk_method and pipeline_id are mutually exclusive. "
                "Specify either chunk_method or pipeline_id, not both."
            )
        
        # If neither chunk_method nor pipeline_id is specified, default to naive
        if "chunk_method" not in request_body and "pipeline_id" not in request_body:
            request_body["chunk_method"] = "naive"
        
        return url, request_body

    def transform_create_vector_store_response(
        self, response: httpx.Response
    ) -> VectorStoreCreateResponse:
        """
        Transform RAGFlow response to VectorStoreCreateResponse format.
        
        RAGFlow response format:
        {
            "code": 0,
            "data": {
                "id": "...",
                "name": "...",
                "create_time": 1745836841611,  # milliseconds
                ...
            }
        }
        """
        try:
            response_json = response.json()
            
            # Check for RAGFlow error response
            if response_json.get("code") != 0:
                error_message = response_json.get("message", "Unknown error")
                raise self.get_error_class(
                    error_message=error_message,
                    status_code=response.status_code,
                    headers=response.headers,
                )
            
            data = response_json.get("data", {})
            
            # Extract dataset ID
            dataset_id = data.get("id")
            if not dataset_id:
                raise ValueError("RAGFlow response missing dataset id")
            
            # Extract name
            name = data.get("name")
            
            # Convert create_time from milliseconds to seconds (Unix timestamp)
            create_time_ms = data.get("create_time", 0)
            created_at = int(create_time_ms / 1000) if create_time_ms else None
            
            # Build VectorStoreCreateResponse
            return VectorStoreCreateResponse(
                id=dataset_id,
                object="vector_store",
                created_at=created_at or 0,
                name=name,
                bytes=0,  # RAGFlow doesn't provide bytes in response
                file_counts=VectorStoreFileCounts(
                    in_progress=0,
                    completed=0,
                    failed=0,
                    cancelled=0,
                    total=0,
                ),
                status="completed",
                expires_after=None,
                expires_at=None,
                last_active_at=None,
                metadata=None,
            )
        except Exception as e:
            # If it's already a ValueError we raised, re-raise it
            if isinstance(e, ValueError) and "RAGFlow response" in str(e):
                raise
            # If it's already our error class (has status_code), re-raise
            if hasattr(e, "status_code"):
                raise
            # Otherwise, wrap in our error class
            raise self.get_error_class(
                error_message=str(e),
                status_code=response.status_code,
                headers=response.headers,
            )

