from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import httpx

import litellm
from litellm.llms.base_llm.vector_store.transformation import BaseVectorStoreConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.router import GenericLiteLLMParams
from litellm.types.vector_stores import (
    BaseVectorStoreAuthCredentials,
    VectorStoreCreateOptionalRequestParams,
    VectorStoreCreateResponse,
    VectorStoreIndexEndpoints,
    VectorStoreResultContent,
    VectorStoreSearchOptionalRequestParams,
    VectorStoreSearchResponse,
    VectorStoreSearchResult,
)

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any

MILVUS_OPTIONAL_PARAMS = {
    "dbName",
    "annsField",
    "limit",
    "filter",
    "offset",
    "groupingField",
    "outputFields",
    "searchParams",
    "partitionNames",
    "consistencyLevel",
}


class MilvusVectorStoreConfig(BaseVectorStoreConfig):
    """
    Configuration for Milvus Vector Store

    This implementation uses the Azure AI Search API for vector store operations.
    Supports vector search with embeddings generated via litellm.embeddings.
    """

    def __init__(self):
        super().__init__()

    def validate_environment(
        self, headers: dict, litellm_params: Optional[GenericLiteLLMParams]
    ) -> dict:
        api_key: Optional[str] = None
        if litellm_params is not None:
            api_key = litellm_params.api_key or get_secret_str("MILVUS_API_KEY")

        if not api_key:
            raise ValueError(
                "MILVUS_API_KEY is not set. Either set it in the litellm_params or set the MILVUS_API_KEY environment variable."
            )

        headers.update({"Authorization": f"Bearer {api_key}"})

        return headers

    def get_auth_credentials(
        self, litellm_params: dict
    ) -> BaseVectorStoreAuthCredentials:
        api_key = litellm_params.get("api_key")
        if not api_key:
            raise ValueError(
                "MILVUS_API_KEY is not set. Either set it in the litellm_params or set the MILVUS_API_KEY environment variable."
            )
        return {
            "headers": {
                "Authorization": f"Bearer {api_key}",
            },
        }

    def get_vector_store_endpoints_by_type(self) -> VectorStoreIndexEndpoints:
        return {
            "read": [
                ("POST", "/v2/vectordb/entities/search"),
                ("POST", "/v2/vectordb/entities/get"),
                ("POST", "/v2/vectordb/entities/query"),
            ],
            "write": [
                ("POST", "/v2/vectordb/entities/upsert"),
                ("POST", "/v2/vectordb/entities/insert"),
            ],
        }

    def map_openai_params(
        self, non_default_params: dict, optional_params: dict, drop_params: bool
    ) -> dict:
        for param, value in non_default_params.items():
            if param in MILVUS_OPTIONAL_PARAMS:
                optional_params[param] = value
        return optional_params

    def get_complete_url(
        self,
        api_base: Optional[str],
        litellm_params: dict,
    ) -> str:
        """
        Get the base endpoint for Milvus API

        Expected format: https://{milvus_api_base}.milvus.io
        """
        api_base = api_base or get_secret_str("MILVUS_API_BASE")

        if not api_base:
            raise ValueError(
                "Milvus API base URL is required. Set MILVUS_API_BASE environment variable or pass api_base in litellm_params."
            )

        if api_base:
            return api_base.rstrip("/")

        return api_base

    def transform_search_vector_store_request(
        self,
        vector_store_id: str,
        query: Union[str, List[str]],
        vector_store_search_optional_params: VectorStoreSearchOptionalRequestParams,
        api_base: str,
        litellm_logging_obj: LiteLLMLoggingObj,
        litellm_params: dict,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Transform search request for Azure AI Search API

        Generates embeddings using litellm.embeddings and constructs Azure AI Search request
        """
        # Convert query to string if it's a list
        if isinstance(query, list):
            query = " ".join(query)

        # Get embedding model from litellm_params (required)
        embedding_model = litellm_params.get("litellm_embedding_model")
        if not embedding_model:
            raise ValueError(
                "embedding_model is required in litellm_params for Milvus. You can call any litellm embedding model."
                "Example: litellm_params['embedding_model'] = 'azure/text-embedding-3-large'"
            )

        embedding_config = litellm_params.get("litellm_embedding_config", {})
        if not embedding_config:
            raise ValueError(
                "embedding_config is required in litellm_params for Milvus. You can call any litellm embedding model."
                "Example: litellm_params['embedding_config'] = {'api_base': 'https://krris-mh44uf7y-eastus2.cognitiveservices.azure.com/', 'api_key': 'os.environ/AZURE_API_KEY', 'api_version': '2025-09-01'}"
            )

        # Get top_k (number of results to return)
        # Generate embedding for the query using litellm.embeddings
        try:
            embedding_response = litellm.embedding(
                model=embedding_model,
                input=[query],
                **embedding_config,
            )
            query_vector = embedding_response.data[0]["embedding"]
        except Exception as e:
            raise Exception(f"Failed to generate embedding for query: {str(e)}")

        # Azure AI Search endpoint for search
        index_name = vector_store_id  # vector_store_id is the index name
        url = f"{api_base}/v2/vectordb/entities/search"

        # Build the request body for Azure AI Search with vector search
        request_body = {
            "collectionName": index_name,
            "data": [query_vector],
            "annsField": "book_intro_vector",
            **vector_store_search_optional_params,
        }

        #########################################################
        # Update logging object with details of the request
        #########################################################
        litellm_logging_obj.model_call_details["input"] = query
        litellm_logging_obj.model_call_details["embedding_model"] = embedding_model

        return url, request_body

    def transform_search_vector_store_response(
        self, response: httpx.Response, litellm_logging_obj: LiteLLMLoggingObj
    ) -> VectorStoreSearchResponse:
        """
        Transform Azure AI Search API response to standard vector store search response

        Handles the format from Azure AI Search which returns:
        {
            "value": [
                {
                    "id": "...",
                    "content": "...",
                    "distance": 0.95,
                }
            ]
        }
        """
        try:
            response_json = response.json()

            # Extract results from Azure AI Search API response
            results = response_json.get("data", [])

            # Try to get text_field from optional_params first, then litellm_params
            optional_params = litellm_logging_obj.model_call_details.get(
                "optional_params", {}
            )
            text_field = optional_params.get("milvus_text_field", "")

            # Fallback to litellm_params if not in optional_params

            if not text_field:
                text_field = litellm_logging_obj.model_call_details.get(
                    "litellm_params", {}
                ).get("milvus_text_field", "")

            # Transform results to standard format
            search_results: List[VectorStoreSearchResult] = []
            for result in results:
                # Extract text content
                text_content = result.get(text_field, "")

                content = [
                    VectorStoreResultContent(
                        text=text_content,
                        type="text",
                    )
                ]

                # Get the search score (distance from the query vector)
                score = result.get("distance", 0.0)

                # Build attributes with all available metadata
                # Exclude system fields and already-processed fields
                attributes = {}
                for key, value in result.items():
                    if key not in ["id", "content", "distance", text_field]:
                        attributes[key] = value

                result_obj = VectorStoreSearchResult(
                    score=score,
                    content=content,
                    file_id=None,
                    filename=None,
                    attributes=attributes,
                )
                search_results.append(result_obj)

            return VectorStoreSearchResponse(
                object="vector_store.search_results.page",
                search_query=litellm_logging_obj.model_call_details.get("input", ""),
                data=search_results,
            )

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
    ) -> Tuple[str, Dict]:
        raise NotImplementedError

    def transform_create_vector_store_response(
        self, response: httpx.Response
    ) -> VectorStoreCreateResponse:
        raise NotImplementedError
