from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import httpx

import litellm
from litellm.llms.azure.common_utils import BaseAzureLLM
from litellm.llms.base_llm.vector_store.transformation import BaseVectorStoreConfig
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


class AzureAIVectorStoreConfig(BaseVectorStoreConfig, BaseAzureLLM):
    """
    Configuration for Azure AI Search Vector Store

    This implementation uses the Azure AI Search API for vector store operations.
    Supports vector search with embeddings generated via litellm.embeddings.
    """

    def __init__(self):
        super().__init__()

    def get_vector_store_endpoints_by_type(self) -> VectorStoreIndexEndpoints:
        return {
            "read": [("GET", "/docs/search"), ("POST", "/docs/search")],
            "write": [("PUT", "/docs")],
        }

    def get_auth_credentials(
        self, litellm_params: dict
    ) -> BaseVectorStoreAuthCredentials:
        api_key = litellm_params.get("api_key")
        if api_key is None:
            raise ValueError("api_key is required")

        return {
            "headers": {
                "api-key": api_key,
            }
        }

    def validate_environment(
        self, headers: dict, litellm_params: Optional[GenericLiteLLMParams]
    ) -> dict:

        basic_headers = self._base_validate_azure_environment(headers, litellm_params)
        basic_headers.update({"Content-Type": "application/json"})
        return basic_headers

    def get_complete_url(
        self,
        api_base: Optional[str],
        litellm_params: dict,
    ) -> str:
        """
        Get the base endpoint for Azure AI Search API

        Expected format: https://{search_service_name}.search.windows.net
        """
        if api_base:
            return api_base.rstrip("/")

        # Get search service name from litellm_params
        search_service_name = litellm_params.get("azure_search_service_name")

        if not search_service_name:
            raise ValueError(
                "Azure AI Search service name is required. "
                "Provide it via litellm_params['azure_search_service_name'] or api_base parameter"
            )

        # Azure AI Search endpoint
        return f"https://{search_service_name}.search.windows.net"

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
                "embedding_model is required in litellm_params for Azure AI Search. "
                "Example: litellm_params['embedding_model'] = 'azure/text-embedding-3-large'"
            )

        embedding_config = litellm_params.get("litellm_embedding_config", {})
        if not embedding_config:
            raise ValueError(
                "embedding_config is required in litellm_params for Azure AI Search. "
                "Example: litellm_params['embedding_config'] = {'api_base': 'https://krris-mh44uf7y-eastus2.cognitiveservices.azure.com/', 'api_key': 'os.environ/AZURE_API_KEY', 'api_version': '2025-09-01'}"
            )

        # Get vector field name (defaults to contentVector)
        vector_field = litellm_params.get("azure_search_vector_field", "contentVector")

        # Get top_k (number of results to return)
        top_k = vector_store_search_optional_params.get("top_k", 10)

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
        url = f"{api_base}/indexes/{index_name}/docs/search?api-version=2024-07-01"

        # Build the request body for Azure AI Search with vector search
        request_body = {
            "search": "*",  # Get all documents (filtered by vector similarity)
            "vectorQueries": [
                {
                    "vector": query_vector,
                    "fields": vector_field,
                    "kind": "vector",
                    "k": top_k,  # Number of nearest neighbors to return
                }
            ],
            "select": "id,content",  # Fields to return (customize based on schema)
            "top": top_k,
        }

        #########################################################
        # Update logging object with details of the request
        #########################################################
        litellm_logging_obj.model_call_details["input"] = query
        litellm_logging_obj.model_call_details["embedding_model"] = embedding_model
        litellm_logging_obj.model_call_details["top_k"] = top_k

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
                    "@search.score": 0.95,
                    ... (other fields)
                }
            ]
        }
        """
        try:
            response_json = response.json()

            # Extract results from Azure AI Search API response
            results = response_json.get("value", [])

            # Transform results to standard format
            search_results: List[VectorStoreSearchResult] = []
            for result in results:
                # Extract document ID
                document_id = result.get("id", "")

                # Extract text content
                text_content = result.get("content", "")

                content = [
                    VectorStoreResultContent(
                        text=text_content,
                        type="text",
                    )
                ]

                # Get the search score (relevance score from Azure AI Search)
                score = result.get("@search.score", 0.0)

                # Use document ID as both file_id and filename
                file_id = document_id
                filename = f"Document {document_id}"

                # Build attributes with all available metadata
                # Exclude system fields and already-processed fields
                attributes = {}
                for key, value in result.items():
                    if key not in ["id", "content", "contentVector", "@search.score"]:
                        attributes[key] = value

                # Always include document_id in attributes
                attributes["document_id"] = document_id

                result_obj = VectorStoreSearchResult(
                    score=score,
                    content=content,
                    file_id=file_id,
                    filename=filename,
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
