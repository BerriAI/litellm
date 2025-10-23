from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import httpx

from litellm.llms.base_llm.vector_store.transformation import BaseVectorStoreConfig
from litellm.llms.vertex_ai.vertex_llm_base import VertexBase
from litellm.types.router import GenericLiteLLMParams
from litellm.types.vector_stores import (
    VectorStoreCreateOptionalRequestParams,
    VectorStoreCreateResponse,
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


class VertexSearchAPIVectorStoreConfig(BaseVectorStoreConfig, VertexBase):
    """
    Configuration for Vertex AI Search API Vector Store

    This implementation uses the Vertex AI Search API for vector store operations.
    """

    def __init__(self):
        super().__init__()

    def validate_environment(
        self, headers: dict, litellm_params: Optional[GenericLiteLLMParams]
    ) -> dict:
        """
        Validate and set up authentication for Vertex AI RAG API
        """
        litellm_params = litellm_params or GenericLiteLLMParams()

        # Get credentials and project info
        vertex_credentials = self.get_vertex_ai_credentials(dict(litellm_params))
        vertex_project = self.get_vertex_ai_project(dict(litellm_params))

        # Get access token using the base class method
        access_token, project_id = self._ensure_access_token(
            credentials=vertex_credentials,
            project_id=vertex_project,
            custom_llm_provider="vertex_ai",
        )

        headers.update(
            {
                "Authorization": f"Bearer {access_token}",
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
        Get the Base endpoint for Vertex AI Search API
        """
        vertex_location = self.get_vertex_ai_location(litellm_params)
        vertex_project = self.get_vertex_ai_project(litellm_params)
        engine_id = litellm_params.get("vector_store_id")
        collection_id = (
            litellm_params.get("vertex_collection_id") or "default_collection"
        )
        if api_base:
            return api_base.rstrip("/")

        # Vertex AI Search API endpoint for search
        return (
            f"https://discoveryengine.googleapis.com/v1/"
            f"projects/{vertex_project}/locations/{vertex_location}/"
            f"collections/{collection_id}/engines/{engine_id}/servingConfigs/default_config"
        )

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
        Transform search request for Vertex AI RAG API
        """
        # Convert query to string if it's a list
        if isinstance(query, list):
            query = " ".join(query)

        # Vertex AI RAG API endpoint for retrieving contexts
        url = f"{api_base}:search"

        # Construct full rag corpus path
        # Build the request body for Vertex AI Search API
        request_body = {"query": query, "pageSize": 10}

        #########################################################
        # Update logging object with details of the request
        #########################################################
        litellm_logging_obj.model_call_details["query"] = query

        return url, request_body

    def transform_search_vector_store_response(
        self, response: httpx.Response, litellm_logging_obj: LiteLLMLoggingObj
    ) -> VectorStoreSearchResponse:
        """
        Transform Vertex AI Search API response to standard vector store search response

        Handles the format from Discovery Engine Search API which returns:
        {
            "results": [
                {
                    "id": "...",
                    "document": {
                        "derivedStructData": {
                            "title": "...",
                            "link": "...",
                            "snippets": [...]
                        }
                    }
                }
            ]
        }
        """
        try:
            response_json = response.json()

            # Extract results from Vertex AI Search API response
            results = response_json.get("results", [])

            # Transform results to standard format
            search_results: List[VectorStoreSearchResult] = []
            for result in results:
                document = result.get("document", {})
                derived_data = document.get("derivedStructData", {})

                # Extract text content from snippets
                snippets = derived_data.get("snippets", [])
                text_content = ""

                if snippets:
                    # Combine all snippets into one text
                    text_parts = [
                        snippet.get("snippet", snippet.get("htmlSnippet", ""))
                        for snippet in snippets
                    ]
                    text_content = " ".join(text_parts)

                # If no snippets, use title as fallback
                if not text_content:
                    text_content = derived_data.get("title", "")

                content = [
                    VectorStoreResultContent(
                        text=text_content,
                        type="text",
                    )
                ]

                # Extract file/document information
                document_link = derived_data.get("link", "")
                document_title = derived_data.get("title", "")
                document_id = result.get("id", "")

                # Use link as file_id if available, otherwise use document ID
                file_id = document_link if document_link else document_id
                filename = document_title if document_title else "Unknown Document"

                # Build attributes with available metadata
                attributes = {
                    "document_id": document_id,
                }

                if document_link:
                    attributes["link"] = document_link
                if document_title:
                    attributes["title"] = document_title

                # Add display link if available
                display_link = derived_data.get("displayLink", "")
                if display_link:
                    attributes["displayLink"] = display_link

                # Add formatted URL if available
                formatted_url = derived_data.get("formattedUrl", "")
                if formatted_url:
                    attributes["formattedUrl"] = formatted_url

                # Note: Search API doesn't provide explicit scores in the response
                # You can use the position/rank as an implicit score
                score = 1.0 / (
                    float(search_results.__len__() + 1)
                )  # Decreasing score based on position

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
                search_query=litellm_logging_obj.model_call_details.get("query", ""),
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
