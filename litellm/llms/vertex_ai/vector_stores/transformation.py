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


class VertexVectorStoreConfig(BaseVectorStoreConfig, VertexBase):
    """
    Configuration for Vertex AI Vector Store RAG API
    
    This implementation uses the Vertex AI RAG Engine API for vector store operations.
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
        
        headers.update({
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        })
        
        return headers

    def get_complete_url(
        self,
        api_base: Optional[str],
        litellm_params: dict,
    ) -> str:
        """
        Get the Base endpoint for Vertex AI RAG API
        """
        vertex_location = self.get_vertex_ai_location(litellm_params)
        vertex_project = self.get_vertex_ai_project(litellm_params)
        
        if api_base:
            return api_base.rstrip("/")
        
        # Vertex AI RAG API endpoint for retrieveContexts
        return f"https://{vertex_location}-aiplatform.googleapis.com/v1/projects/{vertex_project}/locations/{vertex_location}"

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
        url = f"{api_base}:retrieveContexts"
        
        # Use helper methods to get project and location, then construct full rag corpus path
        vertex_project = self.get_vertex_ai_project(litellm_params)
        vertex_location = self.get_vertex_ai_location(litellm_params)
        
        # Construct full rag corpus path
        full_rag_corpus = f"projects/{vertex_project}/locations/{vertex_location}/ragCorpora/{vector_store_id}"
        
        # Build the request body for Vertex AI RAG API
        request_body: Dict[str, Any] = {
            "vertex_rag_store": {
                "rag_resources": [
                    {
                        "rag_corpus": full_rag_corpus
                    }
                ]
            },
            "query": {
                "text": query
            }
        }

        #########################################################
        # Update logging object with details of the request
        #########################################################
        litellm_logging_obj.model_call_details["query"] = query
        
        # Add optional parameters
        max_num_results = vector_store_search_optional_params.get("max_num_results")
        if max_num_results is not None:
            request_body["query"]["rag_retrieval_config"] = {
                "top_k": max_num_results
            }
        
        # Add filters if provided
        filters = vector_store_search_optional_params.get("filters")
        if filters is not None:
            if "rag_retrieval_config" not in request_body["query"]:
                request_body["query"]["rag_retrieval_config"] = {}
            request_body["query"]["rag_retrieval_config"]["filter"] = filters
        
        # Add ranking options if provided
        ranking_options = vector_store_search_optional_params.get("ranking_options")
        if ranking_options is not None:
            if "rag_retrieval_config" not in request_body["query"]:
                request_body["query"]["rag_retrieval_config"] = {}
            request_body["query"]["rag_retrieval_config"]["ranking"] = ranking_options
        
        return url, request_body

    def transform_search_vector_store_response(self, response: httpx.Response, litellm_logging_obj: LiteLLMLoggingObj) -> VectorStoreSearchResponse:
        """
        Transform Vertex AI RAG API response to standard vector store search response
        """
        try:

            response_json = response.json()
            # Extract contexts from Vertex AI response - handle nested structure
            contexts = response_json.get("contexts", {}).get("contexts", [])
            
            # Transform contexts to standard format
            search_results = []
            for context in contexts:
                content = [
                    VectorStoreResultContent(
                        text=context.get("text", ""),
                        type="text",
                    )
                ]
                
                # Extract file information
                source_uri = context.get("sourceUri", "")
                source_display_name = context.get("sourceDisplayName", "")
                
                # Generate file_id from source URI or use display name as fallback
                file_id = source_uri if source_uri else source_display_name
                filename = source_display_name if source_display_name else "Unknown Document"
                
                # Build attributes with available metadata
                attributes = {}
                if source_uri:
                    attributes["sourceUri"] = source_uri
                if source_display_name:
                    attributes["sourceDisplayName"] = source_display_name
                
                # Add page span information if available
                page_span = context.get("pageSpan", {})
                if page_span:
                    attributes["pageSpan"] = page_span
                
                result = VectorStoreSearchResult(
                    score=context.get("score", 0.0),
                    content=content,
                    file_id=file_id,
                    filename=filename,
                    attributes=attributes,
                )
                search_results.append(result)
            
            return VectorStoreSearchResponse(
                object="vector_store.search_results.page",
                search_query=litellm_logging_obj.model_call_details.get("query", ""),
                data=search_results
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
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Transform create request for Vertex AI RAG Corpus
        """
        url = f"{api_base}/ragCorpora"  # Base URL for creating RAG corpus
        
        # Build the request body for Vertex AI RAG Corpus creation
        request_body: Dict[str, Any] = {
            "display_name": vector_store_create_optional_params.get("name", "litellm-vector-store"),
            "description": "Vector store created via LiteLLM"
        }
        
        # Add metadata if provided
        metadata = vector_store_create_optional_params.get("metadata")
        if metadata is not None:
            request_body["labels"] = metadata
        
        return url, request_body

    def transform_create_vector_store_response(self, response: httpx.Response) -> VectorStoreCreateResponse:
        """
        Transform Vertex AI RAG Corpus creation response to standard vector store response
        """
        try:
            response_json = response.json()
            
            # Extract the corpus ID from the response name
            corpus_name = response_json.get("name", "")
            corpus_id = corpus_name.split("/")[-1] if "/" in corpus_name else corpus_name
            
            # Handle createTime conversion
            create_time = response_json.get("createTime", 0)
            if isinstance(create_time, str):
                # Convert ISO timestamp to Unix timestamp
                from datetime import datetime
                try:
                    dt = datetime.fromisoformat(create_time.replace('Z', '+00:00'))
                    create_time = int(dt.timestamp())
                except ValueError:
                    create_time = 0
            elif not isinstance(create_time, int):
                create_time = 0
            
            # Handle labels safely
            labels = response_json.get("labels", {})
            metadata = labels if isinstance(labels, dict) else {}
            
            return VectorStoreCreateResponse(
                id=corpus_id,
                object="vector_store",
                created_at=create_time,
                name=response_json.get("display_name", ""),
                bytes=0,  # Vertex AI doesn't provide byte count in the same way
                file_counts={
                    "in_progress": 0,
                    "completed": 0,
                    "failed": 0,
                    "cancelled": 0,
                    "total": 0
                },
                status="completed",  # Vertex AI corpus creation is typically synchronous
                expires_after=None,
                expires_at=None,
                last_active_at=None,
                metadata=metadata
            )
            
        except Exception as e:
            raise self.get_error_class(
                error_message=str(e),
                status_code=response.status_code,
                headers=response.headers
            ) 