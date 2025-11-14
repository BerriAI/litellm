"""
Translates from Cohere's `/v1/rerank` input format to Vertex AI Discovery Engine's `/rank` input format.

Why separate file? Make it easy to see how transformation works
"""

from typing import Any, Dict, List, Optional, Union

import httpx

import litellm
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.base_llm.rerank.transformation import BaseRerankConfig
from litellm.llms.vertex_ai.vertex_llm_base import VertexBase
from litellm.secret_managers.main import get_secret_str
from litellm.types.rerank import RerankResponse, RerankResponseMeta, RerankBilledUnits, RerankResponseResult



class VertexAIRerankConfig(BaseRerankConfig, VertexBase):
    """
    Configuration for Vertex AI Discovery Engine Rerank API
    
    Reference: https://cloud.google.com/generative-ai-app-builder/docs/ranking#rank_or_rerank_a_set_of_records_according_to_a_query
    """

    def __init__(self) -> None:
        super().__init__()

    def get_complete_url(
        self, 
        api_base: Optional[str], 
        model: str,
        optional_params: Optional[Dict] = None,
    ) -> str:
        """
        Get the complete URL for the Vertex AI Discovery Engine ranking API
        """
        # Try to get project ID from optional_params first (e.g., vertex_project parameter)
        params = optional_params or {}
        
        # Get credentials to extract project ID if needed
        vertex_credentials = self.safe_get_vertex_ai_credentials(params.copy())
        vertex_project = self.safe_get_vertex_ai_project(params.copy())
        
        # Use _ensure_access_token to extract project_id from credentials
        # This is the same method used in vertex embeddings
        _, vertex_project = self._ensure_access_token(
            credentials=vertex_credentials,
            project_id=vertex_project,
            custom_llm_provider="vertex_ai",
        )
        
        # Fallback to environment or litellm config
        project_id = (
            vertex_project
            or get_secret_str("VERTEXAI_PROJECT") 
            or litellm.vertex_project
        )
        
        if not project_id:
            raise ValueError(
                "Vertex AI project ID is required. Please set 'VERTEXAI_PROJECT', 'litellm.vertex_project', or pass 'vertex_project' parameter"
            )
        
        return f"https://discoveryengine.googleapis.com/v1/projects/{project_id}/locations/global/rankingConfigs/default_ranking_config:rank"

    def validate_environment(
        self,
        headers: dict,
        model: str,
        api_key: Optional[str] = None,
        optional_params: Optional[Dict] = None,
    ) -> dict:
        """
        Validate and set up authentication for Vertex AI Discovery Engine API
        """
        # Get credentials and project info from optional_params (which contains vertex_credentials, etc.)
        litellm_params = optional_params.copy() if optional_params else {}
        vertex_credentials = self.safe_get_vertex_ai_credentials(litellm_params)
        vertex_project = self.safe_get_vertex_ai_project(litellm_params)
        
        # Get access token using the base class method
        access_token, project_id = self._ensure_access_token(
            credentials=vertex_credentials,
            project_id=vertex_project,
            custom_llm_provider="vertex_ai",
        )
      
        default_headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
            "X-Goog-User-Project": project_id,
        }

        # If 'Authorization' is provided in headers, it overrides the default.
        if "Authorization" in headers:
            default_headers["Authorization"] = headers["Authorization"]

        # Merge other headers, overriding any default ones except Authorization
        return {**default_headers, **headers}

    def transform_rerank_request(
        self,
        model: str,
        optional_rerank_params: Dict,
        headers: dict,
    ) -> dict:
        """
        Transform the request from Cohere format to Vertex AI Discovery Engine format
        """
        if "query" not in optional_rerank_params:
            raise ValueError("query is required for Vertex AI rerank")
        if "documents" not in optional_rerank_params:
            raise ValueError("documents is required for Vertex AI rerank")
        
        query = optional_rerank_params["query"]
        documents = optional_rerank_params["documents"]
        top_n = optional_rerank_params.get("top_n", None)
        return_documents = optional_rerank_params.get("return_documents", True)
        
        # Convert documents to records format
        records = []
        for idx, document in enumerate(documents):
            if isinstance(document, str):
                content = document
                title = " ".join(document.split()[:3])  # First 3 words as title
            else:
                # Handle dict format
                content = document.get("text", str(document))
                title = document.get("title", " ".join(content.split()[:3]))
            
            records.append({
                "id": str(idx),
                "title": title,
                "content": content
            })
        
        request_data = {
            "model": model,
            "query": query,
            "records": records
        }
        
        if top_n is not None:
            request_data["topN"] = top_n
        
        # Map return_documents to ignoreRecordDetailsInResponse
        # When return_documents is False, we want to ignore record details (return only IDs)
        request_data["ignoreRecordDetailsInResponse"] = not return_documents
        
        return request_data

    def transform_rerank_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: RerankResponse,
        logging_obj: LiteLLMLoggingObj,
        api_key: Optional[str] = None,
        request_data: dict = {},
        optional_params: dict = {},
        litellm_params: dict = {},
    ) -> RerankResponse:
        """
        Transform Vertex AI Discovery Engine response to Cohere format
        """
        try:
            raw_response_json = raw_response.json()
        except Exception as e:
            raise ValueError(f"Failed to parse response: {e}")

        # Extract records from response
        records = raw_response_json.get("records", [])
        
        # Convert to Cohere format
        results = []
        for record in records:
            # Handle both cases: with full details and with only IDs
            if "score" in record:
                # Full response with score and details
                results.append({
                    "index": int(record["id"]), 
                    "relevance_score": record.get("score", 0.0)
                })
            else:
                # Response with only IDs (when ignoreRecordDetailsInResponse=true)
                # We can't provide a relevance score, so we'll use a default
                results.append({
                    "index": int(record["id"]), 
                    "relevance_score": 1.0  # Default score when details are ignored
                })
        
        # Sort by relevance score (descending)
        results.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        # Create response in Cohere format        
        # Convert results to proper RerankResponseResult objects
        rerank_results = []
        for result in results:
            rerank_results.append(RerankResponseResult(
                index=result["index"],
                relevance_score=result["relevance_score"]
            ))
        
        # Create meta object
        meta = RerankResponseMeta(
            billed_units=RerankBilledUnits(
                search_units=len(records)
            )
        )
        
        return RerankResponse(
            id=f"vertex_ai_rerank_{model}",
            results=rerank_results,
            meta=meta
        )

    def get_supported_cohere_rerank_params(self, model: str) -> list:
        return [
            "query",
            "documents", 
            "top_n",
            "return_documents",
        ]

    def map_cohere_rerank_params(
        self,
        non_default_params: dict,
        model: str,
        drop_params: bool,
        query: str,
        documents: List[Union[str, Dict[str, Any]]],
        custom_llm_provider: Optional[str] = None,
        top_n: Optional[int] = None,
        rank_fields: Optional[List[str]] = None,
        return_documents: Optional[bool] = True,
        max_chunks_per_doc: Optional[int] = None,
        max_tokens_per_doc: Optional[int] = None,
    ) -> Dict:
        """
        Map Cohere rerank params to Vertex AI format
        """
        result = {
            "query": query,
            "documents": documents,
            "top_n": top_n,
            "return_documents": return_documents,
        }
        result.update(non_default_params)
        return result

