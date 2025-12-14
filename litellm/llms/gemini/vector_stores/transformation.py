"""
Gemini File Search Vector Store Transformation Layer.

Implements the transformation between LiteLLM's unified vector store API
and Google Gemini's File Search API.
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import httpx

from litellm.llms.base_llm.vector_store.transformation import BaseVectorStoreConfig
from litellm.llms.gemini.common_utils import (
    GeminiError,
    GeminiModelInfo,
    get_api_key_from_env,
)
from litellm.types.router import GenericLiteLLMParams
from litellm.types.vector_stores import (
    VECTOR_STORE_OPENAI_PARAMS,
    BaseVectorStoreAuthCredentials,
    VectorStoreCreateOptionalRequestParams,
    VectorStoreCreateResponse,
    VectorStoreFileCounts,
    VectorStoreIndexEndpoints,
    VectorStoreResultContent,
    VectorStoreSearchOptionalRequestParams,
    VectorStoreSearchResponse,
    VectorStoreSearchResult,
)

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


class GeminiVectorStoreConfig(BaseVectorStoreConfig):
    """
    Vector store configuration for Google Gemini File Search.
    """

    def __init__(self) -> None:
        super().__init__()
        self.model_info = GeminiModelInfo()
        self._cached_api_key: Optional[str] = None

    def get_auth_credentials(
        self, litellm_params: dict
    ) -> BaseVectorStoreAuthCredentials:
        """Gemini uses API key in query params, not headers."""
        return {}

    def get_vector_store_endpoints_by_type(self) -> VectorStoreIndexEndpoints:
        """
        Gemini File Search endpoints.
        
        Note: Search is done via generateContent with file_search tool,
        not a dedicated search endpoint.
        """
        return {
            "read": [("POST", "/models/{model}:generateContent")],
            "write": [("POST", "/fileSearchStores")],
        }

    def get_supported_openai_params(
        self, model: str
    ) -> List[VECTOR_STORE_OPENAI_PARAMS]:
        """Supported parameters for Gemini File Search."""
        return ["max_num_results", "filters"]

    def validate_environment(
        self, headers: dict, litellm_params: Optional[GenericLiteLLMParams]
    ) -> dict:
        """Validate and set up headers for Gemini API."""
        headers = headers or {}
        headers.setdefault("Content-Type", "application/json")
        if litellm_params:
            api_key = litellm_params.get("api_key") or get_api_key_from_env()
            if api_key:
                self._cached_api_key = api_key
        
        return headers

    def get_complete_url(self, api_base: Optional[str], litellm_params: dict) -> str:
        """
        Get the complete base URL for Gemini API.
        
        Note: This returns the base URL WITHOUT the API key.
        The API key will be appended to specific endpoint URLs in the transform methods.
        """
        if api_base is None:
            api_base = GeminiModelInfo.get_api_base()
        
        if api_base is None:
            raise ValueError("GEMINI_API_BASE is not set")
        
        # Ensure we're using the v1beta version for File Search
        api_version = "v1beta"
        return f"{api_base}/{api_version}"

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, httpx.Headers]
    ) -> GeminiError:
        """Return Gemini-specific error class."""
        return GeminiError(
            status_code=status_code,
            message=error_message,
            headers=headers,
        )

    def transform_search_vector_store_request(
        self,
        vector_store_id: str,
        query: Union[str, List[str]],
        vector_store_search_optional_params: VectorStoreSearchOptionalRequestParams,
        api_base: str,
        litellm_logging_obj: LiteLLMLoggingObj,
        litellm_params: dict,
    ) -> Tuple[str, Dict]:
        """
        Transform search request to Gemini's generateContent format.
        
        Gemini File Search works by calling generateContent with a file_search tool.
        """
        # Convert query list to single string if needed
        if isinstance(query, list):
            query = " ".join(query)

        # Get model from litellm_params or use default
        # Note: File Search requires gemini-2.5-flash or later
        model = litellm_params.get("model") or "gemini-2.5-flash"
        if model and model.startswith("gemini/"):
            model = model.replace("gemini/", "")

        # Get API key - Gemini requires it as a query parameter
        api_key = litellm_params.get("api_key") or GeminiModelInfo.get_api_key()
        if not api_key:
            raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY is required")

        # Build the URL for generateContent with API key
        url = f"{api_base}/models/{model}:generateContent?key={api_key}"

        # Build file_search tool configuration (using snake_case as per Gemini docs)
        file_search_config: Dict[str, Any] = {
            "file_search_store_names": [vector_store_id]
        }

        # Add metadata filter if provided
        metadata_filter = vector_store_search_optional_params.get("filters")
        if metadata_filter:
            # Convert to Gemini filter syntax if it's a dict
            if isinstance(metadata_filter, dict):
                # Simple conversion - may need more sophisticated mapping
                filter_parts = []
                for key, value in metadata_filter.items():
                    if isinstance(value, str):
                        filter_parts.append(f'{key} = "{value}"')
                    else:
                        filter_parts.append(f'{key} = {value}')
                file_search_config["metadata_filter"] = " AND ".join(filter_parts)
            else:
                file_search_config["metadata_filter"] = metadata_filter

        # Build request body
        request_body: Dict[str, Any] = {
            "contents": [
                {
                    "parts": [{"text": query}]
                }
            ],
            "tools": [
                {
                    "file_search": file_search_config
                }
            ],
        }

        # Add max_num_results if specified
        max_results = vector_store_search_optional_params.get("max_num_results")
        if max_results:
            # This might need to be added to generationConfig or tool config
            # depending on Gemini's API requirements
            request_body.setdefault("generationConfig", {})["candidateCount"] = 1

        litellm_logging_obj.model_call_details["query"] = query
        litellm_logging_obj.model_call_details["vector_store_id"] = vector_store_id

        return url, request_body

    def transform_search_vector_store_response(
        self, response: httpx.Response, litellm_logging_obj: LiteLLMLoggingObj
    ) -> VectorStoreSearchResponse:
        """
        Transform Gemini's generateContent response to standard format.
        
        Extracts grounding metadata and citations from the response.
        """
        try:
            response_data = response.json()
            results: List[VectorStoreSearchResult] = []

            # Extract candidates and grounding metadata
            candidates = response_data.get("candidates", [])
            
            for candidate in candidates:
                grounding_metadata = candidate.get("groundingMetadata", {})
                grounding_chunks = grounding_metadata.get("groundingChunks", [])
                
                # Process each grounding chunk
                for chunk in grounding_chunks:
                    retrieved_context = chunk.get("retrievedContext")
                    
                    if retrieved_context:
                        # This is from file search
                        text = retrieved_context.get("text", "")
                        uri = retrieved_context.get("uri", "")
                        title = retrieved_context.get("title", "")
                        
                        # Extract file_id from URI if available
                        file_id = uri if uri else None
                        
                        results.append(
                            VectorStoreSearchResult(
                                score=None,  # Gemini doesn't provide explicit scores
                                content=[VectorStoreResultContent(text=text, type="text")],
                                file_id=file_id,
                                filename=title if title else None,
                                attributes={
                                    "uri": uri,
                                    "title": title,
                                },
                            )
                        )

                # Also extract from grounding supports for more detailed citations
                grounding_supports = grounding_metadata.get("groundingSupports", [])
                for support in grounding_supports:
                    segment = support.get("segment", {})
                    text = segment.get("text", "")
                    
                    grounding_chunk_indices = support.get("groundingChunkIndices", [])
                    confidence_scores = support.get("confidenceScores", [])
                    
                    # Use first confidence score as relevance score
                    score = confidence_scores[0] if confidence_scores else None
                    
                    # Only add if we have meaningful text and it's not a duplicate
                    if text:
                        already_exists = False
                        for record in results:
                            contents = record.get("content") or []
                            if contents and contents[0].get("text") == text:
                                already_exists = True
                                break
                        if already_exists:
                            continue
                        results.append(
                            VectorStoreSearchResult(
                                score=score,
                                content=[VectorStoreResultContent(text=text, type="text")],
                                attributes={
                                    "grounding_chunk_indices": grounding_chunk_indices,
                                },
                            )
                        )

            query = litellm_logging_obj.model_call_details.get("query", "")
            
            return VectorStoreSearchResponse(
                object="vector_store.search_results.page",
                search_query=query,
                data=results,
            )

        except Exception as e:
            raise self.get_error_class(
                error_message=f"Failed to parse Gemini response: {str(e)}",
                status_code=response.status_code,
                headers=response.headers,
            )

    def transform_create_vector_store_request(
        self,
        vector_store_create_optional_params: VectorStoreCreateOptionalRequestParams,
        api_base: str,
    ) -> Tuple[str, Dict]:
        """
        Transform create request to Gemini's fileSearchStores format.
        """
        url = f"{api_base}/fileSearchStores"
        
        # Append API key as query parameter (required by Gemini)
        api_key = self._cached_api_key or get_api_key_from_env()
        if api_key:
            url = f"{url}?key={api_key}"

        request_body: Dict[str, Any] = {}

        # Add display name if provided
        name = vector_store_create_optional_params.get("name")
        if name:
            request_body["displayName"] = name

        return url, request_body

    def transform_create_vector_store_response(
        self, response: httpx.Response
    ) -> VectorStoreCreateResponse:
        """
        Transform Gemini's fileSearchStore response to standard format.
        """
        try:
            response_data = response.json()
            
            # Extract store name (format: fileSearchStores/xxxxxxx)
            store_name = response_data.get("name", "")
            display_name = response_data.get("displayName", "")
            create_time = response_data.get("createTime", "")

            # Convert ISO timestamp to Unix timestamp
            import datetime
            created_at = None
            if create_time:
                try:
                    dt = datetime.datetime.fromisoformat(create_time.replace("Z", "+00:00"))
                    created_at = int(dt.timestamp())
                except Exception:
                    created_at = None

            return VectorStoreCreateResponse(
                id=store_name,
                object="vector_store",
                created_at=created_at or 0,
                name=display_name,
                bytes=0,  # Gemini doesn't provide size info on creation
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
            raise self.get_error_class(
                error_message=f"Failed to parse Gemini create response: {str(e)}",
                status_code=response.status_code,
                headers=response.headers,
            )

