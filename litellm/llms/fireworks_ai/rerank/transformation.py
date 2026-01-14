"""
Fireworks AI Rerank API transformation

Reference: https://docs.fireworks.ai/inference-api-reference/rerank
"""

from typing import Any, Dict, List, Optional, Union

import httpx

from litellm._uuid import uuid
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.base_llm.rerank.transformation import BaseRerankConfig
from litellm.llms.fireworks_ai.common_utils import FireworksAIMixin
from litellm.types.rerank import (
    RerankBilledUnits,
    RerankResponse,
    RerankResponseDocument,
    RerankResponseMeta,
    RerankResponseResult,
    RerankTokens,
)


class FireworksAIRerankConfig(FireworksAIMixin, BaseRerankConfig):
    """
    Fireworks AI Rerank API configuration
    """

    def get_complete_url(
        self,
        api_base: Optional[str],
        model: str,
        optional_params: Optional[dict] = None,
    ) -> str:
        if api_base:
            # Remove trailing slashes and ensure clean base URL
            api_base = api_base.rstrip("/")
            if not api_base.endswith("/rerank"):
                if api_base.endswith("/v1"):
                    api_base = f"{api_base}/rerank"
                elif api_base.endswith("/inference/v1"):
                    api_base = f"{api_base}/rerank"
                else:
                    api_base = f"{api_base}/inference/v1/rerank"
            return api_base
        return "https://api.fireworks.ai/inference/v1/rerank"

    def get_supported_cohere_rerank_params(self, model: str) -> list:
        return [
            "query",
            "documents",
            "top_n",
            "return_documents",
        ]

    def map_cohere_rerank_params(
        self,
        non_default_params: Optional[dict],
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
    ) -> Dict[str, Any]:
        """
        Map Cohere rerank params to Fireworks AI rerank params
        """
        params: Dict[str, Any] = {
            "query": query,
            "documents": documents,
        }
        
        if top_n is not None:
            params["top_n"] = top_n
        
        if return_documents is not None:
            params["return_documents"] = return_documents
        
        # Fireworks AI doesn't support these params
        if rank_fields is not None:
            # Silently ignore rank_fields as Fireworks AI doesn't support it
            pass
        
        if max_chunks_per_doc is not None:
            # Silently ignore max_chunks_per_doc as Fireworks AI doesn't support it
            pass
        
        if max_tokens_per_doc is not None:
            # Silently ignore max_tokens_per_doc as Fireworks AI doesn't support it
            pass
        
        return params

    def validate_environment(  # type: ignore[override]
        self,
        headers: dict,
        model: str,
        api_key: Optional[str] = None,
        optional_params: Optional[dict] = None,
    ) -> dict:
        api_key = self._get_api_key(api_key)
        if api_key is None:
            raise ValueError(
                "FIREWORKS_API_KEY is not set. Please set 'FIREWORKS_API_KEY' or 'FIREWORKS_AI_API_KEY' in your environment"
            )

        default_headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
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
        Transform request to Fireworks AI rerank format
        """
        if "query" not in optional_rerank_params:
            raise ValueError("query is required for Fireworks AI rerank")
        if "documents" not in optional_rerank_params:
            raise ValueError("documents is required for Fireworks AI rerank")

        # Handle model name - Fireworks AI expects model name like "fireworks/qwen3-reranker-8b"
        # Remove fireworks_ai/ prefix if present
        if model.startswith("fireworks_ai/"):
            model = model.replace("fireworks_ai/", "")
        
        # If model doesn't start with "fireworks/", add it
        # But don't add if it already has the prefix
        if not model.startswith("fireworks/"):
            model = f"fireworks/{model}"

        request_data = {
            "model": model,
            "query": optional_rerank_params["query"],
            "documents": optional_rerank_params["documents"],
        }

        if "top_n" in optional_rerank_params and optional_rerank_params["top_n"] is not None:
            request_data["top_n"] = optional_rerank_params["top_n"]

        if "return_documents" in optional_rerank_params and optional_rerank_params["return_documents"] is not None:
            request_data["return_documents"] = optional_rerank_params["return_documents"]

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
        Transform Fireworks AI rerank response to LiteLLM RerankResponse format
        """
        try:
            raw_response_json = raw_response.json()
        except Exception as e:
            raise self.get_error_class(
                error_message=f"Failed to parse response: {str(e)}",
                status_code=raw_response.status_code,
                headers=raw_response.headers,
            )

        # Fireworks AI response format:
        # {
        #   "object": "list",
        #   "model": "accounts/fireworks/models/qwen3-reranker-8b",
        #   "data": [
        #     {
        #       "index": 0,
        #       "relevance_score": 0.95,
        #       "document": "..."  
        #     }
        #   ],
        #   "usage": {
        #     "total_tokens": 100,
        #     "prompt_tokens": 50,
        #     "completion_tokens": 50
        #   }
        # }

        # Extract usage information
        usage = raw_response_json.get("usage", {})
        _billed_units = RerankBilledUnits(
            search_units=usage.get("total_tokens", 0)
        )
        _tokens = RerankTokens(
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
        )
        rerank_meta = RerankResponseMeta(billed_units=_billed_units, tokens=_tokens)

        # Extract results - Fireworks AI uses "data" instead of "results"
        _results: Optional[List[dict]] = raw_response_json.get("data") or raw_response_json.get("results")

        if _results is None:
            raise ValueError(f"No results found in the response={raw_response_json}")

        rerank_results: List[RerankResponseResult] = []

        for result in _results:
            # Validate required fields exist
            if not all(key in result for key in ["index", "relevance_score"]):
                raise ValueError(f"Missing required fields in the result={result}")

            # Get document data - Fireworks AI returns document as a string directly
            document_text = result.get("document")
            document = None
            if document_text:
                # Handle both string and object formats
                if isinstance(document_text, str):
                    document = RerankResponseDocument(text=document_text)
                elif isinstance(document_text, dict):
                    # Handle object format if it exists
                    text = document_text.get("text", "")
                    if text:
                        document = RerankResponseDocument(text=str(text))

            # Create typed result
            rerank_result = RerankResponseResult(
                index=int(result["index"]),
                relevance_score=float(result["relevance_score"]),
            )

            # Only add document if it exists
            if document:
                rerank_result["document"] = document

            rerank_results.append(rerank_result)

        # Use model name as id if no id is provided
        response_id = raw_response_json.get("id") or raw_response_json.get("model") or str(uuid.uuid4())

        return RerankResponse(
            id=response_id,
            results=rerank_results,
            meta=rerank_meta,
        )

