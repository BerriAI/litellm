"""
Transformation logic for Voyage AI's /v1/rerank endpoint.

Why separate file? Make it easy to see how transformation works

Docs - https://docs.voyageai.com/docs/reranker
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import httpx

from litellm.llms.base_llm.chat.transformation import LiteLLMLoggingObj
from litellm.llms.base_llm.rerank.transformation import BaseRerankConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.rerank import (
    OptionalRerankParams,
    RerankBilledUnits,
    RerankResponse,
    RerankResponseMeta,
    RerankTokens,
)
from litellm.types.utils import ModelInfo

from ..embedding.transformation import VoyageError


class VoyageRerankConfig(BaseRerankConfig):
    """
    Configuration for Voyage AI Rerank API

    Reference: https://docs.voyageai.com/reference/reranker-api

    Supported models:
    - rerank-2.5 (32K context)
    - rerank-2.5-lite (32K context)
    - rerank-2 (16K context) - legacy
    - rerank-2-lite (8K context) - legacy
    """

    def get_supported_cohere_rerank_params(self, model: str) -> list:
        """
        Voyage AI supports these parameters:
        - query: The query to rerank documents against
        - documents: List of documents to rerank
        - top_k: Number of top results to return (mapped from top_n)
        - return_documents: Whether to return document text in response
        - truncation: Whether to truncate documents that exceed context length
        """
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
        Map Cohere-style rerank params to Voyage AI format.

        Voyage AI uses 'top_k' instead of 'top_n'.
        """
        optional_params: Dict[str, Any] = {
            "query": query,
            "documents": documents,
        }

        if top_n is not None:
            optional_params["top_k"] = top_n

        if return_documents is not None:
            optional_params["return_documents"] = return_documents

        return dict(OptionalRerankParams(**optional_params))

    def get_complete_url(
        self,
        api_base: Optional[str],
        model: str,
        optional_params: Optional[dict] = None,
    ) -> str:
        """
        Get the complete URL for Voyage AI rerank endpoint.

        Default: https://api.voyageai.com/v1/rerank
        """
        if api_base is None:
            return "https://api.voyageai.com/v1/rerank"

        # Clean up api_base and ensure it ends with /v1/rerank
        api_base = api_base.rstrip("/")
        if not api_base.endswith("/v1/rerank"):
            if api_base.endswith("/v1"):
                api_base = f"{api_base}/rerank"
            else:
                api_base = f"{api_base}/v1/rerank"

        return api_base

    def transform_rerank_request(
        self,
        model: str,
        optional_rerank_params: Dict,
        headers: Dict,
    ) -> Dict:
        """
        Transform request to Voyage AI format.

        Voyage AI request format:
        {
            "model": "rerank-2.5",
            "query": "...",
            "documents": [...],
            "top_k": 3,  # optional
            "return_documents": true  # optional
        }
        """
        request_data: Dict[str, Any] = {
            "model": model,
        }

        if "query" in optional_rerank_params:
            request_data["query"] = optional_rerank_params["query"]

        if "documents" in optional_rerank_params:
            request_data["documents"] = optional_rerank_params["documents"]

        if "top_k" in optional_rerank_params:
            request_data["top_k"] = optional_rerank_params["top_k"]

        if "return_documents" in optional_rerank_params:
            request_data["return_documents"] = optional_rerank_params["return_documents"]

        return request_data

    def transform_rerank_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: RerankResponse,
        logging_obj: LiteLLMLoggingObj,
        api_key: Optional[str] = None,
        request_data: Dict = {},
        optional_params: Dict = {},
        litellm_params: Dict = {},
    ) -> RerankResponse:
        """
        Transform Voyage AI response to LiteLLM RerankResponse format.

        Voyage AI response format:
        {
            "object": "list",
            "data": [
                {"relevance_score": 0.88, "index": 0},
                {"relevance_score": 0.35, "index": 2}
            ],
            "model": "rerank-2.5",
            "usage": {"total_tokens": 30}
        }

        LiteLLM expects:
        {
            "results": [
                {"index": 0, "relevance_score": 0.88},
                ...
            ],
            "meta": {...}
        }
        """
        if raw_response.status_code != 200:
            raise VoyageError(
                message=raw_response.text,
                status_code=raw_response.status_code,
            )

        logging_obj.post_call(original_response=raw_response.text)

        try:
            _json_response = raw_response.json()
        except Exception:
            raise VoyageError(
                message=f"Failed to parse response: {raw_response.text}",
                status_code=raw_response.status_code,
            )

        # Voyage AI returns results in "data" key, not "results"
        _results: Optional[List[dict]] = _json_response.get("data")

        if _results is None:
            raise ValueError(f"No results found in the response={_json_response}")

        # Transform Voyage AI's response format to match LiteLLM's expected format
        transformed_results = []
        for result in _results:
            transformed_result: Dict[str, Any] = {
                "index": result["index"],
                "relevance_score": result["relevance_score"],
            }
            # Include document if present in response
            if "document" in result:
                if isinstance(result["document"], str):
                    transformed_result["document"] = {"text": result["document"]}
                else:
                    transformed_result["document"] = result["document"]
            transformed_results.append(transformed_result)

        # Build usage/meta information
        usage = _json_response.get("usage", {})
        total_tokens = usage.get("total_tokens", 0)

        _billed_units = RerankBilledUnits(total_tokens=total_tokens)
        _tokens = RerankTokens(input_tokens=total_tokens, output_tokens=0)
        rerank_meta = RerankResponseMeta(billed_units=_billed_units, tokens=_tokens)

        return RerankResponse(
            id=_json_response.get("id", f"voyage-rerank-{model}"),
            results=transformed_results,  # type: ignore
            meta=rerank_meta,
        )

    def validate_environment(
        self,
        headers: Dict,
        model: str,
        api_key: Optional[str] = None,
        optional_params: Optional[dict] = None,
    ) -> Dict:
        """
        Validate API key and return headers for Voyage AI.

        Checks for API key in this order:
        1. Passed api_key parameter
        2. VOYAGE_API_KEY environment variable
        3. VOYAGE_AI_API_KEY environment variable
        """
        if api_key is None:
            api_key = get_secret_str("VOYAGE_API_KEY") or get_secret_str(
                "VOYAGE_AI_API_KEY"
            )

        if api_key is None:
            raise ValueError(
                "Voyage AI API key is required. Set via `api_key` parameter or "
                "`VOYAGE_API_KEY` environment variable."
            )

        return {
            "Authorization": f"Bearer {api_key}",
            "content-type": "application/json",
        }

    def calculate_rerank_cost(
        self,
        model: str,
        custom_llm_provider: Optional[str] = None,
        billed_units: Optional[RerankBilledUnits] = None,
        model_info: Optional[ModelInfo] = None,
    ) -> Tuple[float, float]:
        """
        Calculate cost for Voyage AI rerank.

        Pricing (per 1K tokens):
        - rerank-2.5: $0.00005
        - rerank-2.5-lite: $0.00002
        - rerank-2: $0.00005
        - rerank-2-lite: $0.00002
        """
        if (
            model_info is None
            or "input_cost_per_token" not in model_info
            or model_info["input_cost_per_token"] is None
            or billed_units is None
        ):
            return 0.0, 0.0

        total_tokens = billed_units.get("total_tokens")
        if total_tokens is None:
            return 0.0, 0.0

        input_cost = model_info["input_cost_per_token"] * total_tokens
        return input_cost, 0.0

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, httpx.Headers]
    ):
        return VoyageError(
            message=error_message,
            status_code=status_code,
            headers=headers,
        )
