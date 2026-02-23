"""
Transformation logic for Voyage AI's /v1/rerank endpoint.

Docs - https://docs.voyageai.com/docs/reranker
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import httpx

from litellm.llms.base_llm.chat.transformation import LiteLLMLoggingObj
from litellm.llms.base_llm.rerank.transformation import BaseRerankConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.rerank import (
    RerankBilledUnits,
    RerankResponse,
    RerankResponseMeta,
    RerankTokens,
)
from litellm.types.utils import ModelInfo

from ..embedding.transformation import VoyageError


class VoyageRerankConfig(BaseRerankConfig):

    def get_supported_cohere_rerank_params(self, model: str) -> list:
        return ["query", "documents", "top_n", "return_documents"]

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
        # Voyage AI uses 'top_k' instead of 'top_n'
        optional_params: Dict[str, Any] = {"query": query, "documents": documents}
        if top_n is not None:
            optional_params["top_k"] = top_n
        if return_documents is not None:
            optional_params["return_documents"] = return_documents
        # Return as dict - OptionalRerankParams is a TypedDict with total=False
        # so all fields are optional and we can return the dict directly
        return optional_params

    def get_complete_url(
        self,
        api_base: Optional[str],
        model: str,
        optional_params: Optional[dict] = None,
    ) -> str:
        if api_base is None:
            return "https://api.voyageai.com/v1/rerank"
        api_base = api_base.rstrip("/")
        if not api_base.endswith("/v1/rerank"):
            if api_base.endswith("/v1"):
                api_base = f"{api_base}/rerank"
            else:
                api_base = f"{api_base}/v1/rerank"
        return api_base

    def transform_rerank_request(
        self, model: str, optional_rerank_params: Dict, headers: Dict
    ) -> Dict:
        return {"model": model, **optional_rerank_params}

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
        if raw_response.status_code != 200:
            raise VoyageError(
                message=raw_response.text, status_code=raw_response.status_code
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

        # Transform to LiteLLM format
        transformed_results = []
        for result in _results:
            transformed_result: Dict[str, Any] = {
                "index": result["index"],
                "relevance_score": result["relevance_score"],
            }
            if "document" in result:
                if isinstance(result["document"], str):
                    transformed_result["document"] = {"text": result["document"]}
                else:
                    transformed_result["document"] = result["document"]
            transformed_results.append(transformed_result)

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
        if api_key is None:
            api_key = get_secret_str("VOYAGE_API_KEY") or get_secret_str("VOYAGE_AI_API_KEY")
        if api_key is None:
            raise ValueError(
                "Voyage AI API key is required. Set via `api_key` parameter or `VOYAGE_API_KEY` env var."
            )
        return {"Authorization": f"Bearer {api_key}", "content-type": "application/json"}

    def calculate_rerank_cost(
        self,
        model: str,
        custom_llm_provider: Optional[str] = None,
        billed_units: Optional[RerankBilledUnits] = None,
        model_info: Optional[ModelInfo] = None,
    ) -> Tuple[float, float]:
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
        return model_info["input_cost_per_token"] * total_tokens, 0.0

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, httpx.Headers]
    ):
        return VoyageError(message=error_message, status_code=status_code, headers=headers)
