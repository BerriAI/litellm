"""
Transformation logic from Cohere's /v1/rerank format to Infinity's  `/v1/rerank` format. 

Why separate file? Make it easy to see how transformation works
"""

import uuid
from typing import List, Optional

import httpx

import litellm
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.cohere.rerank.transformation import CohereRerankConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.rerank import (
    RerankBilledUnits,
    RerankResponse,
    RerankResponseDocument,
    RerankResponseMeta,
    RerankResponseResult,
    RerankTokens,
)

from .common_utils import InfinityError


class InfinityRerankConfig(CohereRerankConfig):
    def get_complete_url(self, api_base: Optional[str], model: str) -> str:
        if api_base is None:
            raise ValueError("api_base is required for Infinity rerank")
        # Remove trailing slashes and ensure clean base URL
        api_base = api_base.rstrip("/")
        if not api_base.endswith("/rerank"):
            api_base = f"{api_base}/rerank"
        return api_base

    def validate_environment(
        self,
        headers: dict,
        model: str,
        api_key: Optional[str] = None,
    ) -> dict:
        if api_key is None:
            api_key = (
                get_secret_str("INFINITY_API_KEY")
                or get_secret_str("INFINITY_API_KEY")
                or litellm.infinity_key
            )

        default_headers = {
            "Authorization": f"bearer {api_key}",
            "accept": "application/json",
            "content-type": "application/json",
        }

        # If 'Authorization' is provided in headers, it overrides the default.
        if "Authorization" in headers:
            default_headers["Authorization"] = headers["Authorization"]

        # Merge other headers, overriding any default ones except Authorization
        return {**default_headers, **headers}

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
        Transform Infinity rerank response

        No transformation required, Infinity follows Cohere API response format
        """
        try:
            raw_response_json = raw_response.json()
        except Exception:
            raise InfinityError(
                message=raw_response.text, status_code=raw_response.status_code
            )

        _billed_units = RerankBilledUnits(**raw_response_json.get("usage", {}))
        _tokens = RerankTokens(
            input_tokens=raw_response_json.get("usage", {}).get("prompt_tokens", 0),
            output_tokens=(
                raw_response_json.get("usage", {}).get("total_tokens", 0)
                - raw_response_json.get("usage", {}).get("prompt_tokens", 0)
            ),
        )
        rerank_meta = RerankResponseMeta(billed_units=_billed_units, tokens=_tokens)

        cohere_results: List[RerankResponseResult] = []
        if raw_response_json.get("results"):
            for result in raw_response_json.get("results"):
                _rerank_response = RerankResponseResult(
                    index=result.get("index"),
                    relevance_score=result.get("relevance_score"),
                )
                if result.get("document"):
                    _rerank_response["document"] = RerankResponseDocument(
                        text=result.get("document")
                    )
                cohere_results.append(_rerank_response)
        if cohere_results is None:
            raise ValueError(f"No results found in the response={raw_response_json}")

        return RerankResponse(
            id=raw_response_json.get("id") or str(uuid.uuid4()),
            results=cohere_results,
            meta=rerank_meta,
        )  # Return response
