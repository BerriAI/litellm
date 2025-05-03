import uuid
from typing import Any, Dict, List, Optional, Union

import httpx

from litellm.secret_managers.main import get_secret_str
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.base_llm.rerank.transformation import BaseRerankConfig
from litellm.types.rerank import OptionalRerankParams, RerankRequest, RerankResponse
from litellm.llms.voyage.common_utils import VoyageError
from litellm.types.rerank import (
    RerankBilledUnits,
    RerankResponseDocument,
    RerankResponseMeta,
    RerankResponseResult,
    RerankTokens,
)

class VoyageRerankConfig(BaseRerankConfig):
    def __init__(self) -> None:
        pass

    def get_complete_url(self, api_base: Optional[str], model: str) -> str:
        if api_base:
            # Remove trailing slashes and ensure clean base URL
            api_base = api_base.rstrip("/")
            if not api_base.endswith("/v1/rerank"):
                api_base = f"{api_base}/v1/rerank"
            return api_base
        return "https://api.voyageai.com/v1/rerank"

    def validate_environment(
        self,
        headers: dict,
        model: str,
        api_key: Optional[str] = None,
    ) -> dict:
        if api_key is None:
            api_key = (
                get_secret_str("VOYAGE_API_KEY")
                or get_secret_str("VOYAGE_AI_API_KEY")
                or get_secret_str("VOYAGE_AI_TOKEN")
            )
        return {
            "Authorization": f"Bearer {api_key}",
            "content-type": "application/json",
        }

    def get_supported_cohere_rerank_params(self, model: str) -> list:
        return [
            "query",
            "documents",
            "top_k",
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
    ) -> OptionalRerankParams:
        """
        Map Voyage rerank params
        """
        optional_params = {}
        supported_params = self.get_supported_cohere_rerank_params(model)
        for k, v in non_default_params.items():
            if k in supported_params:
                optional_params[k] = v

        # Voyage API uses top_k instead of top_n
        # Assign top_k to top_n if top_n is not None
        if top_n is not None:
            optional_params["top_k"] = top_n
            optional_params["top_n"] = None

        return OptionalRerankParams(
            **optional_params,
        )
    def transform_rerank_request(self, model: str, optional_rerank_params: OptionalRerankParams, headers: dict) -> dict:
        # Transform request to RerankRequest spec
        if "query" not in optional_rerank_params:
            raise ValueError("query is required for Cohere rerank")
        if "documents" not in optional_rerank_params:
            raise ValueError("documents is required for Voyage rerank")
        rerank_request = RerankRequest(
            model=model,
            query=optional_rerank_params["query"],
            documents=optional_rerank_params["documents"],
            # Voyage API uses top_k instead of top_n
            top_k=optional_rerank_params.get("top_k", None),
            return_documents=optional_rerank_params.get("return_documents", None),
        )
        return rerank_request.model_dump(exclude_none=True)

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
        Transform Voyage rerank response
        No transformation required, litellm follows Voyage API response format
        """
        try:
            raw_response_json = raw_response.json()
        except Exception:
            raise VoyageError(
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

        voyage_results: List[RerankResponseResult] = []
        if raw_response_json.get("data"):
            for result in raw_response_json.get("data"):
                _rerank_response = RerankResponseResult(
                    index=result.get("index"),
                    relevance_score=result.get("relevance_score"),
                )
                if result.get("document"):
                    _rerank_response["document"] = RerankResponseDocument(
                        text=result.get("document")
                    )
                voyage_results.append(_rerank_response)
        if voyage_results is None:
            raise ValueError(f"No results found in the response={raw_response_json}")

        return RerankResponse(
            id=raw_response_json.get("id") or str(uuid.uuid4()),
            results=voyage_results,
            meta=rerank_meta,
        )  # Return response