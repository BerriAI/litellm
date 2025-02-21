"""
Transformation logic from Cohere's /v1/rerank format to Jina AI's  `/v1/rerank` format. 

Why separate file? Make it easy to see how transformation works

Docs - https://jina.ai/reranker
"""

import uuid
from typing import Any, Dict, List, Optional, Union

from httpx import URL, Response

from litellm.llms.base_llm.chat.transformation import LiteLLMLoggingObj
from litellm.llms.base_llm.rerank.transformation import (
    BaseRerankConfig,
    LiteLLMLoggingObj,
)
from litellm.types.rerank import (
    OptionalRerankParams,
    RerankBilledUnits,
    RerankResponse,
    RerankResponseMeta,
    RerankTokens,
)


class JinaAIRerankConfig(BaseRerankConfig):
    def get_supported_cohere_rerank_params(self, model: str) -> list:
        return [
            "query",
            "top_n",
            "documents",
            "rank_fields",
            "return_documents",
        ]

    def map_cohere_rerank_params(
        self,
        non_default_params: Optional[dict],
        model: str,
        drop_params: bool,
        query: str,
        documents: List[str | Dict[str, Any]],
        custom_llm_provider: Optional[str] = None,
        top_n: Optional[int] = None,
        rank_fields: Optional[List[str]] = None,
        return_documents: Optional[bool] = True,
        max_chunks_per_doc: Optional[int] = None,
    ) -> OptionalRerankParams:
        return OptionalRerankParams(
            query=query,
            top_n=top_n,
            documents=documents,
            rank_fields=rank_fields,
            return_documents=return_documents,
        )

    def get_complete_url(self, api_base: Optional[str], model: str) -> str:
        base_path = "/v1/rerank"

        if api_base is None:
            return "https://api.jina.ai/v1/rerank"
        base = URL(api_base)
        # Reconstruct URL with cleaned path
        cleaned_base = str(base.copy_with(path=base_path))

        return cleaned_base

    def transform_rerank_request(
        self, model: str, optional_rerank_params: OptionalRerankParams, headers: Dict
    ) -> Dict:
        return {
            "model": model,
            **optional_rerank_params,
        }

    def transform_rerank_response(
        self,
        model: str,
        raw_response: Response,
        model_response: RerankResponse,
        logging_obj: LiteLLMLoggingObj,
        api_key: Optional[str] = None,
        request_data: Dict = {},
        optional_params: Dict = {},
        litellm_params: Dict = {},
    ) -> RerankResponse:
        if raw_response.status_code != 200:
            raise Exception(raw_response.text)

        _json_response = raw_response.json()

        _billed_units = RerankBilledUnits(**_json_response.get("usage", {}))
        _tokens = RerankTokens(**_json_response.get("usage", {}))
        rerank_meta = RerankResponseMeta(billed_units=_billed_units, tokens=_tokens)

        _results: Optional[List[dict]] = _json_response.get("results")

        if _results is None:
            raise ValueError(f"No results found in the response={_json_response}")

        return RerankResponse(
            id=_json_response.get("id") or str(uuid.uuid4()),
            results=_results,  # type: ignore
            meta=rerank_meta,
        )  # Return response

    def validate_environment(
        self, headers: Dict, model: str, api_key: str | None = None
    ) -> Dict:
        if api_key is None:
            raise ValueError(
                "api_key is required. Set via `api_key` parameter or `JINA_API_KEY` environment variable."
            )
        return {
            "accept": "application/json",
            "content-type": "application/json",
            "authorization": f"Bearer {api_key}",
        }
