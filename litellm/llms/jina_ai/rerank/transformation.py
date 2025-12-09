"""
Transformation logic from Cohere's /v1/rerank format to Jina AI's  `/v1/rerank` format. 

Why separate file? Make it easy to see how transformation works

Docs - https://jina.ai/reranker
"""

from typing import Any, Dict, List, Optional, Tuple, Union

from httpx import URL, Response

from litellm._uuid import uuid
from litellm.llms.base_llm.chat.transformation import LiteLLMLoggingObj
from litellm.llms.base_llm.rerank.transformation import BaseRerankConfig
from litellm.types.rerank import (
    OptionalRerankParams,
    RerankBilledUnits,
    RerankResponse,
    RerankResponseMeta,
    RerankTokens,
)
from litellm.types.utils import ModelInfo


class JinaAIRerankConfig(BaseRerankConfig):
    def get_supported_cohere_rerank_params(self, model: str) -> list:
        return [
            "query",
            "top_n",
            "documents",
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
        optional_params = {}
        supported_params = self.get_supported_cohere_rerank_params(model)
        for k, v in non_default_params.items():
            if k in supported_params:
                optional_params[k] = v
        return dict(OptionalRerankParams(
            **optional_params,
        ))

    def get_complete_url(
        self, 
        api_base: Optional[str], 
        model: str,
        optional_params: Optional[dict] = None,
    ) -> str:
        base_path = "/v1/rerank"

        if api_base is None:
            return "https://api.jina.ai/v1/rerank"
        base = URL(api_base)
        # Reconstruct URL with cleaned path
        cleaned_base = str(base.copy_with(path=base_path))

        return cleaned_base

    def transform_rerank_request(
        self, model: str, optional_rerank_params: Dict, headers: Dict
    ) -> Dict:
        return {"model": model, **optional_rerank_params}

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

        logging_obj.post_call(original_response=raw_response.text)

        _json_response = raw_response.json()

        _billed_units = RerankBilledUnits(**_json_response.get("usage", {}))
        _tokens = RerankTokens(**_json_response.get("usage", {}))
        rerank_meta = RerankResponseMeta(billed_units=_billed_units, tokens=_tokens)

        _results: Optional[List[dict]] = _json_response.get("results")

        if _results is None:
            raise ValueError(f"No results found in the response={_json_response}")

        # Transform Jina AI's response format to match LiteLLM's expected format
        # Jina AI returns: {"index": 0, "relevance_score": 0.72, "document": "hello"}
        # LiteLLM expects: {"index": 0, "relevance_score": 0.72, "document": {"text": "hello"}}
        transformed_results = []
        for result in _results:
            transformed_result = {
                "index": result["index"],
                "relevance_score": result["relevance_score"],
            }
            # Convert document from string to dict format if it exists
            if "document" in result and isinstance(result["document"], str):
                transformed_result["document"] = {"text": result["document"]}
            elif "document" in result:
                # If it's already a dict, keep it as is
                transformed_result["document"] = result["document"]
            transformed_results.append(transformed_result)

        return RerankResponse(
            id=_json_response.get("id") or str(uuid.uuid4()),
            results=transformed_results,  # type: ignore
            meta=rerank_meta,
        )  # Return response

    def validate_environment(
        self, 
        headers: Dict, 
        model: str, 
        api_key: Optional[str] = None,
        optional_params: Optional[dict] = None,
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

    def calculate_rerank_cost(
        self,
        model: str,
        custom_llm_provider: Optional[str] = None,
        billed_units: Optional[RerankBilledUnits] = None,
        model_info: Optional[ModelInfo] = None,
    ) -> Tuple[float, float]:
        """
        Jina AI reranker is priced at $0.000000018 per token.
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
