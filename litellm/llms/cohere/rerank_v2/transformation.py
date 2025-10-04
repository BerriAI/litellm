from typing import Any, Dict, List, Optional, Union

from litellm.llms.cohere.rerank.transformation import CohereRerankConfig
from litellm.types.rerank import OptionalRerankParams, RerankRequest


class CohereRerankV2Config(CohereRerankConfig):
    """
    Reference: https://docs.cohere.com/v2/reference/rerank
    """

    def __init__(self) -> None:
        pass

    def get_complete_url(self, api_base: Optional[str], model: str) -> str:
        if api_base:
            # Remove trailing slashes and ensure clean base URL
            api_base = api_base.rstrip("/")
            if not api_base.endswith("/v2/rerank"):
                api_base = f"{api_base}/v2/rerank"
            return api_base
        return "https://api.cohere.ai/v2/rerank"

    def get_supported_cohere_rerank_params(self, model: str) -> list:
        return [
            "query",
            "documents",
            "top_n",
            "max_tokens_per_doc",
            "rank_fields",
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
    ) -> Dict:
        """
        Map Cohere rerank params

        No mapping required - returns all supported params
        """
        return dict(OptionalRerankParams(
            query=query,
            documents=documents,
            top_n=top_n,
            rank_fields=rank_fields,
            return_documents=return_documents,
            max_tokens_per_doc=max_tokens_per_doc,
        ))

    def transform_rerank_request(
        self,
        model: str,
        optional_rerank_params: Dict,
        headers: dict,
    ) -> dict:
        if "query" not in optional_rerank_params:
            raise ValueError("query is required for Cohere rerank")
        if "documents" not in optional_rerank_params:
            raise ValueError("documents is required for Cohere rerank")
        rerank_request = RerankRequest(
            model=model,
            query=optional_rerank_params["query"],
            documents=optional_rerank_params["documents"],
            top_n=optional_rerank_params.get("top_n", None),
            rank_fields=optional_rerank_params.get("rank_fields", None),
            return_documents=optional_rerank_params.get("return_documents", None),
            max_tokens_per_doc=optional_rerank_params.get("max_tokens_per_doc", None),
        )
        return rerank_request.model_dump(exclude_none=True)
