from typing import Any, Dict, List, Union

from litellm.llms.cohere.rerank.transformation import CohereRerankConfig
from litellm.types.rerank import OptionalRerankParams, RerankRequest


class CohereRerankV2Config(CohereRerankConfig):
    """
    Reference: https://docs.cohere.com/v2/reference/rerank
    """

    def __init__(self) -> None:
        pass

    def get_complete_url(
        self,
        api_base: str | None,
        model: str,
        optional_params: dict | None = None,
    ) -> str:
        if api_base:
            api_base = api_base.rstrip("/")
            if api_base.endswith("/v2"):
                api_base = f"{api_base}/rerank"
            elif not api_base.endswith("/v2/rerank"):
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
        non_default_params: dict | None,
        model: str,
        drop_params: bool,
        query: str,
        documents: List[Union[str, Dict[str, Any]]],
        custom_llm_provider: str | None = None,
        top_n: int | None = None,
        rank_fields: List[str] | None = None,
        return_documents: bool | None = True,
        max_chunks_per_doc: int | None = None,
        max_tokens_per_doc: int | None = None,
        instruction: str | None = None,
    ) -> Dict:
        """
        Map Cohere rerank params

        No mapping required - returns all supported params
        """
        return dict(
            OptionalRerankParams(
                query=query,
                documents=documents,
                top_n=top_n,
                rank_fields=rank_fields,
                return_documents=return_documents,
                max_tokens_per_doc=max_tokens_per_doc,
            )
        )

    def transform_rerank_request(
        self,
        model: str,
        optional_rerank_params: Dict,
        headers: dict,
        litellm_params: dict | None = None,
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
