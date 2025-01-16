from typing import Any, Dict, List, Optional, Union

from litellm.llms.base_llm.rerank.transformation import BaseRerankConfig
from litellm.types.rerank import OptionalRerankParams


def get_optional_rerank_params(
    rerank_provider_config: BaseRerankConfig,
    model: str,
    drop_params: bool,
    query: str,
    documents: List[Union[str, Dict[str, Any]]],
    custom_llm_provider: Optional[str] = None,
    top_n: Optional[int] = None,
    rank_fields: Optional[List[str]] = None,
    return_documents: Optional[bool] = True,
    max_chunks_per_doc: Optional[int] = None,
    non_default_params: Optional[dict] = None,
) -> OptionalRerankParams:
    return rerank_provider_config.map_cohere_rerank_params(
        model=model,
        drop_params=drop_params,
        query=query,
        documents=documents,
        custom_llm_provider=custom_llm_provider,
        top_n=top_n,
        rank_fields=rank_fields,
        return_documents=return_documents,
        max_chunks_per_doc=max_chunks_per_doc,
        non_default_params=non_default_params,
    )
