from typing import Any, Final, Literal

import litellm
from litellm.llms.base_llm.rerank.transformation import BaseRerankConfig


def get_decision_rerank_strategy(
    model: str, custom_llm_provider: str | None = None
) -> Literal["pointwise", "multiple_choice"] | None:
    """Read the catalog method for the model, including a gateway's provider-prefixed model."""
    try:
        resolved_model, provider, _, _ = litellm.get_llm_provider(model=model, custom_llm_provider=custom_llm_provider)
    except litellm.BadRequestError:
        return None
    candidates: Final = (f"{provider}/{resolved_model}", resolved_model, model)
    catalog_model: Final = next(
        (
            name
            for name in candidates
            if litellm.model_cost.get(name, {}).get("rerank_strategy") is not None  # pyright: ignore[reportUnknownMemberType]  # check capability presence without invoking native provider metadata
        ),
        None,
    )
    if catalog_model is None:
        return None
    strategy: Final = litellm.get_model_info(model=catalog_model).get("rerank_strategy")
    return strategy if strategy in ("pointwise", "multiple_choice") else None


def get_optional_rerank_params(
    rerank_provider_config: BaseRerankConfig,
    model: str,
    drop_params: bool,
    query: str,
    documents: list[str | dict[str, Any]],
    custom_llm_provider: str | None = None,
    top_n: int | None = None,
    rank_fields: list[str] | None = None,
    return_documents: bool | None = True,
    max_chunks_per_doc: int | None = None,
    max_tokens_per_doc: int | None = None,
    instruction: str | None = None,
    non_default_params: dict | None = None,
) -> dict:
    all_non_default_params: Final = non_default_params or {}
    if query is not None:
        all_non_default_params["query"] = query
    if top_n is not None:
        all_non_default_params["top_n"] = top_n
    if documents is not None:
        all_non_default_params["documents"] = documents
    if return_documents is not None:
        all_non_default_params["return_documents"] = return_documents
    if max_chunks_per_doc is not None:
        all_non_default_params["max_chunks_per_doc"] = max_chunks_per_doc
    if max_tokens_per_doc is not None:
        all_non_default_params["max_tokens_per_doc"] = max_tokens_per_doc
    if instruction is not None:
        # Also surfaced in non_default_params so providers that read it from
        # there (e.g. DeepInfra) keep working now that `rerank()` consumes
        # `instruction` as a named param instead of leaving it in **kwargs.
        all_non_default_params["instruction"] = instruction
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
        max_tokens_per_doc=max_tokens_per_doc,
        instruction=instruction,
        non_default_params=all_non_default_params,
    )
