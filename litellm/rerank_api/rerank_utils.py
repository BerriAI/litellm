from typing import Any, Final, Literal

import litellm
from litellm.litellm_core_utils.core_helpers import as_str_mapping
from litellm.llms.base_llm.rerank.transformation import BaseRerankConfig
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager


def _catalog_rerank_strategy(model: str) -> Literal["pointwise", "multiple_choice"] | None:
    info: Final = as_str_mapping(litellm.model_cost.get(model))  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]  # Narrow the legacy untyped catalog entry at the boundary.
    strategy: Final = None if info is None else info.get("rerank_strategy")
    match strategy:
        case "pointwise":
            return "pointwise"
        case "multiple_choice":
            return "multiple_choice"
        case _:
            return None


def get_decision_rerank_strategy(
    model: str, custom_llm_provider: LlmProviders
) -> Literal["pointwise", "multiple_choice"] | None:
    """Read the catalog method for an already resolved model and provider."""
    if ProviderConfigManager.get_provider_decisions_config(model, custom_llm_provider) is None:
        return None
    return _catalog_rerank_strategy(f"{custom_llm_provider.value}/{model}") or _catalog_rerank_strategy(model)


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
