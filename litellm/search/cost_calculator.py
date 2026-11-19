"""
Cost calculation for search providers.
"""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

from pydantic import TypeAdapter, ValidationError

import litellm
from litellm.litellm_core_utils.core_helpers import as_str_mapping, attribute_or_none
from litellm.litellm_core_utils.get_model_cost_map import GetModelCostMap
from litellm.litellm_core_utils.llm_cost_calc.pricing_bridge import AdjustFacts, SearchDocument, price_units
from litellm.llms.base_llm.search.transformation import SearchResponse
from litellm.rust_bridge._native import NativeReceipt
from litellm.types.utils import ModelInfo
from litellm.utils import get_model_info

PROVIDER_USAGE_ADAPTER: Final[TypeAdapter[tuple[Mapping[str, object], ...]]] = TypeAdapter(
    tuple[Mapping[str, object], ...]
)
EMPTY_OPTIONAL_PARAMS: Final[Mapping[str, object]] = MappingProxyType({})
SEARCH_MODE_PARAM_BY_PROVIDER: Final[Mapping[str, str]] = MappingProxyType(
    {"tavily": "search_depth", "linkup": "depth"}
)


def _provider_usage(
    optional_params: Mapping[str, object] | None,
    usage_param: str,
) -> tuple[Mapping[str, object], ...] | None:
    params: Final = optional_params if optional_params is not None else EMPTY_OPTIONAL_PARAMS
    raw_usage: Final[object] = params.get(usage_param)
    try:
        return PROVIDER_USAGE_ADAPTER.validate_python(raw_usage)
    except ValidationError:
        return None


def _registered_at_runtime(entry: str) -> bool:
    """Whether `entry`'s live price differs from the catalog as loaded, i.e. `register_model` changed it."""
    live: Final[object] = litellm.model_cost.get(entry)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  # registry is dynamically typed
    return live != GetModelCostMap.loaded_model_cost_map().get(entry)  # pyright: ignore[reportUnknownVariableType]  # `live` comes from the untyped registry


def _tavily_auto_depth(response: SearchResponse | None) -> str | None:
    metadata: Final = as_str_mapping(attribute_or_none(response, "provider_metadata"))
    auto_parameters: Final = as_str_mapping(metadata.get("auto_parameters")) if metadata is not None else None
    depth: Final = auto_parameters.get("search_depth") if auto_parameters is not None else None
    return depth if isinstance(depth, str) else None


def _pricing_model(
    model: str,
    custom_llm_provider: str | None,
    optional_params: Mapping[str, object] | None,
    search_response: SearchResponse | None,
) -> str:
    """Use the priced mode entry for a generic search model. For Tavily, fall back to the reported depth when
    no depth was requested. A price registered at runtime on the generic entry keeps applying until the mode
    entry is registered too."""
    mode_param: Final = SEARCH_MODE_PARAM_BY_PROVIDER.get(custom_llm_provider or "")
    generic_model: Final = f"{custom_llm_provider}/search"
    if mode_param is None or model not in (generic_model, f"{custom_llm_provider}/{generic_model}"):
        return model
    requested_mode: Final = (optional_params if optional_params is not None else EMPTY_OPTIONAL_PARAMS).get(mode_param)
    mode: Final = (
        _tavily_auto_depth(search_response)
        if requested_mode is None and custom_llm_provider == "tavily"
        else requested_mode
    )
    mode_model: Final = f"{custom_llm_provider}/search-{mode}"
    priced: Final = mode_model in litellm.model_cost  # pyright: ignore[reportUnknownMemberType]  # registry is dynamically typed
    if not priced or (_registered_at_runtime(generic_model) and not _registered_at_runtime(mode_model)):
        return model
    return mode_model


def _tiered_cost_per_query(model_info: ModelInfo, optional_params: Mapping[str, object] | None) -> float:
    """The `max_results_range` tier price, which the Rust core does not price yet."""
    tiered_pricing: Final = model_info.get("tiered_pricing") or []
    max_results: Final = (optional_params or {}).get("max_results", 10)
    return next(
        (
            tier["input_cost_per_query"]
            for tier in tiered_pricing
            if tier["max_results_range"][0] <= max_results <= tier["max_results_range"][1]
        ),
        tiered_pricing[-1]["input_cost_per_query"],
    )


def _parallel_ai_billed(optional_params: Mapping[str, object]) -> tuple[str, int, int]:
    from litellm.llms.parallel_ai.search.cost_calculator import (
        PARALLEL_AI_USAGE_PARAM,
        parallel_ai_pricing_model,
        parallel_ai_search_counts,
    )

    requests, additional_results = parallel_ai_search_counts(
        optional_params, _provider_usage(optional_params, PARALLEL_AI_USAGE_PARAM)
    )
    return parallel_ai_pricing_model(optional_params), requests, additional_results


def search_cost(
    model: str,
    custom_llm_provider: str | None,
    number_of_queries: int,
    optional_params: Mapping[str, object] | None,
    adjust: AdjustFacts | None = None,
    configured_cost_per_query: float | None = None,
    *,
    search_response: SearchResponse | None = None,
) -> NativeReceipt | float:
    """The receipt of a search, or for a row priced by requested results its cost before discount and margin."""
    pricing_model, requests, additional_results = (
        _parallel_ai_billed(optional_params if optional_params is not None else EMPTY_OPTIONAL_PARAMS)
        if custom_llm_provider == "parallel_ai" and configured_cost_per_query is None
        else (_pricing_model(model, custom_llm_provider, optional_params, search_response), number_of_queries, 0)
    )
    model_info: Final = (
        None
        if configured_cost_per_query is not None
        else get_model_info(model=pricing_model, custom_llm_provider=custom_llm_provider)
    )
    outcome: Final = price_units(
        SearchDocument(
            kind="search",
            model=pricing_model,
            provider=custom_llm_provider,
            queries=requests,
            additional_results=additional_results,
            adjust=adjust,
        ),
        tables=({"input_cost_per_query": configured_cost_per_query},) if configured_cost_per_query is not None else (),
        catalog=model_info,
    )
    if isinstance(outcome, NativeReceipt):
        return outcome
    if outcome.reason == "not_ported" and model_info is not None:
        return number_of_queries * _tiered_cost_per_query(model_info, optional_params)
    return 0.0


def search_provider_cost_per_query(
    model: str,
    custom_llm_provider: str | None = None,
    number_of_queries: int = 1,
    optional_params: Mapping[str, object] | None = None,
    *,
    search_response: SearchResponse | None = None,
) -> tuple[float, float]:
    """
    Calculate cost for search-only providers.

    Returns (input_cost, output_cost) where input_cost = queries * cost_per_query
    Supports tiered pricing based on max_results parameter, and per-mode price entries selected by the
    provider's mode parameter (Tavily `search_depth`, Linkup `depth`).

    Args:
        model: Model name (e.g., "exa_ai/search", "tavily/search")
        custom_llm_provider: Provider name (e.g., "exa_ai", "tavily")
        number_of_queries: Number of search queries performed (default: 1)
        optional_params: Optional parameters including max_results for tiered pricing
        search_response: Provider response for pricing automatically selected search depth

    Returns:
        Tuple of (input_cost, output_cost) where output_cost is always 0.0
    """
    cost: Final = search_cost(
        model, custom_llm_provider, number_of_queries, optional_params, search_response=search_response
    )
    return (cost.prompt_cost, 0.0) if isinstance(cost, NativeReceipt) else (cost, 0.0)
