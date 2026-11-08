"""
Cost calculation for search providers.
"""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

from pydantic import TypeAdapter, ValidationError

import litellm
from litellm.litellm_core_utils.get_model_cost_map import GetModelCostMap
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


def _pricing_model(
    model: str,
    custom_llm_provider: str | None,
    optional_params: Mapping[str, object] | None,
) -> str:
    """`<provider>/search-<mode>` when `model` is the generic `<provider>/search` entry (bare or provider-prefixed) and
    the request's mode parameter names a priced entry, else `model`. A price registered at runtime on the generic entry
    keeps applying until the mode entry is registered too."""
    mode_param: Final = SEARCH_MODE_PARAM_BY_PROVIDER.get(custom_llm_provider or "")
    generic_model: Final = f"{custom_llm_provider}/search"
    if mode_param is None or model not in (generic_model, f"{custom_llm_provider}/{generic_model}"):
        return model
    mode: Final = (optional_params if optional_params is not None else EMPTY_OPTIONAL_PARAMS).get(mode_param)
    mode_model: Final = f"{custom_llm_provider}/search-{mode}"
    priced: Final = mode_model in litellm.model_cost  # pyright: ignore[reportUnknownMemberType]  # registry is dynamically typed
    if not priced or (_registered_at_runtime(generic_model) and not _registered_at_runtime(mode_model)):
        return model
    return mode_model


def search_provider_cost_per_query(
    model: str,
    custom_llm_provider: str | None = None,
    number_of_queries: int = 1,
    optional_params: Mapping[str, object] | None = None,
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

    Returns:
        Tuple of (input_cost, output_cost) where output_cost is always 0.0
    """
    if custom_llm_provider == "parallel_ai":
        from litellm.llms.parallel_ai.search.cost_calculator import (
            PARALLEL_AI_USAGE_PARAM,
            parallel_ai_search_cost,
        )

        input_cost: Final = parallel_ai_search_cost(
            optional_params=optional_params if optional_params is not None else EMPTY_OPTIONAL_PARAMS,
            usage=_provider_usage(optional_params, PARALLEL_AI_USAGE_PARAM),
        )
        return (input_cost, 0.0)

    model_info: Final = get_model_info(
        model=_pricing_model(model, custom_llm_provider, optional_params),
        custom_llm_provider=custom_llm_provider,
    )

    # Check for tiered pricing (e.g., Exa AI based on max_results)
    tiered_pricing: Final = model_info.get("tiered_pricing")
    if tiered_pricing and isinstance(tiered_pricing, list):
        max_results: Final = (optional_params or {}).get("max_results", 10)  # default 10 results
        cost_per_query = 0.0

        for tier in tiered_pricing:
            range_min, range_max = tier["max_results_range"]
            if range_min <= max_results <= range_max:
                cost_per_query = tier["input_cost_per_query"]
                break
        else:
            # Fallback to highest tier if out of range
            cost_per_query = tiered_pricing[-1]["input_cost_per_query"]
    else:
        # Simple flat rate
        cost_per_query = float(model_info.get("input_cost_per_query") or 0.0)

    total_cost: Final = number_of_queries * cost_per_query
    return (total_cost, 0.0)  # (input_cost, output_cost)
