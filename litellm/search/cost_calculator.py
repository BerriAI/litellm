"""
Cost calculation for search providers.
"""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

from pydantic import TypeAdapter, ValidationError

from litellm.utils import get_model_info

PROVIDER_USAGE_ADAPTER: Final[TypeAdapter[tuple[Mapping[str, object], ...]]] = TypeAdapter(
    tuple[Mapping[str, object], ...]
)
EMPTY_OPTIONAL_PARAMS: Final[Mapping[str, object]] = MappingProxyType({})


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


def search_provider_cost_per_query(
    model: str,
    custom_llm_provider: str | None = None,
    number_of_queries: int = 1,
    optional_params: Mapping[str, object] | None = None,
) -> tuple[float, float]:
    """
    Calculate cost for search-only providers.

    Returns (input_cost, output_cost) where input_cost = queries * cost_per_query
    Supports tiered pricing based on max_results parameter.

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

    model_info: Final = get_model_info(model=model, custom_llm_provider=custom_llm_provider)

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
