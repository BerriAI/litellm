from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

from litellm.utils import get_model_info

EXA_AI_INCLUDED_RESULTS: Final = 10
EXA_AI_ADDITIONAL_RESULT_COST: Final = 0.001
EXA_AI_STANDARD_SEARCH_MODEL: Final = "exa_ai/search"
EXA_AI_PRICING_MODEL_BY_TYPE: Final[Mapping[str, str]] = MappingProxyType(
    {
        "deep-lite": "exa_ai/search-deep",
        "deep": "exa_ai/search-deep",
        "deep-reasoning": "exa_ai/search-deep-reasoning",
    }
)


def _non_negative_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        return None
    return float(value)


def _requested_results(optional_params: Mapping[str, object]) -> int:
    for key in ("max_results", "numResults"):
        requested = _non_negative_number(optional_params.get(key))
        if requested is not None:
            return int(requested)
    return EXA_AI_INCLUDED_RESULTS


def _base_request_cost(search_type: object) -> float:
    pricing_model: Final = (
        EXA_AI_PRICING_MODEL_BY_TYPE.get(search_type, EXA_AI_STANDARD_SEARCH_MODEL)
        if isinstance(search_type, str)
        else EXA_AI_STANDARD_SEARCH_MODEL
    )
    model_info: Final = get_model_info(model=pricing_model, custom_llm_provider="exa_ai")
    return float(model_info.get("input_cost_per_query") or 0.0)


def exa_ai_list_price(optional_params: Mapping[str, object]) -> float:
    """
    List price of one Exa /search request, for responses that carry no `costDollars.total`.

    Multi-query searches are joined into a single request, so this prices one request:
    the per-type base price plus a per-result charge above the results that price includes.
    """
    additional_results: Final = max(_requested_results(optional_params) - EXA_AI_INCLUDED_RESULTS, 0)
    return _base_request_cost(optional_params.get("type")) + additional_results * EXA_AI_ADDITIONAL_RESULT_COST
