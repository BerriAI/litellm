from collections.abc import Mapping
from typing import Final

EXA_AI_INCLUDED_RESULTS: Final = 10
EXA_AI_ADDITIONAL_RESULT_COST: Final = 0.001


def _non_negative_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _requested_results(optional_params: Mapping[str, object]) -> int:
    for key in ("max_results", "numResults"):
        requested = _non_negative_int(optional_params.get(key))
        if requested is not None:
            return requested
    return EXA_AI_INCLUDED_RESULTS


def exa_ai_list_price(base_price: float, optional_params: Mapping[str, object]) -> float:
    """
    List price of one Exa /search request, for responses that carry no `costDollars.total`.

    Multi-query searches are joined into a single request, so this prices one request:
    the search type's base price plus a per-result charge above the results that price includes.
    """
    additional_results: Final = max(_requested_results(optional_params) - EXA_AI_INCLUDED_RESULTS, 0)
    return base_price + additional_results * EXA_AI_ADDITIONAL_RESULT_COST
