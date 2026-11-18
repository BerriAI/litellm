from collections.abc import Mapping
from typing import Final

from pydantic import TypeAdapter

EXA_AI_INCLUDED_RESULTS: Final = 10
EXA_AI_ADDITIONAL_RESULT_COST: Final = 0.001
EXA_AI_SUMMARY_COST_PER_PAGE: Final = 0.001
EXA_AI_TYPES_WITHOUT_SUMMARIES: Final = frozenset({"lite"})
_CONTENTS: Final = TypeAdapter(Mapping[str, object])


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


def _requests_summaries(optional_params: Mapping[str, object]) -> bool:
    if optional_params.get("type") in EXA_AI_TYPES_WITHOUT_SUMMARIES:
        return False
    contents: Final = optional_params.get("contents")
    return isinstance(contents, Mapping) and _CONTENTS.validate_python(contents).get("summary") not in (None, False)


def exa_ai_list_price(base_price: float, optional_params: Mapping[str, object]) -> float:
    """
    List price of one Exa /search request, for responses that carry no `costDollars.total`.

    Multi-query searches are joined into a single request, so this prices one request:
    the search type's base price, a per-result charge above the results that price includes,
    and a per-page charge for each requested result when summaries are requested on a type that offers them.
    """
    results: Final = _requested_results(optional_params)
    additional_results: Final = max(results - EXA_AI_INCLUDED_RESULTS, 0)
    summarized_pages: Final = results if _requests_summaries(optional_params) else 0
    return (
        base_price
        + additional_results * EXA_AI_ADDITIONAL_RESULT_COST
        + summarized_pages * EXA_AI_SUMMARY_COST_PER_PAGE
    )
