from collections.abc import Mapping
from typing import Annotated, Final

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from litellm.constants import INTERNAL_KWARG_PREFIX
from litellm.litellm_core_utils.get_model_cost_map import AliasedCostEntry

EXA_AI_COST_DOLLARS_PARAM: Final = f"{INTERNAL_KWARG_PREFIX}exa_ai_cost_dollars"
_PRICE_FIELDS: Final = frozenset({"input_cost_per_query", "tiered_pricing"})
_ENTRY: Final = TypeAdapter(Mapping[str, object])


class _ExaCostDollars(BaseModel):
    model_config = ConfigDict(frozen=True)

    total: Annotated[float, Field(strict=True, ge=0, allow_inf_nan=False)]


class ExaAITier(BaseModel):
    model_config = ConfigDict(frozen=True)

    input_cost_per_query: float
    max_results_range: tuple[int, int]


_TIERS: Final = TypeAdapter(tuple[ExaAITier, ...])

EXA_AI_LEGACY_TIERS: Final = (
    ExaAITier(input_cost_per_query=0.005, max_results_range=(0, 25)),
    ExaAITier(input_cost_per_query=0.025, max_results_range=(26, 100)),
)
"""The tiers LiteLLM ships on `exa_ai/search`. Only older releases bill them, so they never change."""


def exa_ai_reported_cost(params: Mapping[str, object]) -> float | None:
    """The `costDollars.total` Exa reported for the request, when it is a finite non-negative number."""
    try:
        return _ExaCostDollars.model_validate(params.get(EXA_AI_COST_DOLLARS_PARAM)).total
    except ValidationError:
        return None


def _legacy_tiers(tiers: object) -> bool:
    try:
        return _TIERS.validate_python(tiers) == EXA_AI_LEGACY_TIERS
    except ValidationError:
        return False


def exa_ai_price_set(entry: object) -> bool:
    """Whether an Exa price entry holds a price someone set, which replaces the cost Exa reports. LiteLLM ships
    Exa with its legacy tiers only, so anything else there was set by `register_model()`, an admin's own cost map,
    or an edited bundled file."""
    if isinstance(entry, AliasedCostEntry) and entry.registered_fields & _PRICE_FIELDS:
        return True
    if not isinstance(entry, Mapping):
        return False
    fields: Final = _ENTRY.validate_python(entry)
    tiers: Final = fields.get("tiered_pricing")
    return fields.get("input_cost_per_query") is not None or (tiers is not None and not _legacy_tiers(tiers))


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
        requested: Final = _non_negative_int(optional_params.get(key))
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
