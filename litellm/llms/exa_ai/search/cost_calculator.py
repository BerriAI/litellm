from collections.abc import Mapping
from types import MappingProxyType
from typing import Annotated, Final

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from litellm.constants import DEFAULT_MODEL_COST_MAP_URL, INTERNAL_KWARG_PREFIX
from litellm.litellm_core_utils.get_model_cost_map import (
    AliasedCostEntry,
    GetModelCostMap,
    get_model_cost_map_source_info,
)
from litellm.llms.base_llm.search.transformation import SearchResponse

EXA_AI_COST_DOLLARS_PARAM: Final = f"{INTERNAL_KWARG_PREFIX}exa_ai_cost_dollars"
_PRICE_FIELDS: Final = frozenset({"input_cost_per_query", "tiered_pricing"})
_ENTRY: Final = TypeAdapter(Mapping[str, object])
EXA_AI_CATALOG_PRICES: Final[Mapping[str, float]] = MappingProxyType(
    {
        "exa_ai/search": 0.007,
        "exa_ai/search-lite": 0.004,
        "exa_ai/search-instant": 0.004,
        "exa_ai/search-deep": 0.012,
        "exa_ai/search-deep-lite": 0.012,
        "exa_ai/search-deep-reasoning": 0.015,
    }
)


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


def exa_ai_reported_cost(params: Mapping[str, object], response: object = None) -> float | None:
    """The `costDollars.total` Exa reported for the request, when it is a finite non-negative number."""
    try:
        payload: Final = response.model_extra if isinstance(response, SearchResponse) else response
        metadata: Final = _ENTRY.validate_python(payload or {}).get("provider_metadata")
        receipt: Final = _ENTRY.validate_python(metadata or {}).get("costDollars")
        return _ExaCostDollars.model_validate(
            receipt if receipt is not None else params.get(EXA_AI_COST_DOLLARS_PARAM)
        ).total
    except ValidationError:
        return None


def _legacy_tiers(tiers: object) -> bool:
    try:
        return _TIERS.validate_python(tiers) == EXA_AI_LEGACY_TIERS
    except ValidationError:
        return False


def exa_ai_price_fields(model: str, entry: object) -> frozenset[str]:
    if not isinstance(entry, Mapping):
        return frozenset()
    fields: Final = _ENTRY.validate_python(entry)
    registered: Final[frozenset[str]] = (
        entry.registered_fields & _PRICE_FIELDS if isinstance(entry, AliasedCostEntry) else frozenset()
    )
    if registered:
        return registered
    source: Final = get_model_cost_map_source_info()
    if source["source"] == "remote" and source["url"] == DEFAULT_MODEL_COST_MAP_URL:
        loaded: Final = GetModelCostMap.loaded_model_cost_map().get(model, {})
        return frozenset(
            name for name in _PRICE_FIELDS if fields.get(name) is not None and fields[name] != loaded.get(name)
        )
    return frozenset(
        name
        for name in _PRICE_FIELDS
        if fields.get(name) is not None
        and (
            fields[name] != EXA_AI_CATALOG_PRICES.get(model)
            if name == "input_cost_per_query"
            else not _legacy_tiers(fields[name])
        )
    )


EXA_AI_INCLUDED_RESULTS: Final = 10
EXA_AI_ADDITIONAL_RESULT_COST: Final = 0.001
EXA_AI_SUMMARY_COST_PER_PAGE: Final = 0.001
EXA_AI_TYPES_WITHOUT_SUMMARIES: Final = frozenset({"lite"})


def _non_negative_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def exa_ai_requested_results(optional_params: Mapping[str, object]) -> int:
    return next(
        (
            requested
            for key in ("max_results", "numResults")
            if (requested := _non_negative_int(optional_params.get(key))) is not None
        ),
        EXA_AI_INCLUDED_RESULTS,
    )


def _requests_summaries(optional_params: Mapping[str, object]) -> bool:
    if optional_params.get("type") in EXA_AI_TYPES_WITHOUT_SUMMARIES:
        return False
    contents: Final = optional_params.get("contents")
    return isinstance(contents, Mapping) and _ENTRY.validate_python(contents).get("summary") not in (None, False)


def exa_ai_list_price(base_price: float, optional_params: Mapping[str, object]) -> float:
    """
    List price of one Exa /search request, for responses that carry no `costDollars.total`.

    Multi-query searches are joined into a single request, so this prices one request:
    the search type's base price, a per-result charge above the results that price includes,
    and a per-page charge for each requested result when summaries are requested on a type that offers them.
    """
    results: Final = exa_ai_requested_results(optional_params)
    additional_results: Final = max(results - EXA_AI_INCLUDED_RESULTS, 0)
    summarized_pages: Final = results if _requests_summaries(optional_params) else 0
    return (
        base_price
        + additional_results * EXA_AI_ADDITIONAL_RESULT_COST
        + summarized_pages * EXA_AI_SUMMARY_COST_PER_PAGE
    )
