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
