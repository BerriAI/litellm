"""Which deployments accrue PTU flat cost, and what that costs them per token.

Reserved provisioned throughput is billed by the hour whether or not requests are sent, so
a deployment that accrues flat cost must not also bill per token. The two halves live here
together because they have to agree: a deployment the rollup declines to charge but the
router prices at zero serves its traffic for free.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Final

from litellm.secret_managers.main import get_secret_bool
from litellm.types.router import ModelInfo
from litellm.types.utils import CustomPricingLiteLLMParams, MirroredPricingParams

PTU_COST_ATTRIBUTION_ENV_VAR: Final = "LITELLM_ENABLE_PTU_COST_ATTRIBUTION"


def is_ptu_cost_attribution_enabled() -> bool:
    """Whether PTU flat-cost attribution is turned on for this process."""
    return get_secret_bool(PTU_COST_ATTRIBUTION_ENV_VAR, False) is True


PTU_ZEROED_PRICING_FIELDS: Final = tuple(f for f in MirroredPricingParams.model_fields if f != "tiered_pricing") + (
    "cache_creation_input_token_cost_above_1hr",
    "cache_creation_input_token_cost_above_200k_tokens",
    "cache_read_input_token_cost_above_200k_tokens",
)
# tiered_pricing is emptied rather than zeroed: its tiers outrank the zeros written beside
# them, so a zero here would leave the cost map's tiers billing the traffic the reserved
# capacity already covers.
PTU_EMPTIED_PRICING_FIELDS: Final = frozenset(("tiered_pricing",))
# search_context_cost_per_query holds its rates in a table keyed by context size, and an
# absent table means the provider's own default rather than free, so it is zeroed in place
# and written on every PTU deployment rather than only where a table is already stored.
PTU_ZEROED_TABLE_FIELDS: Final = frozenset(("search_context_cost_per_query",))
SEARCH_CONTEXT_SIZES: Final = ("search_context_size_low", "search_context_size_medium", "search_context_size_high")
# Rate fields only. CustomPricingLiteLLMParams also carries settings that are not charges,
# and zeroing one of those would destroy the deployment's configuration rather than stop a
# charge.
CUSTOM_PRICING_FIELDS: Final = frozenset(f for f in CustomPricingLiteLLMParams.model_fields if "cost" in f)
PTU_ZEROED_PRICING: Final[Mapping[str, float | tuple[()] | Mapping[str, float]]] = MappingProxyType(
    {
        **dict.fromkeys(PTU_ZEROED_PRICING_FIELDS, 0.0),
        **dict.fromkeys(PTU_EMPTIED_PRICING_FIELDS, ()),
        **dict.fromkeys(PTU_ZEROED_TABLE_FIELDS, MappingProxyType(dict.fromkeys(SEARCH_CONTEXT_SIZES, 0.0))),
    }
)


@dataclass(frozen=True, slots=True)
class PTUTerms:
    """The reservation a deployment declares, once every field has been validated."""

    team_id: str
    ptu_count: int
    cost_per_ptu_per_hour: float
    effective_from: datetime
    effective_to: datetime | None


def _to_utc(parsed: datetime) -> datetime:
    """``parsed`` as UTC, reading a naive value as UTC rather than local time."""
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)


def _as_utc(value: object) -> datetime | None:
    """A model_info datetime as UTC, parsing an ISO string, else None."""
    if isinstance(value, datetime):
        return _to_utc(value)
    if not isinstance(value, str):
        return None
    try:
        return _to_utc(datetime.fromisoformat(value.replace("Z", "+00:00")))
    except ValueError:
        return None


def ptu_terms(model_info: Mapping[str, object]) -> PTUTerms | None:
    """The reservation this deployment accrues flat cost for, else None.

    A start is required rather than inferred because flat cost accrues from it, and a
    present but unparseable bound would read as no bound and widen the window to the whole
    day, so either one leaves the deployment unpriced until the config is fixed.
    """
    ptu_count: Final = model_info.get("ptu_count")
    cost_per_hour: Final = model_info.get("cost_per_ptu_per_hour")
    team_id: Final = model_info.get("team_id")
    if ptu_count is None or cost_per_hour is None or not team_id:
        return None
    try:
        ptu_count_int: Final = int(ptu_count)
        cost_per_hour_float: Final = float(cost_per_hour)
    except (TypeError, ValueError, OverflowError):
        return None
    if not 0 < ptu_count_int <= ModelInfo.MAX_PTU_COUNT:
        return None
    if not 0 <= cost_per_hour_float <= ModelInfo.MAX_COST_PER_PTU_PER_HOUR:
        return None

    raw_from: Final = model_info.get("ptu_effective_from")
    raw_to: Final = model_info.get("ptu_effective_to")
    effective_from: Final = _as_utc(raw_from)
    effective_to: Final = _as_utc(raw_to)
    if effective_from is None or (raw_to is not None and effective_to is None):
        return None
    if effective_to is not None and effective_to <= effective_from:
        return None
    return PTUTerms(
        team_id=str(team_id),
        ptu_count=ptu_count_int,
        cost_per_ptu_per_hour=cost_per_hour_float,
        effective_from=effective_from,
        effective_to=effective_to,
    )


def zeroed_ptu_pricing(
    model_info: Mapping[str, object], declared: Mapping[str, object]
) -> Mapping[str, float | tuple[()] | Mapping[str, float]] | None:
    """The pricing a deployment accruing flat cost must carry, else None.

    Both conditions hold or nothing is zeroed. Without the flag no flat cost accrues, so
    zeroing would leave the deployment serving for free with nothing charged in its place,
    which is what an SDK user who happens to carry ptu_count would otherwise get. The terms
    are checked first only because they are a few dict reads, while the flag can resolve
    through a configured secret manager, and this runs for every deployment registered.

    Any further rate the deployment itself declares is zeroed alongside the standing set,
    since one left standing bills the traffic the reserved capacity already paid for.
    """
    if ptu_terms(model_info) is None:
        return None
    if not is_ptu_cost_attribution_enabled():
        return None
    return MappingProxyType(
        {
            **PTU_ZEROED_PRICING,
            **dict.fromkeys(
                CUSTOM_PRICING_FIELDS.intersection(declared)
                .difference(PTU_ZEROED_TABLE_FIELDS)
                .difference(PTU_EMPTIED_PRICING_FIELDS),
                0.0,
            ),
        }
    )
