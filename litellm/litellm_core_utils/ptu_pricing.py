"""Which deployments accrue PTU flat cost, and what that costs them per token.

Reserved provisioned throughput is billed by the hour whether or not requests are sent, so
a deployment that accrues flat cost must not also bill per token. The two halves live here
together because they have to agree: a deployment the rollup declines to charge but the
router prices at zero serves its traffic for free.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime, time, timezone
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
    "google_maps_grounding_cost_per_query",
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
    """A model_info datetime as UTC, parsing an ISO string, else None.

    An unquoted ``2027-01-01`` in config.yaml is loaded as a ``date``, not a string, and a
    reservation bound that fails to parse takes the whole deployment out of PTU handling,
    so the day is read as its opening midnight rather than discarded. ``datetime`` derives
    from ``date``, so it has to be matched first.
    """
    if isinstance(value, datetime):
        return _to_utc(value)
    if isinstance(value, date):
        return datetime.combine(value, time.min, tzinfo=timezone.utc)
    if not isinstance(value, str):
        return None
    try:
        return _to_utc(datetime.fromisoformat(value.replace("Z", "+00:00")))
    except ValueError:
        return None


def _named(reason: str, model_name: str | None) -> str:
    """The reason on its own for a caller that already has the deployment in hand, else named."""
    return reason if model_name is None else f"PTU configuration on model '{model_name}' is invalid: {reason}"


def ptu_identity_error(
    *, declared_id: str | None, taken: bool, current_id: str | None = None, model_name: str | None = None
) -> str | None:
    """Why this config-declared reservation cannot be identified, else None.

    A deployment declared in config.yaml is otherwise keyed by a hash of its resolved
    ``litellm_params``, so rotating a credential or editing an endpoint mints a second
    identity and the reservation is charged again under it. The flat cost is keyed by that
    id, and a charge already written is never retracted, so the duplicate is permanent.

    ``current_id`` is what the deployment is keyed by today. Naming it is the difference
    between an operator carrying their history forward and an operator inventing a fresh
    id, which starts a second identity beside the charges already written.
    """
    if not declared_id:
        return _named(
            "model_info.id is required when PTU fields are set. Without one the deployment is "
            "identified by a hash of its litellm_params, so rotating a credential bills the "
            "reservation a second time under the new identity. Set it to the id this deployment "
            f"already uses, {current_id or 'shown by GET /model/info'}, so the flat cost already "
            "written stays under one identity; any other value starts a second one",
            model_name,
        )
    if taken:
        return _named(
            f"model_info.id '{declared_id}' is declared on more than one deployment. Each would key "
            "the same flat-cost row, so one reservation would go unbilled",
            model_name,
        )
    return None


PTU_MODEL_INFO_FIELDS: Final = ("ptu_count", "cost_per_ptu_per_hour", "ptu_effective_from", "ptu_effective_to")


def declares_ptu(model_info: Mapping[str, object]) -> bool:
    """Whether any PTU field is set here, including one too malformed to charge."""
    return any(model_info.get(field) is not None for field in PTU_MODEL_INFO_FIELDS)


def ptu_config_error(model_info: Mapping[str, object], *, model_name: str | None = None) -> str | None:
    """Why this PTU configuration cannot be honoured, else None.

    Both the model endpoints and config.yaml registration ask this, so a deployment that
    one refuses is refused by the other for the same stated reason.

    Window ordering is checked before the count/rate gate. A patch that touches only one end
    of the window carries no count or rate, so leaving the order to that gate would let an
    inverted window reach the row; the next load then fails to parse it and drops the
    deployment out of the router, where no further patch can repair it.
    """
    effective_from: Final = _as_utc(model_info.get("ptu_effective_from"))
    effective_to: Final = _as_utc(model_info.get("ptu_effective_to"))
    if effective_from is not None and effective_to is not None and effective_to <= effective_from:
        return _named("ptu_effective_to must be after ptu_effective_from", model_name)

    has_count: Final = model_info.get("ptu_count") is not None
    has_rate: Final = model_info.get("cost_per_ptu_per_hour") is not None
    if not has_count and not has_rate:
        return None
    if has_count != has_rate:
        return _named("ptu_count and cost_per_ptu_per_hour must be set together", model_name)
    if effective_from is None:
        return _named(
            "ptu_effective_from is required when PTU fields are set. Flat cost accrues from that "
            "instant, so without it the start would have to be inferred and a deployment configured "
            "today could be billed for days it did not exist",
            model_name,
        )
    if not model_info.get("team_id"):
        return _named("team_id is required when PTU fields are set (one model maps to one team)", model_name)
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
