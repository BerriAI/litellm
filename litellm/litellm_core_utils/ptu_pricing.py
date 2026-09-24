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
from litellm.types.utils import AzureSpillover, CustomPricingLiteLLMParams, MirroredPricingParams

PTU_COST_ATTRIBUTION_ENV_VAR: Final = "LITELLM_ENABLE_PTU_COST_ATTRIBUTION"
AZURE_SPILLOVER_HEADER: Final = "x-ms-is-spilled-over"
AZURE_SPILLOVER_FROM_HEADER: Final = "x-ms-spillover-from-deployment"


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
    """The reservation a deployment declares, once every field has been validated.

    ``shares`` maps every team the capacity is attributed to onto its PTUs and adds up to
    ``ptu_count``: a deployment declaring a single ``team_id`` holds the whole count under it.
    """

    shares: Mapping[str, int]
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


PTU_MODEL_INFO_FIELDS: Final = (
    "ptu_count",
    "cost_per_ptu_per_hour",
    "ptu_effective_from",
    "ptu_effective_to",
    "ptu_shares",
)


def parsed_ptu_shares(raw: object) -> Mapping[str, int] | None:
    """``ptu_shares`` as team id -> whole PTUs, else None when empty or any entry is unusable.

    A share is a count of reserved units, so it has to be a positive integer; ``bool`` is
    excluded because it is an ``int`` subclass and ``True`` would read as one PTU.
    """
    if not isinstance(raw, Mapping) or not raw:
        return None
    entries: Final = tuple((str(team_id), share) for team_id, share in raw.items())
    if any(
        not team_id or isinstance(share, bool) or not isinstance(share, int) or share <= 0 for team_id, share in entries
    ):
        return None
    return MappingProxyType(dict(entries))


def _declared_shares(model_info: Mapping[str, object], ptu_count: int) -> Mapping[str, int] | None:
    """Who holds the capacity: the single ``team_id`` holding all of it, or the ``ptu_shares``
    that add up to it, else None."""
    team_id: Final = model_info.get("team_id")
    raw_shares: Final = model_info.get("ptu_shares")
    if team_id and raw_shares is None:
        return MappingProxyType({str(team_id): ptu_count})
    if team_id or raw_shares is None:
        return None
    shares: Final = parsed_ptu_shares(raw_shares)
    if shares is None or sum(shares.values()) != ptu_count:
        return None
    return shares


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
    return _ptu_holder_error(model_info, model_name)


def _ptu_holder_error(model_info: Mapping[str, object], model_name: str | None) -> str | None:
    """Why the teams this capacity is attributed to cannot be read, else None.

    The shares have to add up to the count exactly: a shortfall would leave PTU-hours the
    provider bills attributed to nobody, and a surplus would attribute capacity that was
    never reserved.
    """
    team_id: Final = model_info.get("team_id")
    raw_shares: Final = model_info.get("ptu_shares")
    if team_id and raw_shares is not None:
        return _named(
            "team_id and ptu_shares cannot both be set; ptu_shares lists every team the capacity is split across",
            model_name,
        )
    if not team_id and raw_shares is None:
        return _named("team_id or ptu_shares is required when PTU fields are set", model_name)
    if raw_shares is None:
        return None
    shares: Final = parsed_ptu_shares(raw_shares)
    if shares is None:
        return _named("ptu_shares must map at least one team_id to a positive whole number of PTUs", model_name)
    try:
        ptu_count: Final = int(str(model_info.get("ptu_count")))
    except ValueError:
        return None
    allocated: Final = sum(shares.values())
    if allocated != ptu_count:
        return _named(f"ptu_shares must add up to ptu_count ({allocated} of {ptu_count} allocated)", model_name)
    return None


def ptu_terms(model_info: Mapping[str, object]) -> PTUTerms | None:
    """The reservation this deployment accrues flat cost for, else None.

    A start is required rather than inferred because flat cost accrues from it, and a
    present but unparseable bound would read as no bound and widen the window to the whole
    day, so either one leaves the deployment unpriced until the config is fixed.
    """
    ptu_count: Final = model_info.get("ptu_count")
    cost_per_hour: Final = model_info.get("cost_per_ptu_per_hour")
    if ptu_count is None or cost_per_hour is None:
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
    shares: Final = _declared_shares(model_info, ptu_count_int)
    if shares is None:
        return None
    return PTUTerms(
        shares=shares,
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


def is_spilled_over_ptu_request(
    model_info: Mapping[str, object],
    response_headers: Mapping[str, object] | None,
    additional_headers: Mapping[str, object] | None,
) -> bool:
    """Whether Azure served this request from pay-as-you-go capacity, so the zeroed PTU rates must not apply."""
    if ptu_terms(model_info) is None:
        return False
    if not is_ptu_cost_attribution_enabled():
        return False
    return azure_spillover(response_headers, additional_headers) is not None


def azure_spillover(
    response_headers: Mapping[str, object] | None,
    additional_headers: Mapping[str, object] | None,
) -> AzureSpillover | None:
    """The spillover Azure reports in the response headers, else None."""
    for headers, prefix in (
        (response_headers, ""),
        (additional_headers, "llm_provider-"),
    ):
        if headers is None or str(headers.get(f"{prefix}{AZURE_SPILLOVER_HEADER}")).lower() != "true":
            continue
        return AzureSpillover(
            from_deployment=str(v) if (v := headers.get(f"{prefix}{AZURE_SPILLOVER_FROM_HEADER}")) is not None else None
        )
    return None
