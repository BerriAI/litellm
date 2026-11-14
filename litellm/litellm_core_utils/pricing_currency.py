"""Custom deployment prices configured in a source currency, normalized once into the USD rates every cost reader uses.

A deployment opts in with ``pricing_currency`` (plus ``usd_per_currency_unit`` for anything but USD) on its
``litellm_params`` or ``model_info``. The built deployment carries only USD prices. What the operator typed is kept on
``model_info.pricing_source``, and a rebuilt deployment keeps that record only while converting it reproduces exactly
the deployment's own price set, so the record can never misstate or add to the prices actually billed.
"""

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, Overflow, localcontext
from enum import StrEnum
from types import MappingProxyType
from typing import Final, Literal, Protocol

from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

from litellm.types.utils import (
    ABOVE_THRESHOLD_COST_KEY_PATTERN,
    MONETARY_PRICING_FIELDS,
    MirroredPricingParams,
    OffPeakPricing,
)


class PricingCurrency(StrEnum):
    USD = "USD"
    EUR = "EUR"
    HUF = "HUF"


PRICING_CURRENCY: Final = "pricing_currency"
USD_PER_CURRENCY_UNIT: Final = "usd_per_currency_unit"
PRICING_SOURCE: Final = "pricing_source"
USD_PRICING: Final = "usd_pricing"
PRICING_SNAPSHOT: Final = "pricing_snapshot"
CURRENCY_FIELDS: Final = frozenset({PRICING_CURRENCY, USD_PER_CURRENCY_UNIT})
PRICING_PROJECTION_KEYS: Final = frozenset({PRICING_SOURCE, USD_PRICING})

_TIER_RANGE_KEY: Final = "range"
_OFF_PEAK_RATE_KEYS: Final = frozenset(OffPeakPricing.__annotations__) - {"hours_utc", "windows", "weekday_timezone"}
_OFF_PEAK_SETTING_KEYS: Final = frozenset({"hours_utc", "windows", "weekday_timezone"})
_SEARCH_CONTEXT_RATE_KEYS: Final = frozenset(
    {"search_context_size_low", "search_context_size_medium", "search_context_size_high"}
)
_MIRRORED_FIELDS: Final = frozenset(MirroredPricingParams.model_fields)
_SUPPORTED_CURRENCIES: Final = frozenset(currency.value for currency in PricingCurrency)
_DECIMAL_PRECISION: Final = 34
_STR_MAPPING: Final = TypeAdapter(Mapping[str, object])
_ITEMS: Final = TypeAdapter(tuple[object, ...])


@dataclass(frozen=True, slots=True)
class PricingCurrencyError:
    kind: Literal[
        "unsupported_currency",
        "invalid_rate",
        "usd_rate_not_one",
        "conflicting_currency",
        "invalid_price",
        "out_of_range",
        "unsupported_field",
        "no_prices",
    ]
    message: str


class PricingSource(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    currency: PricingCurrency
    usd_per_currency_unit: str
    revision: str
    litellm_params: Mapping[str, object]
    model_info: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class NormalizedPricing:
    litellm_params: Mapping[str, object]
    model_info: Mapping[str, object]
    source: PricingSource | None


class _ReadsFields(Protocol):
    def get(self, key: str, default: object = None, /) -> object: ...


def needs_pricing_normalization(litellm_params: _ReadsFields, model_info: _ReadsFields) -> bool:
    return (
        any(litellm_params.get(field) is not None or model_info.get(field) is not None for field in CURRENCY_FIELDS)
        or model_info.get(PRICING_SOURCE) is not None
    )


def _decimal(value: object) -> Decimal | None:
    if isinstance(value, bool) or not isinstance(value, (int, float, str, Decimal)):
        return None
    try:
        parsed: Final = Decimal(str(value).strip())
    except InvalidOperation:
        return None
    return parsed if parsed.is_finite() else None


def _usd(field: str, value: object, rate: Decimal) -> float | PricingCurrencyError:
    amount: Final = _decimal(value)
    if amount is None or amount < 0:
        return PricingCurrencyError("invalid_price", f"{field} must be a finite, non-negative number, got {value!r}")
    with localcontext() as context:
        context.prec = _DECIMAL_PRECISION
        context.traps[Overflow] = False
        usd: Final = amount * rate
    converted: Final = float(usd) if usd.is_finite() else math.inf
    if not math.isfinite(converted) or (amount > 0 and converted == 0.0):
        return PricingCurrencyError(
            "out_of_range", f"{field} ({value!r}) converted to USD cannot be represented as a finite positive price"
        )
    return converted


def _str_mapping(value: object) -> Mapping[str, object] | None:
    if not isinstance(value, Mapping):
        return None
    try:
        return _STR_MAPPING.validate_python(value)
    except ValidationError:
        return None


def _first_error(values: Sequence[object]) -> PricingCurrencyError | None:
    return next((value for value in values if isinstance(value, PricingCurrencyError)), None)


def _converted_rates(
    field: str, value: object, rate: Decimal, rate_keys: frozenset[str], kept_keys: frozenset[str]
) -> object | PricingCurrencyError:
    """A mapping whose ``rate_keys`` hold prices and whose ``kept_keys`` hold settings; any other key is refused so
    a price this schema does not name can never stay in the source currency while billed as USD."""
    if value is None:
        return None
    section: Final = _str_mapping(value)
    if section is None:
        return PricingCurrencyError("invalid_price", f"{field} must be a mapping of prices, got {value!r}")
    unknown: Final = sorted(key for key in section if key not in rate_keys and key not in kept_keys)
    if unknown:
        return PricingCurrencyError(
            "unsupported_field", f"{field} holds {', '.join(unknown)}, which currency conversion does not price"
        )
    pairs: Final = tuple(
        (key, item if key in kept_keys or item is None else _usd(f"{field}.{key}", item, rate))
        for key, item in section.items()
    )
    error: Final = _first_error(tuple(item for _, item in pairs))
    return error if error is not None else dict(pairs)


def _converted_tiers(field: str, value: object, rate: Decimal) -> object | PricingCurrencyError:
    if value is None:
        return None
    if isinstance(value, (str, Mapping)) or not isinstance(value, Sequence):
        return PricingCurrencyError("invalid_price", f"{field} must be a list of tiers, got {value!r}")
    tiers: Final = tuple(
        _converted_rates(f"{field}[{index}]", tier, rate, MONETARY_PRICING_FIELDS, frozenset({_TIER_RANGE_KEY}))
        for index, tier in enumerate(_ITEMS.validate_python(value))
    )
    error: Final = _first_error(tiers)
    return error if error is not None else list(tiers)


def _converted_video_clips(field: str, value: object, rate: Decimal) -> object | PricingCurrencyError:
    if value is None:
        return None
    clips: Final = _str_mapping(value)
    if clips is None:
        return PricingCurrencyError("invalid_price", f"{field} must be a mapping of clip prices, got {value!r}")
    resolutions: Final = tuple(
        (key, _converted_clip_prices(f"{field}.{key}", item, rate)) for key, item in clips.items()
    )
    error: Final = _first_error(tuple(item for _, item in resolutions))
    return error if error is not None else dict(resolutions)


def _converted_clip_prices(field: str, value: object, rate: Decimal) -> object | PricingCurrencyError:
    durations: Final = _str_mapping(value)
    return _converted_rates(field, value, rate, frozenset(durations or ()), frozenset())


def _converted_price(field: str, value: object, rate: Decimal) -> object | PricingCurrencyError:
    match field:
        case "tiered_pricing":
            return _converted_tiers(field, value, rate)
        case "off_peak_pricing":
            return _converted_rates(field, value, rate, _OFF_PEAK_RATE_KEYS, _OFF_PEAK_SETTING_KEYS)
        case "output_cost_per_video_clip":
            return _converted_video_clips(field, value, rate)
        case "search_context_cost_per_query":
            return _converted_rates(field, value, rate, _SEARCH_CONTEXT_RATE_KEYS, frozenset())
        case "search_queries_cost_per_query" if isinstance(value, Mapping):
            return _converted_rates(field, value, rate, _SEARCH_CONTEXT_RATE_KEYS, frozenset())
        case _:
            return _usd(field, value, rate)


def _converted_prices(prices: Mapping[str, object], rate: Decimal) -> Mapping[str, object] | PricingCurrencyError:
    pairs: Final = tuple((field, _converted_price(field, value, rate)) for field, value in prices.items())
    error: Final = _first_error(tuple(value for _, value in pairs))
    return error if error is not None else MappingProxyType(dict(pairs))


def _priced_fields(section: Mapping[str, object]) -> Mapping[str, object]:
    return MappingProxyType(
        {field: value for field, value in section.items() if value is not None and field in MONETARY_PRICING_FIELDS}
    )


def _unshadowed_prices(info_prices: Mapping[str, object], params_prices: Mapping[str, object]) -> Mapping[str, object]:
    """``model_info`` prices that bill: a mirrored field ``litellm_params`` also sets is overwritten by it."""
    return MappingProxyType(
        {
            field: value
            for field, value in info_prices.items()
            if not (field in _MIRRORED_FIELDS and field in params_prices)
        }
    )


def _undeclared_price_error(section: Mapping[str, object]) -> PricingCurrencyError | None:
    undeclared: Final = sorted(
        field
        for field in section
        if field not in MONETARY_PRICING_FIELDS and ABOVE_THRESHOLD_COST_KEY_PATTERN.search(field) is not None
    )
    if not undeclared:
        return None
    return PricingCurrencyError(
        "unsupported_field",
        f"{', '.join(undeclared)} cannot be priced in a non-USD currency because it is not a declared price field",
    )


def _currency_setting(
    setting: str, litellm_params: Mapping[str, object], model_info: Mapping[str, object]
) -> object | PricingCurrencyError:
    from_params: Final = litellm_params.get(setting)
    from_info: Final = model_info.get(setting)
    if from_params is not None and from_info is not None and not _same_setting(from_params, from_info):
        return PricingCurrencyError(
            "conflicting_currency",
            f"{setting} is set to {from_params!r} in litellm_params and {from_info!r} in model_info",
        )
    return from_params if from_params is not None else from_info


def _same_setting(left: object, right: object) -> bool:
    left_number: Final = _decimal(left)
    right_number: Final = _decimal(right)
    if left_number is not None and right_number is not None:
        return left_number == right_number
    return left == right


def _currency(value: object) -> PricingCurrency | PricingCurrencyError:
    if isinstance(value, str) and value in _SUPPORTED_CURRENCIES:
        return PricingCurrency(value)
    supported: Final = ", ".join(sorted(_SUPPORTED_CURRENCIES))
    return PricingCurrencyError("unsupported_currency", f"pricing_currency must be one of {supported}, got {value!r}")


def _rate(currency: PricingCurrency, value: object) -> Decimal | PricingCurrencyError:
    if currency is PricingCurrency.USD:
        if value is None or _decimal(value) == 1:
            return Decimal(1)
        return PricingCurrencyError(
            "usd_rate_not_one", f"usd_per_currency_unit must be omitted or 1 for USD prices, got {value!r}"
        )
    rate: Final = _decimal(value) if value is not None else None
    if rate is None or rate <= 0:
        return PricingCurrencyError(
            "invalid_rate",
            f"usd_per_currency_unit must be a finite number above 0 for {currency.value} prices, got {value!r}",
        )
    return rate


def _without(section: Mapping[str, object], fields: frozenset[str]) -> Mapping[str, object]:
    return MappingProxyType({key: value for key, value in section.items() if key not in fields})


def pricing_revision(
    currency: PricingCurrency,
    rate: Decimal,
    litellm_params: Mapping[str, object],
    model_info: Mapping[str, object],
) -> str:
    canonical: Final = json.dumps(
        {
            "currency": currency.value,
            "rate": str(rate),
            "litellm_params": dict(litellm_params),
            "model_info": dict(model_info),
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def _parsed_source(raw: object) -> PricingSource | None:
    if raw is None:
        return None
    try:
        return PricingSource.model_validate(raw)
    except ValidationError:
        return None


def _verified_source(litellm_params: Mapping[str, object], model_info: Mapping[str, object]) -> PricingSource | None:
    """The deployment's ``pricing_source`` when converting it reproduces the deployment's complete price set: every
    recorded field is a declared price, and every price either section carries is one the record converts to."""
    source: Final = _parsed_source(model_info.get(PRICING_SOURCE))
    if source is None or source.currency is PricingCurrency.USD:
        return None
    if not source.litellm_params and not source.model_info:
        return None
    if any(field not in MONETARY_PRICING_FIELDS for field in (*source.litellm_params, *source.model_info)):
        return None
    rate: Final = _rate(source.currency, source.usd_per_currency_unit)
    if isinstance(rate, PricingCurrencyError):
        return None
    if source.revision != pricing_revision(source.currency, rate, source.litellm_params, source.model_info):
        return None
    params_usd: Final = _converted_prices(source.litellm_params, rate)
    info_usd: Final = _converted_prices(source.model_info, rate)
    if isinstance(params_usd, PricingCurrencyError) or isinstance(info_usd, PricingCurrencyError):
        return None
    shadowed: Final = frozenset(field for field in params_usd if field in _MIRRORED_FIELDS)
    info_prices: Final = _priced_fields(model_info)
    mirrors_match: Final = all(info_prices[field] == params_usd[field] for field in shadowed if field in info_prices)
    own_info_prices: Final = {field: value for field, value in info_prices.items() if field not in shadowed}
    return (
        source
        if mirrors_match
        and dict(_priced_fields(litellm_params)) == dict(params_usd)
        and own_info_prices == dict(info_usd)
        else None
    )


def _rebuilt(litellm_params: Mapping[str, object], model_info: Mapping[str, object]) -> NormalizedPricing:
    """A deployment without currency settings: its prices are already USD, and it keeps a ``pricing_source`` it
    carries, unchanged, only while that record reproduces them."""
    source: Final = _verified_source(litellm_params, model_info)
    if source is None:
        return NormalizedPricing(
            litellm_params=litellm_params, model_info=_without(model_info, PRICING_PROJECTION_KEYS), source=None
        )
    kept_info: Final = model_info if USD_PRICING not in model_info else _without(model_info, frozenset({USD_PRICING}))
    return NormalizedPricing(litellm_params=litellm_params, model_info=kept_info, source=source)


def normalize_deployment_pricing(
    litellm_params: Mapping[str, object], model_info: Mapping[str, object]
) -> NormalizedPricing | PricingCurrencyError:
    configured_currency: Final = _currency_setting(PRICING_CURRENCY, litellm_params, model_info)
    if isinstance(configured_currency, PricingCurrencyError):
        return configured_currency
    configured_rate: Final = _currency_setting(USD_PER_CURRENCY_UNIT, litellm_params, model_info)
    if isinstance(configured_rate, PricingCurrencyError):
        return configured_rate
    if configured_currency is None and configured_rate is None:
        return _rebuilt(litellm_params, model_info)
    currency: Final = _currency(PricingCurrency.USD.value if configured_currency is None else configured_currency)
    if isinstance(currency, PricingCurrencyError):
        return currency
    rate: Final = _rate(currency, configured_rate)
    if isinstance(rate, PricingCurrencyError):
        return rate
    undeclared: Final = _undeclared_price_error(litellm_params) or _undeclared_price_error(model_info)
    if undeclared is not None and currency is not PricingCurrency.USD:
        return undeclared
    params_prices: Final = _priced_fields(litellm_params)
    info_prices: Final = _unshadowed_prices(_priced_fields(model_info), params_prices)
    params_usd: Final = _converted_prices(params_prices, rate)
    if isinstance(params_usd, PricingCurrencyError):
        return params_usd
    info_usd: Final = _converted_prices(info_prices, rate)
    if isinstance(info_usd, PricingCurrencyError):
        return info_usd
    params_base: Final = _without(litellm_params, CURRENCY_FIELDS)
    info_base: Final = _without(model_info, CURRENCY_FIELDS | PRICING_PROJECTION_KEYS)
    shadowed: Final = frozenset(field for field in params_prices if field in _MIRRORED_FIELDS)
    if currency is PricingCurrency.USD:
        return NormalizedPricing(litellm_params=params_base, model_info=info_base, source=None)
    if not params_prices and not info_prices:
        return PricingCurrencyError(
            "no_prices",
            f"pricing_currency {currency.value} needs at least one custom price on the deployment, "
            "or leave it out to use the cost map's USD prices",
        )
    source: Final = PricingSource(
        currency=currency,
        usd_per_currency_unit=str(rate),
        revision=pricing_revision(currency, rate, params_prices, info_prices),
        litellm_params=dict(params_prices),
        model_info=dict(info_prices),
    )
    return NormalizedPricing(
        litellm_params={**params_base, **params_usd},
        model_info={
            **info_base,
            **info_usd,
            **{field: params_usd[field] for field in _priced_fields(model_info) if field in shadowed},
            PRICING_SOURCE: source.model_dump(mode="json"),
        },
        source=source,
    )


def pricing_source_of(model_info: Mapping[str, object] | None) -> PricingSource | None:
    """The ``pricing_source`` a deployment carries. Only trust it on records a deployment build produced: router
    records and the metadata the router stamps on a call, never a client payload."""
    if model_info is None:
        return None
    return _parsed_source(model_info.get(PRICING_SOURCE))


def source_currency_view(
    litellm_params: Mapping[str, object], model_info: Mapping[str, object]
) -> tuple[Mapping[str, object], Mapping[str, object]]:
    """The deployment as the operator configured it: prices back in the source currency, the currency settings on
    ``litellm_params``, mirrored fields showing the ``litellm_params`` price, and the USD rates under ``usd_pricing``.
    Anything else, including a record that does not reproduce the current prices, comes back without a currency
    record. A client that saves this view back stores the original configuration, never the USD projection."""
    source: Final = _verified_source(litellm_params, model_info)
    if source is None:
        return litellm_params, _without(model_info, PRICING_PROJECTION_KEYS)
    mirrored: Final = {field: value for field, value in source.litellm_params.items() if field in _MIRRORED_FIELDS}
    usd: Final = {
        **{field: model_info.get(field) for field in source.model_info},
        **{field: litellm_params.get(field) for field in source.litellm_params},
    }
    return (
        {
            **litellm_params,
            **source.litellm_params,
            PRICING_CURRENCY: source.currency.value,
            USD_PER_CURRENCY_UNIT: source.usd_per_currency_unit,
        },
        {**model_info, **source.model_info, **mirrored, USD_PRICING: usd},
    )


def without_pricing_projection(model_info: Mapping[str, object]) -> Mapping[str, object]:
    """``model_info`` without the server-derived currency record a read handed out, so saving that read back can
    never plant a provenance record or USD figures as configuration."""
    return _without(model_info, PRICING_PROJECTION_KEYS)


def configures_pricing_currency(litellm_params: Mapping[str, object], model_info: Mapping[str, object]) -> bool:
    return any(litellm_params.get(field) is not None or model_info.get(field) is not None for field in CURRENCY_FIELDS)


def pricing_configuration_error(litellm_params: Mapping[str, object], model_info: Mapping[str, object]) -> str | None:
    configured: Final = without_pricing_projection(model_info)
    if not configures_pricing_currency(litellm_params, configured):
        return None
    normalized: Final = normalize_deployment_pricing(litellm_params, configured)
    return normalized.message if isinstance(normalized, PricingCurrencyError) else None


def request_pricing_currency_error(kwargs: Mapping[str, object]) -> str | None:
    """Request-level custom prices are USD. A call that names another currency is refused rather than billed as if
    its prices were USD."""
    currency: Final = kwargs.get(PRICING_CURRENCY)
    rate: Final = kwargs.get(USD_PER_CURRENCY_UNIT)
    if currency in (None, PricingCurrency.USD.value) and (rate is None or _decimal(rate) == 1):
        return None
    return (
        "pricing_currency and usd_per_currency_unit are deployment settings. Set them on a Router deployment's "
        "litellm_params or model_info. Prices passed on a single call are always USD."
    )


def frozen_pricing_snapshot(entry: Mapping[str, object]) -> str:
    """The price entry a call was routed under, as text: immutable however deep its tier or off-peak tables go,
    and it survives the copies and serialization call metadata goes through."""
    return json.dumps(dict(entry), sort_keys=True, default=str)


def routed_pricing_snapshot(container: Mapping[str, object] | None) -> Mapping[str, object] | None:
    """The price entry captured when the router selected the call's currency-priced deployment, from the call's
    ``metadata`` or ``litellm_metadata``. Each read parses a fresh copy, and the entry is used only when it was
    captured for the revision of the deployment the call carries."""
    if container is None:
        return None
    buckets: Final = tuple(
        _STR_MAPPING.validate_python(container[key])
        for key in ("metadata", "litellm_metadata")
        if isinstance(container.get(key), Mapping)
    )
    entries: Final = (_snapshot_entry(bucket.get("model_info"), bucket.get(PRICING_SNAPSHOT)) for bucket in buckets)
    return next((entry for entry in entries if entry is not None), None)


def _snapshot_entry(model_info: object, snapshot: object) -> Mapping[str, object] | None:
    if not isinstance(model_info, Mapping) or not isinstance(snapshot, str):
        return None
    source: Final = pricing_source_of(_STR_MAPPING.validate_python(model_info))
    if source is None:
        return None
    try:
        entry: Final = _STR_MAPPING.validate_python(json.loads(snapshot))
    except (ValueError, ValidationError):
        return None
    captured: Final = pricing_source_of(entry)
    return entry if captured is not None and captured.revision == source.revision else None
