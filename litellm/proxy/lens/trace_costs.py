import re
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from functools import reduce
from itertools import chain
from math import isfinite
from types import MappingProxyType
from typing import Annotated, Final, Literal, TypeAlias

from pydantic import Field, TypeAdapter, ValidationError, model_validator

import litellm
from litellm._internal_context import pinned_billing_time
from litellm.cost_calculator import (
    _resolve_billable_service_tier,  # pyright: ignore[reportPrivateUsage]  # reuse the gateway's tier precedence
    completion_cost,  # pyright: ignore[reportUnknownVariableType]  # unused legacy parameters lack annotations
    pricing_entry_for_cost_calc,
)
from litellm.integrations.otel.model.semconv import litellm_provider_candidates, resolve_provider
from litellm.litellm_core_utils.llm_cost_calc.tiered_pricing import (
    select_tier_for_input,  # pyright: ignore[reportUnknownVariableType]  # validate the legacy selector's output below
)
from litellm.litellm_core_utils.llm_cost_calc.utils import BilledTokenRates, get_billed_token_rates
from litellm.llms.azure_ai.cost_calculator import is_azure_model_router
from litellm.proxy.lens.models import Record
from litellm.types.utils import (
    CacheCreationTokenDetails,
    LlmProvidersSet,
    ModelResponse,
    PromptTokensDetailsWrapper,
    ServiceTier,
    Usage,
)
from litellm.utils import (
    _check_provider_match,  # pyright: ignore[reportPrivateUsage, reportUnknownVariableType]  # reuse the catalog's offline provider compatibility check
    get_model_cost_key,
)

_KEYS: Final = frozenset(
    {
        "gen_ai.operation.name",
        "gen_ai.output.type",
        "gen_ai.request.model",
        "gen_ai.response.model",
        "gen_ai.provider.name",
        "gen_ai.system",
        "openai.request.service_tier",
        "openai.response.service_tier",
        "gen_ai.openai.request.service_tier",
        "gen_ai.openai.response.service_tier",
        "anthropic.response.service_tier",
    }
)
_PREFIXES: Final = ("gen_ai.usage.", "anthropic.usage.cache_creation.")
_COUNTS: Final = MappingProxyType(
    {
        "input_tokens": ("gen_ai.usage.input_tokens", "gen_ai.usage.prompt_tokens"),
        "output_tokens": ("gen_ai.usage.output_tokens", "gen_ai.usage.completion_tokens"),
        "cache_read_tokens": ("gen_ai.usage.cache_read.input_tokens",),
        "cache_write_tokens": ("gen_ai.usage.cache_write.input_tokens", "gen_ai.usage.cache_creation.input_tokens"),
        "cache_write_5m_tokens": ("anthropic.usage.cache_creation.ephemeral_5m_input_tokens",),
        "cache_write_1h_tokens": ("anthropic.usage.cache_creation.ephemeral_1h_input_tokens",),
        "reasoning_tokens": ("gen_ai.usage.reasoning.output_tokens",),
        "total_tokens": ("gen_ai.usage.total_tokens",),
    }
)
_COUNT_KEYS: Final = frozenset(chain.from_iterable(_COUNTS.values()))
_ZERO: Final = re.compile(r"0+(?:\.0+)?")
_EPOCH: Final = datetime(1970, 1, 1, tzinfo=timezone.utc)
_TIERS: Final = frozenset(tier.value for tier in ServiceTier) | {"default", "standard"}
_NON_TOKEN_UNITS: Final = ("per_character", "cost_per_second", "per_query", "per_request", "per_page", "per_session")
_REGIONAL_MULTIPLIERS: Final = (
    "regional_processing_uplift_multiplier_eu",
    "regional_processing_uplift_multiplier_us",
    "regional_endpoint_uplift_multiplier",
)
_HOURLY: Final = "cache_creation_input_token_cost_above_1hr"
_REMOTE_METADATA: Final = frozenset({"huggingface", "ollama", "ollama_chat", "lemonade"})
Count: TypeAlias = Annotated[int, Field(strict=True, ge=0, le=2**32 - 1)]
Cost: TypeAlias = Annotated[float, Field(ge=0, allow_inf_nan=False)]
_RATE: Final[TypeAdapter[float]] = TypeAdapter(Cost)
_PRICING_TIERS: Final = TypeAdapter(tuple[Mapping[str, object], ...])
_PRICING_ENTRY: Final[TypeAdapter[Mapping[str, object] | None]] = TypeAdapter(Mapping[str, object] | None)
AttributeKey: TypeAlias = Annotated[str, Field(min_length=1, max_length=128)]
AttributeValue: TypeAlias = Annotated[str, Field(max_length=512)]


class TraceCostInput(Record):
    start_ns: int = Field(strict=True, ge=-(2**63), le=2**63 - 1)
    attributes: Mapping[AttributeKey, AttributeValue] = Field(max_length=64)

    @model_validator(mode="after")
    def pricing_attributes_only(self) -> "TraceCostInput":
        if any(key not in _KEYS and not key.startswith(_PREFIXES) for key in self.attributes):
            raise ValueError("Only trace pricing attributes are accepted")
        return self


class TraceCostsRequest(Record):
    calls: tuple[TraceCostInput, ...] = Field(max_length=128)


class TraceCostsResponse(Record):
    costs: tuple[Cost | None, ...] = Field(max_length=128)


class _Evidence(Record):
    request_model: str | None = Field(default=None, min_length=1, max_length=512)
    response_model: str | None = Field(default=None, min_length=1, max_length=512)
    provider: str | None = Field(default=None, min_length=1, max_length=64)
    system: str | None = Field(default=None, min_length=1, max_length=64)
    request_tier: str | None = Field(default=None, min_length=1, max_length=64)
    response_tier: str | None = Field(default=None, min_length=1, max_length=64)
    input_tokens: Count
    output_tokens: Count
    cache_read_tokens: Count | None = None
    cache_write_tokens: Count | None = None
    cache_write_5m_tokens: Count | None = None
    cache_write_1h_tokens: Count | None = None
    reasoning_tokens: Count | None = None
    total_tokens: Count | None = None


def _count(attributes: Mapping[str, str], keys: tuple[str, ...]) -> int | None | Literal["invalid"]:
    values: Final = tuple(attributes[key] for key in keys if key in attributes)
    if not values:
        return None
    if any(not value or not value.isascii() or not value.isdecimal() for value in values):
        return "invalid"
    parsed: Final = tuple(int(value) for value in values)
    return parsed[0] if all(value == parsed[0] for value in parsed) else "invalid"


def _text(attributes: Mapping[str, str], keys: tuple[str, ...]) -> str | None:
    values: Final = tuple(attributes[key] for key in keys if key in attributes)
    if not values:
        return None
    return values[0] if all(value == values[0] and value == value.strip() for value in values) else ""


def _evidence(attributes: Mapping[str, str]) -> _Evidence | None:
    if attributes.get("gen_ai.operation.name") not in (None, "chat", "text_completion", "generate_content"):
        return None
    if attributes.get("gen_ai.output.type") not in (None, "text", "json"):
        return None
    if any(
        key.startswith(_PREFIXES) and key not in _COUNT_KEYS and _ZERO.fullmatch(value) is None
        for key, value in attributes.items()
    ):
        return None
    try:
        parsed: Final = _Evidence.model_validate(
            {
                **{name: _count(attributes, keys) for name, keys in _COUNTS.items()},
                "request_model": _text(attributes, ("gen_ai.request.model",)),
                "response_model": _text(attributes, ("gen_ai.response.model",)),
                "provider": _text(attributes, ("gen_ai.provider.name",)),
                "system": _text(attributes, ("gen_ai.system",)),
                "request_tier": _text(
                    attributes, ("openai.request.service_tier", "gen_ai.openai.request.service_tier")
                ),
                "response_tier": _text(
                    attributes,
                    (
                        "openai.response.service_tier",
                        "gen_ai.openai.response.service_tier",
                        "anthropic.response.service_tier",
                    ),
                ),
            }
        )
    except ValidationError:
        return None
    if (parsed.cache_read_tokens or 0) + (parsed.cache_write_tokens or 0) > parsed.input_tokens:
        return None
    if (parsed.reasoning_tokens or 0) > parsed.output_tokens:
        return None
    if parsed.total_tokens is not None and parsed.total_tokens != parsed.input_tokens + parsed.output_tokens:
        return None
    has_ttl: Final = parsed.cache_write_5m_tokens is not None or parsed.cache_write_1h_tokens is not None
    if (
        has_ttl
        and (parsed.cache_write_5m_tokens or 0) + (parsed.cache_write_1h_tokens or 0) != parsed.cache_write_tokens
    ):
        return None
    if any(tier is not None and tier.lower() not in _TIERS for tier in (parsed.request_tier, parsed.response_tier)):
        return None
    return parsed


def _usage(evidence: _Evidence) -> Usage:
    ttl: Final = evidence.cache_write_5m_tokens is not None or evidence.cache_write_1h_tokens is not None
    return Usage(
        prompt_tokens=evidence.input_tokens,
        completion_tokens=evidence.output_tokens,
        total_tokens=evidence.input_tokens + evidence.output_tokens,
        reasoning_tokens=evidence.reasoning_tokens,
        prompt_tokens_details=PromptTokensDetailsWrapper(
            cached_tokens=evidence.cache_read_tokens,
            cache_write_tokens=evidence.cache_write_tokens,
            cache_creation_token_details=CacheCreationTokenDetails(
                ephemeral_5m_input_tokens=evidence.cache_write_5m_tokens or 0,
                ephemeral_1h_input_tokens=evidence.cache_write_1h_tokens or 0,
            )
            if ttl
            else None,
        ),
    )


def _routing_multiplier(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return True
    try:
        return _RATE.validate_python(value) != 1
    except ValidationError:
        return True


def _unsupported_pricing(entry: Mapping[str, object]) -> bool:
    if entry.get("mode") not in (None, "chat", "completion"):
        return True
    if any(
        value is not None and (key == "citation_cost_per_token" or any(unit in key for unit in _NON_TOKEN_UNITS))
        for key, value in entry.items()
    ):
        return True
    regional: Final = any(_routing_multiplier(entry.get(key)) for key in _REGIONAL_MULTIPLIERS)
    provider_specific: Final = _PRICING_ENTRY.validate_python(entry.get("provider_specific_entry"))
    return regional or (
        provider_specific is not None and any(_routing_multiplier(value) for value in provider_specific.values())
    )


def _selected_tier(entry: Mapping[str, object], usage: Usage) -> Mapping[str, object]:
    try:
        tiers: Final = _PRICING_TIERS.validate_python(entry.get("tiered_pricing", ()))
    except ValidationError:
        return MappingProxyType({})
    selected: Final = _PRICING_ENTRY.validate_python(
        select_tier_for_input([dict(tier) for tier in tiers], usage.prompt_tokens)
    )
    return selected if selected is not None and "input_cost_per_token" in selected else MappingProxyType({})


def _declared_rate(entry: Mapping[str, object], key: str) -> bool:
    value: Final = entry.get(key)
    if isinstance(value, bool):
        return False
    try:
        _RATE.validate_python(value)
    except ValidationError:
        return False
    return True


def _sufficient(
    evidence: _Evidence, rates: BilledTokenRates, entry: Mapping[str, object], selected: Mapping[str, object]
) -> bool:
    if not all(_declared_rate({**entry, **selected}, key) for key in ("input_cost_per_token", "output_cost_per_token")):
        return False
    values: Final = (
        rates.input_cost_per_token,
        rates.output_cost_per_token,
        rates.cache_read_input_token_cost,
        rates.cache_creation_input_token_cost,
        rates.cache_creation_input_token_cost_above_1hr,
        rates.output_cost_per_reasoning_token,
    )
    if any(not isfinite(rate) or rate < 0 for rate in values):
        return False
    if evidence.input_tokens > 0:
        if evidence.cache_read_tokens is None and rates.cache_read_input_token_cost != rates.input_cost_per_token:
            return False
        if evidence.cache_write_tokens is None and any(
            rate != rates.input_cost_per_token
            for rate in (rates.cache_creation_input_token_cost, rates.cache_creation_input_token_cost_above_1hr)
        ):
            return False
    if (
        evidence.output_tokens > 0
        and evidence.reasoning_tokens is None
        and rates.output_cost_per_reasoning_token != rates.output_cost_per_token
    ):
        return False
    return not (
        (evidence.cache_write_tokens or 0) > 0
        and evidence.cache_write_5m_tokens is None
        and evidence.cache_write_1h_tokens is None
        and (
            rates.cache_creation_input_token_cost != rates.cache_creation_input_token_cost_above_1hr
            or _HOURLY in selected
        )
    )


def _pricing_entry(model: str, response: ModelResponse, provider: str) -> Mapping[str, object] | None:
    matched: Final = pricing_entry_for_cost_calc(
        model=model,
        completion_response=response,
        custom_llm_provider=provider,
        custom_pricing=None,
        base_model=None,
        router_model_id=None,
        region_name=None,
        litellm_logging_obj=None,
    )
    return matched[1] if matched is not None else None


def _select_provider(
    model: str, candidates: frozenset[str] | None, catalog_provider: object, names: tuple[str, ...]
) -> str | None:
    if candidates is None:
        return catalog_provider if isinstance(catalog_provider, str) else None
    if len(candidates) == 1:
        return next(iter(candidates))
    matches: Final = tuple(provider for provider in candidates if get_model_cost_key(f"{provider}/{model}") is not None)
    native: Final = tuple(
        provider for provider in matches if provider in names and provider != resolve_provider(provider)
    )
    if len(native) == 1:
        return native[0]
    if isinstance(catalog_provider, str) and catalog_provider in candidates:
        return catalog_provider
    if len(matches) == 1:
        return matches[0]
    compatible: Final = (
        tuple(
            provider
            for provider in candidates
            if _check_provider_match({"litellm_provider": catalog_provider}, provider)
        )
        if isinstance(catalog_provider, str)
        else ()
    )
    return compatible[0] if len(compatible) == 1 else None


def _catalog_provider(model: str, recorded: str | None, system: str | None) -> str | None:
    names: Final = tuple(
        name for name in (recorded, system) if name is not None and name not in ("litellm", "litellm.chat")
    )
    candidates: Final = (
        reduce(lambda left, right: left & right, map(litellm_provider_candidates, names)) if names else None
    )
    exact: Final = get_model_cost_key(model)
    entry: Final = _PRICING_ENTRY.validate_python(
        litellm.model_cost[exact] if exact is not None else None  # pyright: ignore[reportUnknownMemberType]  # validate the untyped runtime catalog boundary
    )
    provider: Final = _select_provider(
        model, candidates, entry.get("litellm_provider") if entry is not None else None, names
    )
    if provider is None or provider in _REMOTE_METADATA:
        return None
    selected_key: Final = get_model_cost_key(f"{provider}/{model}") or exact
    selected: Final = _PRICING_ENTRY.validate_python(
        litellm.model_cost[selected_key] if selected_key is not None else None  # pyright: ignore[reportUnknownMemberType]  # validate the untyped runtime catalog boundary
    )
    if selected is None or (provider not in LlmProvidersSet and provider != selected.get("litellm_provider")):
        return None
    return provider if _check_provider_match(dict(selected), provider) else None


def _calculate(call: TraceCostInput, evidence: _Evidence) -> float | None:
    model: Final = evidence.response_model or evidence.request_model
    if model is None:
        return None
    provider: Final = _catalog_provider(model, evidence.provider, evidence.system)
    if provider is None:
        return None
    if any(
        is_azure_model_router(name, custom_llm_provider=provider)
        for name in (evidence.request_model, model)
        if name is not None
    ):
        return None
    usage: Final = _usage(evidence)
    response: Final = ModelResponse(model=model, usage=usage, service_tier=evidence.response_tier)
    entry: Final = _pricing_entry(model, response, provider)
    if entry is None or _unsupported_pricing(entry):
        return None
    if evidence.request_model is not None and evidence.request_model != model:
        requested: Final = _pricing_entry(evidence.request_model, ModelResponse(model=evidence.request_model), provider)
        if requested is None or _unsupported_pricing(requested):
            return None
    instant: Final = _EPOCH + timedelta(microseconds=call.start_ns // 1000)
    tier: Final = _resolve_billable_service_tier(evidence.request_tier, evidence.response_tier)
    with pinned_billing_time(instant):
        rates: Final = get_billed_token_rates(model, provider, usage, service_tier=tier, current_time=instant)
        if rates is None or not _sufficient(evidence, rates, entry, _selected_tier(entry, usage)):
            return None
        amount: Final = completion_cost(
            completion_response=response,
            model=model,
            custom_llm_provider=provider,
            call_type="completion",
            optional_params={"service_tier": evidence.request_tier},
        )
    return amount if isfinite(amount) and amount >= 0 else None


def trace_cost(call: TraceCostInput) -> float | None:
    evidence: Final = _evidence(call.attributes)
    if evidence is None:
        return None
    try:
        return _calculate(call, evidence)
    except Exception:  # noqa: BLE001  # legacy pricing failures cannot expose internal state or fail a trace read
        return None


def trace_costs(request: TraceCostsRequest) -> TraceCostsResponse:
    return TraceCostsResponse(costs=tuple(trace_cost(call) for call in request.calls))
