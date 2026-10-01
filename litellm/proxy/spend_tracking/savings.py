"""
Per-request cost-savings computation for the Cost Optimization dashboard.

Turns the token-level savings recorded on a request into dollar amounts using
the model's own pricing. Daily rollup rows are keyed by date and entity, not by
model, so the dollars have to be computed here (where the model and its prices
are known) and summed into the daily tables; tokens cannot be priced after they
have been aggregated across models.
"""

from collections.abc import Callable, Mapping
from datetime import datetime
from math import isclose, isfinite
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal, NamedTuple

from pydantic import BaseModel, ConfigDict, Field

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.constants import INTERNAL_CALL_ORIGIN_METADATA_KEY
from litellm.litellm_core_utils.llm_cost_calc.utils import (
    _get_cost_per_unit,
    calculate_prompt_caching_savings,
    generic_cost_per_token,
)
from litellm.types.integrations.anthropic_cache_control_hook import (
    GATEWAY_INJECTED_CACHE_METADATA_KEY,
    GATEWAY_INJECTED_FOR_EVERY_DEPLOYMENT,
)

if TYPE_CHECKING:
    from litellm.router import Router
from litellm.types.utils import ModelInfo, PromptTokensDetailsWrapper, Usage


class SavingsSpend(NamedTuple):
    compression: float
    prompt_caching: float
    autorouter: float = 0.0
    gateway_injected_caching: float = 0.0


def _coerce_billed_at(value: datetime | str | None) -> datetime | None:
    if isinstance(value, datetime) or value is None:
        return value
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


class _ModelIdentity(NamedTuple):
    model: str
    provider: str


def _resolve_model(model: str | None, custom_llm_provider: str | None) -> _ModelIdentity | None:
    """Canonical ``(model, provider)``, or ``None`` when the model cannot be resolved.

    The two sides of the comparison arrive spelled differently: the spend log records a
    normalized model name alongside its provider, while the baseline arrives as the
    operator wrote it in config, with the provider prefixed, implied, or absent. Raw
    string equality therefore reads `anthropic/claude-opus-5` as a switch away from
    `claude-opus-5`, and pricing a bare name with no provider can resolve it to a
    different vendor's rates than the deployment it names.
    """
    if not model:
        return None
    try:
        resolved_model, provider, _, _ = litellm.get_llm_provider(model=model, custom_llm_provider=custom_llm_provider)
    except Exception as e:  # noqa: BLE001  # get_llm_provider raises for unroutable names; degrade to an unavailable estimate
        verbose_proxy_logger.debug(
            "savings: cannot resolve provider for model=%s custom_llm_provider=%s (%s)", model, custom_llm_provider, e
        )
        return None
    return _ModelIdentity(model=resolved_model, provider=provider)


def _effective_model_info(router: "Router | None", deployment_id: str | None, model: str) -> ModelInfo | None:
    """What a deployment is actually charged, or ``None`` to price by name.

    `Router.get_deployment_model_info` owns this: it merges a deployment's configured
    prices over the built-in map, folds in `base_model` defaults for deployments whose
    name is not a model, and falls back to the model name when nothing is overridden.
    Resolving a name here instead reads the public rate, which a deployment with a
    negotiated price does not pay, and an Azure deployment name prices to nothing at all.
    """
    if router is None or deployment_id is None:
        return None
    try:
        return router.get_deployment_model_info(deployment_id, model)
    except Exception as e:  # noqa: BLE001  # a dashboard metric must not fail the spend write
        verbose_proxy_logger.debug("savings: no deployment pricing for %s (%s)", model, e)
        return None


def _model_info(model: _ModelIdentity) -> ModelInfo | None:
    """The public rates for ``model``, or ``None`` when it has none."""
    try:
        return litellm.get_model_info(model=model.model, custom_llm_provider=model.provider)
    except Exception as e:  # noqa: BLE001  # get_model_info raises bare Exception for unmapped models
        verbose_proxy_logger.debug("savings: no pricing for provider=%s model=%s (%s)", model.provider, model.model, e)
        return None


class PricingBasis(NamedTuple):
    """The tier and region a request was priced on, as the cost calculator resolved them.

    Read back off the request's recorded ``cost_breakdown`` rather than re-derived. The
    tier the biller used comes from ``optional_params``, which no log record carries, and
    the served tier that does survive on the usage object is a different fact with the
    opposite precedence, so a spend-time re-derivation would disagree with the invoice on
    exactly the requests where the tier changed the price.
    """

    service_tier: str | None = None
    data_residency: str | None = None
    vertex_location: str | None = None


_STANDARD_RATES: Final = PricingBasis()


class BaselineCostSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    model: str
    provider: str
    prices: ModelInfo | None
    basis: PricingBasis = _STANDARD_RATES
    actual_spend: float = Field(allow_inf_nan=False, ge=0)
    actual_token_cost: float | None = Field(default=None, allow_inf_nan=False, ge=0)
    classifier_cost: float = Field(default=0.0, allow_inf_nan=False, ge=0)


def baseline_cost_snapshot(
    model: str,
    prices: ModelInfo | None,
    actual_spend: float,
    cost_breakdown: Mapping[str, object] | None,
    routing_decision: Mapping[str, object] | None,
) -> BaselineCostSnapshot:
    return BaselineCostSnapshot(
        model=model,
        provider="anthropic",
        prices=prices,
        actual_spend=actual_spend,
        basis=_pricing_basis(cost_breakdown),
        actual_token_cost=_recorded_token_cost(cost_breakdown),
        classifier_cost=classifier_cost_from_decision(routing_decision) or 0.0,
    )


class BaselineCosts(NamedTuple):
    actual: float
    baseline: float

    @property
    def savings(self) -> float:
        return self.baseline - self.actual


def price_baseline_comparison(
    snapshot: BaselineCostSnapshot,
    baseline_usage: Usage | None,
    provenance: Literal["observed_identical", "modeled"] | None,
) -> BaselineCosts | None:
    if baseline_usage is None or provenance is None:
        return None
    actual: Final = snapshot.actual_spend + snapshot.classifier_cost
    if provenance == "observed_identical":
        return BaselineCosts(actual=actual, baseline=snapshot.actual_spend)
    if snapshot.prices is None or snapshot.actual_token_cost is None:
        return None
    token_cost: Final = _cost_of_usage(
        _ModelIdentity(snapshot.model, snapshot.provider), baseline_usage, snapshot.prices, snapshot.basis
    )
    if token_cost is None or not isfinite(token_cost) or token_cost < 0:
        return None
    baseline: Final = snapshot.actual_spend + token_cost - snapshot.actual_token_cost
    if not isfinite(baseline) or baseline < 0:
        return None
    return BaselineCosts(actual=actual, baseline=baseline)


def _pricing_basis(cost_breakdown: Mapping[str, object] | None) -> PricingBasis:
    """The basis recorded on a request, defaulting to standard rates when absent.

    Rows written before this field shipped carry neither key, and there is no backfill:
    they price at standard rates, which is what they already did.

    These values survive a JSON round trip on the way here, so none is guaranteed to be
    a string. `generic_cost_per_token` calls `.lower()` on them without a type check, and
    the resulting `AttributeError` would be swallowed into a silent zero by the caller's
    `except`, so anything that is not a string is dropped here instead.
    """
    if not cost_breakdown:
        return _STANDARD_RATES
    service_tier: Final = cost_breakdown.get("service_tier")
    data_residency: Final = cost_breakdown.get("data_residency")
    vertex_location: Final = cost_breakdown.get("vertex_location")
    return PricingBasis(
        service_tier=service_tier if isinstance(service_tier, str) else None,
        data_residency=data_residency if isinstance(data_residency, str) else None,
        vertex_location=vertex_location if isinstance(vertex_location, str) else None,
    )


def _recorded_token_cost(cost_breakdown: Mapping[str, object] | None) -> float | None:
    """What the biller charged for this request's tokens, or ``None`` when unrecorded.

    ``input_cost`` already carries the cache buckets, so it and ``output_cost`` sum to
    exactly what `generic_cost_per_token` returns for the same request; the separate
    ``cache_read_cost`` and ``cache_creation_cost`` entries decompose that sum rather than
    adding to it, and including them would charge those tokens twice.

    Built-in tool cost, discount and margin are deliberately left out. They are properties
    of the request and the operator's contract rather than of the model the router picked,
    so they land on both sides of the comparison or neither, and only the total the
    counterfactual can also be priced on belongs here.
    """
    if not cost_breakdown:
        return None
    input_cost: Final = cost_breakdown.get("input_cost")
    output_cost: Final = cost_breakdown.get("output_cost")
    if not isinstance(input_cost, (int, float)) or not isinstance(output_cost, (int, float)):
        return None
    return float(input_cost) + float(output_cost)


def _cost_of_usage(
    model: _ModelIdentity,
    usage: Usage,
    model_info: ModelInfo | None = None,
    basis: PricingBasis = _STANDARD_RATES,
) -> float | None:
    """What ``usage`` costs on ``model``, or ``None`` when the model has no pricing."""
    try:
        if model.provider == "anthropic":
            from litellm.llms.anthropic.cost_calculation import cost_per_token

            prompt_cost, completion_cost = cost_per_token(
                model=model.model,
                usage=usage,
                service_tier=basis.service_tier,
                model_info=model_info,
            )
        else:
            prompt_cost, completion_cost = generic_cost_per_token(
                model=model.model,
                usage=usage,
                custom_llm_provider=model.provider,
                service_tier=basis.service_tier,
                data_residency=basis.data_residency,
                model_info=model_info,
                vertex_location=basis.vertex_location,
            )
    except Exception as e:  # noqa: BLE001  # get_model_info raises bare Exception for unmapped models; degrade to zero savings
        verbose_proxy_logger.debug(
            "savings: cannot price usage for provider=%s model=%s (%s)", model.provider, model.model, e
        )
        return None
    return prompt_cost + completion_cost


def _cache_token_split(usage: Usage) -> tuple[int, int]:
    """``(cache_read_tokens, cache_creation_tokens)`` for a request."""
    details: Final = usage.prompt_tokens_details
    if details is None:
        return 0, 0
    read: Final = getattr(details, "cached_tokens", 0) or 0
    created = (getattr(details, "cache_creation_tokens", 0) or 0) or (getattr(details, "cache_write_tokens", 0) or 0)
    return int(read), int(created)


def _baseline_cache_rate_keys(baseline_info: ModelInfo | None) -> tuple[bool, bool]:
    """Whether the baseline model has a ``(cache read, cache write)`` rate of its own.

    A missing rate is not a free bucket. `_get_token_base_cost` resolves an absent
    `cache_read_input_token_cost` or `cache_creation_input_token_cost` to 0.0, so a
    baseline whose provider prices caching implicitly, which is every OpenAI, Azure and
    Gemini entry for cache writes, would carry the whole prompt for nothing and turn a
    profitable route into a reported loss. Such a model pays its plain input rate for
    those tokens, so the buckets it cannot price become ordinary input below.

    The two buckets need different tests, because a `0.0` means something different in
    each and the cost map proves it.

    Reads: an explicit `0.0` is a real "cache reads are free" price, so absence rather
    than truthiness is the right test. It only means that on a model that actually
    caches, though. Six entries pair `cache_read_input_token_cost` of `0` with
    `supports_prompt_caching` of `False`, `gemini-robotics-er-1.5-preview` and
    `openrouter/z-ai/glm-4.7` among them, where the zero is a placeholder for a model
    that has no cache rather than a free one. Honouring it priced 20,000 baseline tokens
    at `$0.00` instead of `$0.006`.

    Writes: truthiness stays. A `0.0` cache-creation price is not free, it means writes
    bill at the plain input rate, and 36 entries rely on that inheritance including
    `deepseek/deepseek-chat`. Reading it as a real price dropped a 10,000 token
    first-turn baseline from `$0.0028` to `$0.00`.
    """
    if baseline_info is None:
        return True, True
    prices_reads: Final = baseline_info.get("cache_read_input_token_cost") is not None and bool(
        baseline_info.get("supports_prompt_caching")
    )
    return prices_reads, bool(baseline_info.get("cache_creation_input_token_cost"))


def _baseline_usage(usage: Usage, baseline_info: ModelInfo | None = None) -> Usage:
    cache_read, cache_creation = _cache_token_split(usage)
    details: Final = usage.prompt_tokens_details
    if details is None or (cache_read <= 0 and cache_creation <= 0):
        return usage
    prices_reads, prices_writes = _baseline_cache_rate_keys(baseline_info)
    reads: Final = cache_read if prices_reads else 0
    writes: Final = cache_creation if prices_writes else 0
    if (reads, writes) == (cache_read, cache_creation):
        return usage
    other_modalities: Final = sum(
        (getattr(details, field, 0) or 0) for field in ("audio_tokens", "image_tokens", "video_tokens")
    )
    return Usage(
        **{
            **usage.model_dump(),
            # Rebuild through Usage so private fallback counts agree with the public buckets.
            "cache_read_input_tokens": reads,
            "cache_creation_input_tokens": writes,
            "prompt_tokens_details": PromptTokensDetailsWrapper(
                **{
                    **details.model_dump(),
                    "cached_tokens": reads,
                    "cache_creation_tokens": writes,
                    "cache_write_tokens": writes,
                    "cache_creation_token_details": details.cache_creation_token_details if writes else None,
                    "text_tokens": max(usage.prompt_tokens - reads - writes - other_modalities, 0),
                }
            ),
        },
    )


def compute_autorouter_savings(
    baseline_model: str | None,
    selected_model: str | None,
    selected_provider: str | None,
    usage: Usage,
    conversation_continuing: bool = True,
    selected_info: ModelInfo | None = None,
    baseline_info: ModelInfo | None = None,
    cost_breakdown: Mapping[str, object] | None = None,
    baseline_deployment_id: str | None = None,
    selected_deployment_id: str | None = None,
    baseline_usage: Usage | None = None,
    baseline_provenance: Literal["observed_initial", "modeled"] | None = None,
) -> float | None:
    """Price established baseline usage; conversation shape cannot establish cache warmth."""
    baseline: Final = _resolve_model(baseline_model, None)
    selected: Final = _resolve_model(selected_model, selected_provider)
    if baseline is None or selected is None:
        return None
    if baseline_usage is None and any(_cache_token_split(usage)):
        return None
    basis: Final = _pricing_basis(cost_breakdown)
    effective_baseline_info: Final = baseline_info if baseline_info is not None else _model_info(baseline)
    modeled_usage: Final = baseline_usage if baseline_usage is not None else usage
    baseline_cost: Final = _cost_of_usage(
        baseline, _baseline_usage(modeled_usage, effective_baseline_info), effective_baseline_info, basis
    )
    recorded_selected_cost: Final = _recorded_token_cost(cost_breakdown)
    selected_cost: Final = (
        recorded_selected_cost
        if recorded_selected_cost is not None
        else _cost_of_usage(selected, usage, selected_info, basis)
    )
    if baseline_cost is None or selected_cost is None:
        return None
    if baseline_provenance == "observed_initial":
        same_prices: Final = effective_baseline_info == (
            selected_info if selected_info is not None else _model_info(selected)
        )
        equivalent: Final = (
            baseline_usage is not None
            and baseline_usage == usage
            and baseline == selected
            and bool(baseline_deployment_id)
            and baseline_deployment_id == selected_deployment_id
            and same_prices
            and recorded_selected_cost is not None
            and isclose(baseline_cost, recorded_selected_cost, rel_tol=1e-9, abs_tol=1e-12)
        )
        return 0.0 if equivalent else None
    difference: Final = baseline_cost - selected_cost
    return difference if isfinite(difference) else None


def _usage_from_spend_log(usage_object: Mapping[str, object] | None) -> Usage | None:
    """Rebuild the request's ``Usage`` from the copy the spend log recorded."""
    if not usage_object:
        return None
    try:
        return Usage(**usage_object)
    except Exception as e:  # noqa: BLE001  # a malformed usage_object must not fail the daily spend write
        # Warning, not debug: this silently zeroes the auto-router driver for every
        # affected row, and a shape change in Usage would otherwise show up only as a
        # dashboard that quietly reads $0.00.
        verbose_proxy_logger.warning("savings: unusable usage_object, auto-router savings will read zero (%s)", e)
        return None


def marks_gateway_injection(metadata: Mapping[str, object] | None, model_id: str | None) -> bool:
    """Whether the gateway put cache breakpoints on the payload THIS row was billed for.

    ``AnthropicCacheControlHook.record_gateway_injection`` stamps the deployment it
    injected for, and a row carries the deployment it was billed for, so the two agree
    only on the leg that was actually injected. Every retry, failover and fallback of a
    request shares one metadata bucket and one ``litellm_call_id``, so the deployment is
    what tells those legs apart, and a marker left by a sibling reads here as no injection
    without anyone having to strip it. An injection that ran before any deployment was
    chosen is in the payload every leg sends, so it is marked for all of them and credits
    each. Absent on requests the gateway never acted on
    (client-supplied ``cache_control``, implicit provider caching) and on rows written
    before the marker shipped; all of it is the fail-closed direction.
    """
    if not metadata:
        return False
    injected_deployment: Final = metadata.get(GATEWAY_INJECTED_CACHE_METADATA_KEY)
    if not isinstance(injected_deployment, str):
        return False
    return injected_deployment in (GATEWAY_INJECTED_FOR_EVERY_DEPLOYMENT, model_id)


def extract_cache_read_tokens(usage_object: Mapping[str, object] | None) -> int:
    """Cache-read tokens from a logged usage object, whatever shape recorded them.

    Anthropic writes a top-level ``cache_read_input_tokens``; OpenAI-compatible
    providers (moonshotai, openai, deepseek, etc.) write
    ``prompt_tokens_details.cached_tokens``. This is the one owner of that
    normalization: callers hand over the usage object rather than threading a
    count that could disagree with it.
    """
    if not usage_object:
        return 0
    explicit: Final = usage_object.get("cache_read_input_tokens")
    if isinstance(explicit, (int, float)) and explicit:
        return int(explicit)
    details: Final = usage_object.get("prompt_tokens_details")
    if not isinstance(details, Mapping):
        return 0
    cached: Final = details.get("cached_tokens")
    return int(cached) if isinstance(cached, (int, float)) else 0


def extract_cache_creation_tokens(usage_object: Mapping[str, object] | None) -> int:
    """Cache-write tokens from a logged usage object, whatever shape recorded them.

    Anthropic writes a top-level ``cache_creation_input_tokens``; OpenAI-compatible
    providers (kimi-k2 etc.) write ``prompt_tokens_details.cache_write_tokens`` or
    ``prompt_tokens_details.cache_creation_tokens``.
    """
    if not usage_object:
        return 0
    explicit: Final = usage_object.get("cache_creation_input_tokens")
    if isinstance(explicit, (int, float)) and explicit:
        return int(explicit)
    details: Final = usage_object.get("prompt_tokens_details")
    if not isinstance(details, Mapping):
        return 0
    written: Final = next(
        (
            value
            for value in (details.get("cache_write_tokens"), details.get("cache_creation_tokens"))
            if isinstance(value, (int, float)) and value
        ),
        0,
    )
    return int(written)


def _proxy_llm_router() -> "Router | None":
    """The running proxy's router, or ``None`` outside a proxy (public rates only)."""
    try:
        from litellm.proxy.proxy_server import llm_router
    except Exception:  # noqa: BLE001  # SDK-only usage has no proxy module to import
        return None
    return llm_router


def _numeric_savings(value: object) -> float | None:
    """``value`` as a recorded savings figure, or ``None`` when it is not one."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
        return None
    return float(value)


def recorded_estimated_autorouter_savings(metadata: Mapping[str, object]) -> float | None:
    estimate: Final = metadata.get("autorouter_savings_estimate")
    if (
        not isinstance(estimate, Mapping)
        or type(estimate.get("version")) is not int
        or estimate.get("version") not in (1, 2, 3)
        or estimate.get("status") != "estimated"
    ):
        return None
    return _numeric_savings(metadata.get("autorouter_savings"))


def classifier_cost_from_decision(routing_decision: Mapping[str, object] | None) -> float | None:
    """The LLM-classifier cost a routing decision recorded, or ``None`` when it holds none.

    ``None`` covers the decision-less request, the heuristic short-circuit that never
    called a classifier, the unpriced classifier model, and a malformed value alike:
    in every one of those cases there is no dollar figure to move, so callers treat
    ``None`` as zero rather than as an error. The one owner of that reading, shared by
    the savings netting, the session rollup and the response header, so the three can
    never disagree about what counts as a classifier charge.
    """
    decision: Final = routing_decision if isinstance(routing_decision, Mapping) else {}
    return _numeric_savings(decision.get("classifier_cost"))


def autorouter_savings_for_request(
    model: str | None,
    custom_llm_provider: str | None,
    routing_decision: Mapping[str, object] | None,
    usage_object: Mapping[str, object] | None,
    model_id: str | None = None,
    llm_router: "Callable[[], Router | None] | None" = None,
    cost_breakdown: Mapping[str, object] | None = None,
    baseline_usage: Usage | None = None,
    baseline_provenance: Literal["observed_initial", "modeled"] | None = None,
) -> float | None:
    """Return net savings for established usage, or None when the estimate is unavailable."""
    usage: Final = _usage_from_spend_log(usage_object)
    if usage is None or not model:
        return None
    decision: Final = routing_decision if isinstance(routing_decision, Mapping) else {}
    recorded: Final = decision.get("savings_baseline_model")
    recorded_id: Final = decision.get("savings_baseline_deployment_id")
    baseline_model: Final = recorded if isinstance(recorded, str) else None
    baseline_id: Final = recorded_id if isinstance(recorded_id, str) else None
    if not decision or not baseline_model:
        return None
    router_instance: Final = llm_router() if llm_router else None
    gross: Final = compute_autorouter_savings(
        baseline_model=baseline_model,
        selected_model=model,
        selected_provider=custom_llm_provider,
        usage=usage,
        selected_info=_effective_model_info(router_instance, model_id, model or ""),
        baseline_info=_effective_model_info(router_instance, baseline_id, baseline_model or ""),
        cost_breakdown=cost_breakdown,
        baseline_deployment_id=baseline_id,
        selected_deployment_id=model_id,
        baseline_usage=baseline_usage,
        baseline_provenance=baseline_provenance,
    )
    if gross is None:
        return None
    classifier_cost: Final = classifier_cost_from_decision(decision)
    return gross if classifier_cost is None else gross - classifier_cost


def autorouter_savings_for_logging_payload(
    request_metadata: Mapping[str, object],
    model: str | None,
    custom_llm_provider: str | None,
    model_id: str | None,
    usage_object: Mapping[str, object] | None,
    cost_breakdown: Mapping[str, object] | None,
    baseline_usage: Usage | None = None,
    baseline_provenance: Literal["observed_initial", "modeled"] | None = None,
) -> float | None:
    """The figure the logging payload records for a request, or ``None`` when none should be.

    Internal sub-calls (the auto-router classifier, shadow eval's shadow and judge legs)
    are excluded here for the same reason the spend writer zeroes them: they can carry a
    real routing decision, but they are not requests the caller made, so a figure stamped
    on them would report savings for traffic no user sent.
    """
    if request_metadata.get(INTERNAL_CALL_ORIGIN_METADATA_KEY):
        return None
    routing_decision: Final = request_metadata.get("routing_decision")
    return autorouter_savings_for_request(
        model=model,
        custom_llm_provider=custom_llm_provider,
        routing_decision=routing_decision if isinstance(routing_decision, Mapping) else None,
        usage_object=usage_object,
        model_id=model_id,
        llm_router=_proxy_llm_router,
        cost_breakdown=cost_breakdown,
        baseline_usage=baseline_usage,
        baseline_provenance=baseline_provenance,
    )


def _request_savings_pricing(
    model: str | None,
    custom_llm_provider: str | None,
    model_id: str | None,
    llm_router: "Callable[[], Router | None] | None",
) -> tuple[str | None, ModelInfo | None]:
    router_instance: Final = llm_router() if llm_router else None
    identity: Final = _resolve_model(model, custom_llm_provider)
    pricing: Final = _effective_model_info(router_instance, model_id, model or "") or (
        _model_info(identity) if identity else None
    )
    return identity.provider if identity else custom_llm_provider, pricing


def _prompt_caching_savings(
    pricing: ModelInfo | None,
    provider: str | None,
    usage_object: Mapping[str, object] | None,
    cost_breakdown: Mapping[str, object] | None,
    billed_at: datetime | str | None,
) -> float | None:
    usage: Final = _usage_from_spend_log(usage_object)
    if pricing is None or usage is None:
        return None
    basis: Final = _pricing_basis(cost_breakdown)
    result: Final = calculate_prompt_caching_savings(
        model_info=pricing,
        usage=usage,
        custom_llm_provider=provider,
        service_tier=basis.service_tier,
        data_residency=basis.data_residency,
        vertex_location=basis.vertex_location,
        billed_at=_coerce_billed_at(billed_at),
    )
    return result if isfinite(result) else None


def prompt_caching_savings_for_request(
    model: str | None,
    custom_llm_provider: str | None,
    usage_object: Mapping[str, object] | None,
    model_id: str | None = None,
    llm_router: "Callable[[], Router | None] | None" = None,
    cost_breakdown: Mapping[str, object] | None = None,
    billed_at: datetime | str | None = None,
) -> float | None:
    request_pricing: Final = _request_savings_pricing(model, custom_llm_provider, model_id, llm_router)
    return _prompt_caching_savings(request_pricing[1], request_pricing[0], usage_object, cost_breakdown, billed_at)


def compute_savings_spend(
    model: str | None,
    custom_llm_provider: str | None,
    compression_saved_tokens: int,
    gateway_injected_cache: bool,
    routing_decision: Mapping[str, object] | None = None,
    usage_object: Mapping[str, object] | None = None,
    model_id: str | None = None,
    llm_router: "Callable[[], Router | None] | None" = None,
    cost_breakdown: Mapping[str, object] | None = None,
    recorded_autorouter_savings: object = None,
    recorded_autorouter_savings_estimate: Mapping[str, object] | None = None,
    billed_at: datetime | str | None = None,
) -> SavingsSpend:
    """
    Dollar savings for one request, split by optimization driver.

    Compression savings price the tokens compression removed at the model's
    input rate. Prompt-caching savings are NET: the cache-read discount minus the
    premium paid to write those entries, both derived here from ``usage_object`` so no
    caller can hand in a count that disagrees with the usage record.

    The uncached counterfactual pays the ordinary input rate for the same prompt size
    and tier. Cache writes subtract only the premium over that rate, split by TTL.
    Savings stay signed: a write-only request can lose money, and daily rollups net
    those losses against read savings.

    Caching is reported twice. ``prompt_caching`` is every net dollar caching saved,
    whoever caused it, which is what a customer means by "what did caching save me".
    ``gateway_injected_caching`` is the subset the gateway can claim credit for, carrying
    a value only when ``gateway_injected_cache`` is set, i.e. litellm itself added the
    ``cache_control`` breakpoints (configured injection points or the auto prompt-caching
    flag). A client that sent its own breakpoints, and a provider that
    caches implicitly (OpenAI, Gemini), produce the same usage shape with no gateway
    action, so they count toward the total and not toward the attributed figure.

    Reporting both rather than gating the one column keeps the customer-facing number
    stable across the change and leaves attribution a separate question. The attributed
    figure is normally the smaller of the two, being a subset of the same requests, but
    not always: a request that only writes cache and never reads it has negative net
    savings, and dropping such a request from the attributed figure can lift it above
    the total. Auto-router savings compare established baseline usage against the
    recorded selected-model cost. Versioned unknown estimates contribute no dollars
    to this subtotal and are excluded from the separately reported coverage cohort.

    ``llm_router`` is passed as a provider rather than a router because every spend write
    calls this and only auto-routed ones need one, so looking it up eagerly at the call
    site would fetch and discard it on the rest.

    ``cost_breakdown`` supplies the biller's tier and region to caching and auto-router
    savings. Caching also uses the logged prompt size and TTL split. Compression retains
    its flat input-rate estimate; changing that counterfactual is a separate concern.

    ``recorded_autorouter_savings`` is the figure the logging path stamped on the spend
    log's metadata, honoured over recomputation so the rollup, the turn table and the
    per-request record cannot disagree; rows written before the field shipped carry
    nothing and recompute, mirroring ``_recorded_token_cost``.
    """
    # Deployment rates when the request came through one, public rates otherwise --
    # `_effective_model_info` merges a deployment's configured prices over the built-in
    # map, so a negotiated price is not silently replaced by the list rate.
    request_pricing: Final = _request_savings_pricing(model, custom_llm_provider, model_id, llm_router)
    provider: Final = request_pricing[0]
    pricing: Final = request_pricing[1]
    input_cost: Final = (_get_cost_per_unit(pricing, "input_cost_per_token") or 0.0) if pricing else 0.0
    compression: Final = max(compression_saved_tokens, 0) * input_cost
    prompt_caching: Final = _prompt_caching_savings(pricing, provider, usage_object, cost_breakdown, billed_at) or 0.0
    gateway_injected_caching: Final = prompt_caching if gateway_injected_cache else 0.0

    # The figure the logging path recorded wins, before the usage gate on purpose: a row
    # whose usage no longer parses still carries the number computed when it did.
    recorded_savings: Final = (
        recorded_estimated_autorouter_savings(
            MappingProxyType(
                {
                    "autorouter_savings": recorded_autorouter_savings,
                    "autorouter_savings_estimate": recorded_autorouter_savings_estimate,
                }
            )
        )
        if recorded_autorouter_savings_estimate is not None
        else _numeric_savings(recorded_autorouter_savings)
    )
    autorouter: Final = (
        recorded_savings
        if recorded_savings is not None or recorded_autorouter_savings_estimate is not None
        else autorouter_savings_for_request(
            model=model,
            custom_llm_provider=custom_llm_provider,
            routing_decision=routing_decision,
            usage_object=usage_object,
            model_id=model_id,
            llm_router=llm_router,
            cost_breakdown=cost_breakdown,
        )
    )
    return SavingsSpend(
        compression=compression,
        prompt_caching=prompt_caching,
        autorouter=0.0 if autorouter is None else autorouter,
        gateway_injected_caching=gateway_injected_caching,
    )
