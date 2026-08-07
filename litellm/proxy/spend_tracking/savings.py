"""
Per-request cost-savings computation for the Cost Optimization dashboard.

Turns the token-level savings recorded on a request into dollar amounts using
the model's own pricing. Daily rollup rows are keyed by date and entity, not by
model, so the dollars have to be computed here (where the model and its prices
are known) and summed into the daily tables; tokens cannot be priced after they
have been aggregated across models.
"""

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Final, NamedTuple

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.llm_cost_calc.utils import generic_cost_per_token

if TYPE_CHECKING:
    from litellm.router import Router
from litellm.types.utils import ModelInfo, PromptTokensDetailsWrapper, Usage


class SavingsSpend(NamedTuple):
    compression: float
    prompt_caching: float
    autorouter: float = 0.0


def _input_and_cache_read_cost(model: str | None, custom_llm_provider: str | None) -> tuple[float, float]:
    """
    Return ``(input_cost_per_token, cache_read_cost_per_token)`` for a model.

    Falls open to ``(0.0, 0.0)`` when the model is unknown so savings degrade to
    zero rather than raising inside the spend writer. When a model has no
    separate cache-read price the cache-read cost mirrors the input cost, which
    yields zero caching savings.
    """
    if not model:
        return 0.0, 0.0
    try:
        info: Final = litellm.get_model_info(model=model, custom_llm_provider=custom_llm_provider)
    except Exception as e:  # noqa: BLE001  # get_model_info raises bare Exception for unmapped models; degrade to zero savings
        verbose_proxy_logger.debug(
            "savings: no model info for provider=%s model=%s (%s)", custom_llm_provider, model, e
        )
        return 0.0, 0.0
    input_cost: Final = float(info.get("input_cost_per_token") or 0.0)
    cache_read_cost: Final = info.get("cache_read_input_token_cost")
    if cache_read_cost is None:
        return input_cost, input_cost
    return input_cost, float(cache_read_cost)


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
    except Exception as e:  # noqa: BLE001  # get_llm_provider raises for unroutable names; degrade to zero savings
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


_STANDARD_RATES: Final = PricingBasis()


def _pricing_basis(cost_breakdown: Mapping[str, object] | None) -> PricingBasis:
    """The basis recorded on a request, defaulting to standard rates when absent.

    Rows written before this field shipped carry neither key, and there is no backfill:
    they price at standard rates, which is what they already did.

    Both values survive a JSON round trip on the way here, so neither is guaranteed to be
    a string. `generic_cost_per_token` calls `.lower()` on both without a type check, and
    the resulting `AttributeError` would be swallowed into a silent zero by the caller's
    `except`, so anything that is not a string is dropped here instead.
    """
    if not cost_breakdown:
        return _STANDARD_RATES
    service_tier: Final = cost_breakdown.get("service_tier")
    data_residency: Final = cost_breakdown.get("data_residency")
    return PricingBasis(
        service_tier=service_tier if isinstance(service_tier, str) else None,
        data_residency=data_residency if isinstance(data_residency, str) else None,
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
        prompt_cost, completion_cost = generic_cost_per_token(
            model=model.model,
            usage=usage,
            custom_llm_provider=model.provider,
            service_tier=basis.service_tier,
            data_residency=basis.data_residency,
            model_info=model_info,
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


_CACHE_SPLIT_FIELDS: Final = frozenset(
    ("cached_tokens", "cache_creation_tokens", "cache_write_tokens", "cache_creation_token_details", "text_tokens")
)


def _baseline_cache_rate_keys(baseline_info: ModelInfo | None) -> tuple[bool, bool]:
    """Whether the baseline model has a ``(cache read, cache write)`` rate of its own.

    A missing rate is not a free bucket. `_get_token_base_cost` resolves an absent
    `cache_read_input_token_cost` or `cache_creation_input_token_cost` to 0.0, so a
    baseline whose provider prices caching implicitly, which is every OpenAI, Azure and
    Gemini entry for cache writes, would carry the whole prompt for nothing and turn a
    profitable route into a reported loss. Such a model pays its plain input rate for
    those tokens, so the buckets it cannot price become ordinary input below.
    """
    if baseline_info is None:
        return True, True
    return bool(baseline_info.get("cache_read_input_token_cost")), bool(
        baseline_info.get("cache_creation_input_token_cost")
    )


def _baseline_usage(usage: Usage, conversation_continuing: bool, baseline_info: ModelInfo | None = None) -> Usage:
    """The same request as a single-model baseline would have met it.

    The baseline is one model serving every turn, so whether it had this prompt cached
    is simply whether the conversation was already underway. On a continuing
    conversation it wrote the prompt on an earlier turn and would only read it now, so
    the cache tokens move into the read bucket and whatever this request paid to write
    counts against the saving; that write is what switching models costs.

    On a conversation's first turn nothing was cached anywhere, for any model. The
    baseline would have written the same prompt, so the cache buckets stay where they are
    and both arms carry the write at their own rates, unless the baseline has no rate for
    a bucket, in which case those tokens are its plain input. Charging the write to this case
    too, which is all a single rollup row can support, understates a first turn to a
    few percent of its value and can render a profitable route as a loss.

    A continuing turn that mostly read from cache is the third case: the selected model
    was already warm, so it is the one that has been serving this conversation and the
    baseline's cache holds exactly what its does. The tokens written are the turn's own
    growth, new to every model, and the baseline would have paid to write them too.
    Moving them would forgive the baseline a write it really owes and shrink the
    reported saving. "Mostly read" rather than "read anything" on purpose: a switch onto
    a model holding a small prefix of this prompt still writes most of it, and must keep
    counting that write against the saving.

    Only the cache buckets move. Every other field the request was priced on travels
    through untouched, audio and image and video counts among them, because the baseline
    is this same request served by a model that happened to be warm; naming the fields to
    keep instead would price the baseline on a request that never ran, and would go stale
    the next time a priced field is added.
    """
    cache_read, cache_creation = _cache_token_split(usage)
    details: Final = usage.prompt_tokens_details
    if details is None or (cache_read <= 0 and cache_creation <= 0):
        return usage

    # The tokens this request paid to write move into the cached count and the creation
    # charge is dropped: on one model that cache was already warm, so the baseline would
    # have read them rather than paying to create them. The 5m/1h breakdown goes with
    # them; left behind it re-charges the write.
    warm: Final = conversation_continuing and cache_creation > 0 and cache_read <= cache_creation
    reads = cache_read + cache_creation if warm else cache_read
    writes = 0 if warm else cache_creation

    prices_reads, prices_writes = _baseline_cache_rate_keys(baseline_info)
    reads = reads if prices_reads else 0
    writes = writes if prices_writes else 0
    if (reads, writes) == (cache_read, cache_creation):
        return usage

    other_modalities: Final = sum(
        (getattr(details, field, 0) or 0) for field in ("audio_tokens", "image_tokens", "video_tokens")
    )
    return Usage(
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        total_tokens=usage.total_tokens,
        completion_tokens_details=usage.completion_tokens_details,
        prompt_tokens_details=PromptTokensDetailsWrapper(
            **details.model_dump(exclude=_CACHE_SPLIT_FIELDS),
            cached_tokens=reads,
            cache_creation_tokens=writes,
            cache_write_tokens=writes,
            cache_creation_token_details=details.cache_creation_token_details if writes else None,
            # Whatever no longer sits in a cache bucket is plain input on the baseline.
            text_tokens=max(usage.prompt_tokens - reads - writes - other_modalities, 0),
        ),
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
) -> float:
    """Net dollars the router saved, or cost, by serving this request on ``selected_model``.

    Signed on purpose. Switching models leaves the new one with a cold cache, so the
    request pays a cache-creation charge that staying on one model would not have
    incurred; when that charge outweighs the cheaper rates, routing lost money and the
    dashboard has to be able to say so. Zero when both sides resolve to the same
    deployment, or when either cannot be resolved or priced.

    Only one side of this subtraction is a counterfactual. What the request cost on the
    model that served it is a number the operator was actually billed, and the cost
    calculator already wrote it down, so ``cost_breakdown`` is read rather than
    re-derived. Recomputing it means restating every pricing dimension the biller
    applied, and each one omitted is a silent disagreement with the ``spend`` column
    beside it; a request billed at a priority tier recomputed at standard rates reads as
    half its real cost.

    The baseline has no such record, since it never ran, so it is priced through the same
    cost engine on the basis the biller used for this request. An operator running that
    one model instead of the router would have sent this request to the same tier and the
    same region, because both are properties of the request and the deployment's
    contract, not of which model the router happened to pick.

    ``conversation_continuing`` says whether the baseline would already have had this
    prompt cached. It defaults to True because that is the conservative reading: a
    request whose shape the router could not determine is charged the write and
    under-claims rather than inflating a savings figure.
    """
    # No provider argument for the baseline on purpose: it arrives from the routing
    # metadata as a single self-describing string, already qualified by the auto-router,
    # so there is no second field that could disagree with it.
    baseline: Final = _resolve_model(baseline_model, None)
    selected: Final = _resolve_model(selected_model, selected_provider)
    if baseline is None or selected is None:
        return 0.0
    # Same model is only the same cost when it is also the same deployment. Two
    # deployments of one model can carry different negotiated rates, and routing from
    # the dear one to the cheap one is a real saving that short-circuiting on the model
    # name alone reports as zero.
    if baseline == selected:
        return 0.0
    basis: Final = _pricing_basis(cost_breakdown)
    effective_baseline_info: Final = baseline_info if baseline_info is not None else _model_info(baseline)
    baseline_cost: Final = _cost_of_usage(
        baseline,
        _baseline_usage(usage, conversation_continuing, effective_baseline_info),
        effective_baseline_info,
        basis,
    )
    # Falls back to pricing the request only when the biller recorded nothing, which is
    # every row written before the breakdown carried its basis.
    selected_cost = _recorded_token_cost(cost_breakdown)
    if selected_cost is None:
        selected_cost = _cost_of_usage(selected, usage, selected_info, basis)
    if baseline_cost is None or selected_cost is None:
        return 0.0
    return baseline_cost - selected_cost


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


def compute_savings_spend(
    model: str | None,
    custom_llm_provider: str | None,
    compression_saved_tokens: int,
    routing_decision: Mapping[str, object] | None = None,
    usage_object: Mapping[str, object] | None = None,
    model_id: str | None = None,
    llm_router: "Callable[[], Router | None] | None" = None,
    cost_breakdown: Mapping[str, object] | None = None,
) -> SavingsSpend:
    """
    Dollar savings for one request, split by optimization driver.

    Compression savings price the tokens compression removed at the model's
    input rate. Prompt-caching savings price the cache-read tokens at the
    difference between the input rate and the discounted cache-read rate; the
    read count is derived here from ``usage_object`` so no caller can hand in a
    count that disagrees with the usage record. Auto-router savings compare the
    served ``model`` against the counterfactual baseline the router recorded on
    its ``routing_decision``, and are zero unless the two differ. That record
    also says whether the conversation was already underway, which is what tells
    a mid-conversation switch from a first turn.

    ``llm_router`` is passed as a provider rather than a router because every spend write
    calls this and only auto-routed ones need one, so looking it up eagerly at the call
    site would fetch and discard it on the rest.

    ``cost_breakdown`` is what the cost calculator recorded for this request, and it
    carries both what the request really cost and the tier and region it was priced on.
    Only the auto-router driver reads it. Compression and prompt caching price a
    hypothetical token delta off flat rate keys, so they are blind to tiered pricing in
    the same way; that is pre-existing behaviour on two shipped drivers rather than
    something introduced here, and moving those numbers is its own change.
    """
    input_cost, cache_read_cost = _input_and_cache_read_cost(model, custom_llm_provider)
    compression: Final = max(compression_saved_tokens, 0) * input_cost
    cache_read_input_tokens: Final = extract_cache_read_tokens(usage_object)
    prompt_caching: Final = max(cache_read_input_tokens, 0) * max(input_cost - cache_read_cost, 0.0)

    usage: Final = _usage_from_spend_log(usage_object)
    if usage is None or not model:
        return SavingsSpend(compression=compression, prompt_caching=prompt_caching)

    # The configured `autorouter_savings_baseline_model` wins; otherwise the baseline
    # the deciding router recorded on its decision; neither means the driver is off.
    decision: Final = routing_decision if isinstance(routing_decision, Mapping) else {}
    recorded: Final = decision.get("savings_baseline_model")
    recorded_id: Final = decision.get("savings_baseline_deployment_id")
    configured: Final = litellm.autorouter_savings_baseline_model
    baseline_model: Final = configured or (recorded if isinstance(recorded, str) else None)
    baseline_id: Final = recorded_id if configured is None and isinstance(recorded_id, str) else None
    autorouter: Final = (
        compute_autorouter_savings(
            baseline_model=baseline_model,
            selected_model=model,
            selected_provider=custom_llm_provider,
            usage=usage,
            # Absent means the router never recorded a shape, which is the conservative
            # reading: charge the cache write rather than claim a first turn's saving.
            conversation_continuing=decision.get("conversation_continuing") is not False,
            selected_info=_effective_model_info(
                (router_instance := llm_router() if llm_router else None), model_id, model or ""
            ),
            baseline_info=_effective_model_info(router_instance, baseline_id, baseline_model or ""),
            cost_breakdown=cost_breakdown,
        )
        if decision and baseline_model
        else 0.0
    )
    return SavingsSpend(compression=compression, prompt_caching=prompt_caching, autorouter=autorouter)
