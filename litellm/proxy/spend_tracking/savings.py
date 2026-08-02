"""
Per-request cost-savings computation for the Cost Optimization dashboard.

Turns the token-level savings recorded on a request into dollar amounts using
the model's own pricing. Daily rollup rows are keyed by date and entity, not by
model, so the dollars have to be computed here (where the model and its prices
are known) and summed into the daily tables; tokens cannot be priced after they
have been aggregated across models.
"""

from collections.abc import Mapping
from typing import NamedTuple

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.llm_cost_calc.utils import generic_cost_per_token
from litellm.router_strategy.savings_baseline import cost_key
from litellm.types.utils import PromptTokensDetailsWrapper, Usage


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
        info = litellm.get_model_info(model=model, custom_llm_provider=custom_llm_provider)
    except Exception as e:  # noqa: BLE001  # get_model_info raises bare Exception for unmapped models; degrade to zero savings
        verbose_proxy_logger.debug(
            "savings: no model info for provider=%s model=%s (%s)", custom_llm_provider, model, e
        )
        return 0.0, 0.0
    input_cost = float(info.get("input_cost_per_token") or 0.0)
    cache_read_cost = info.get("cache_read_input_token_cost")
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


def _cost_of_usage(model: _ModelIdentity, usage: Usage, pricing_key: str | None = None) -> float | None:
    """What ``usage`` costs on ``model``, or ``None`` when the model has no pricing.

    ``pricing_key`` is the key litellm bills this deployment under, which is the
    deployment's own id when it overrides its prices and the model name otherwise. The
    router keeps overrides off the shared model-name key so deployments sharing a
    backend model do not pollute each other, so pricing the name would read the public
    rate for a model nobody is charged the public rate for, on whichever arm is
    overridden, in either direction.
    """
    try:
        prompt_cost, completion_cost = generic_cost_per_token(
            model=pricing_key or model.model, usage=usage, custom_llm_provider=model.provider
        )
    except Exception as e:  # noqa: BLE001  # get_model_info raises bare Exception for unmapped models; degrade to zero savings
        verbose_proxy_logger.debug(
            "savings: cannot price usage for provider=%s model=%s (%s)", model.provider, model.model, e
        )
        return None
    return prompt_cost + completion_cost


def _cache_token_split(usage: Usage) -> tuple[int, int]:
    """``(cache_read_tokens, cache_creation_tokens)`` for a request."""
    details = usage.prompt_tokens_details
    if details is None:
        return 0, 0
    read = getattr(details, "cached_tokens", 0) or 0
    created = (getattr(details, "cache_creation_tokens", 0) or 0) or (getattr(details, "cache_write_tokens", 0) or 0)
    return int(read), int(created)


_CACHE_SPLIT_FIELDS = frozenset(
    ("cached_tokens", "cache_creation_tokens", "cache_write_tokens", "cache_creation_token_details", "text_tokens")
)


def _baseline_usage(usage: Usage, conversation_continuing: bool) -> Usage:
    """The same request as a single-model baseline would have met it.

    The baseline is one model serving every turn, so whether it had this prompt cached
    is simply whether the conversation was already underway. On a continuing
    conversation it wrote the prompt on an earlier turn and would only read it now, so
    the cache tokens move into the read bucket and whatever this request paid to write
    counts against the saving; that write is what switching models costs.

    On a conversation's first turn nothing was cached anywhere, for any model. The
    baseline would have written the same prompt, so the usage passes through untouched
    and both arms carry the write at their own rates. Charging the write to this case
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
    details = usage.prompt_tokens_details
    if details is None or cache_creation <= 0 or not conversation_continuing:
        return usage
    if cache_read > cache_creation:
        return usage
    return Usage(
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        total_tokens=usage.total_tokens,
        completion_tokens_details=usage.completion_tokens_details,
        prompt_tokens_details=PromptTokensDetailsWrapper(
            **details.model_dump(exclude=_CACHE_SPLIT_FIELDS),
            # The tokens this request paid to write are moved into the cached count and
            # the creation charge is dropped: on one model that cache was already warm,
            # so the baseline would have read them rather than paying to create them.
            # The 5m/1h breakdown goes with it; left behind it re-charges the write.
            cached_tokens=cache_read + cache_creation,
            cache_creation_tokens=0,
            cache_write_tokens=0,
            cache_creation_token_details=None,
            text_tokens=max(usage.prompt_tokens - cache_read - cache_creation, 0),
        ),
    )


def compute_autorouter_savings(
    baseline_model: str | None,
    selected_model: str | None,
    selected_provider: str | None,
    usage: Usage,
    conversation_continuing: bool = True,
    baseline_pricing_key: str | None = None,
    selected_pricing_key: str | None = None,
) -> float:
    """Net dollars the router saved, or cost, by serving this request on ``selected_model``.

    Signed on purpose. Switching models leaves the new one with a cold cache, so the
    request pays a cache-creation charge that staying on one model would not have
    incurred; when that charge outweighs the cheaper rates, routing lost money and the
    dashboard has to be able to say so. Zero when both sides resolve to the same
    deployment, or when either cannot be resolved or priced.

    ``conversation_continuing`` says whether the baseline would already have had this
    prompt cached. It defaults to True because that is the conservative reading: a
    request whose shape the router could not determine is charged the write and
    under-claims rather than inflating a savings figure.
    """
    # No provider argument for the baseline on purpose: it arrives from the routing
    # metadata as a single self-describing string, already qualified by the auto-router,
    # so there is no second field that could disagree with it.
    baseline = _resolve_model(baseline_model, None)
    selected = _resolve_model(selected_model, selected_provider)
    if baseline is None or selected is None:
        return 0.0
    # Same model is only the same cost when it is also the same deployment. Two
    # deployments of one model can carry different negotiated rates, and routing from
    # the dear one to the cheap one is a real saving that short-circuiting on the model
    # name alone reports as zero.
    if baseline == selected and baseline_pricing_key == selected_pricing_key:
        return 0.0
    baseline_cost = _cost_of_usage(baseline, _baseline_usage(usage, conversation_continuing), baseline_pricing_key)
    selected_cost = _cost_of_usage(selected, usage, selected_pricing_key)
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


def compute_savings_spend(
    model: str | None,
    custom_llm_provider: str | None,
    compression_saved_tokens: int,
    cache_read_input_tokens: int,
    routing_decision: Mapping[str, object] | None = None,
    usage_object: Mapping[str, object] | None = None,
    model_map_information: Mapping[str, object] | None = None,
    model_id: str | None = None,
) -> SavingsSpend:
    """
    Dollar savings for one request, split by optimization driver.

    Compression savings price the tokens compression removed at the model's
    input rate. Prompt-caching savings price the cache-read tokens at the
    difference between the input rate and the discounted cache-read rate.
    Auto-router savings compare the served ``model`` against the counterfactual
    baseline the router recorded on its ``routing_decision``, and are zero unless the
    two differ. That record also says whether the conversation was already underway,
    which is what tells a mid-conversation switch from a first turn.
    """
    input_cost, cache_read_cost = _input_and_cache_read_cost(model, custom_llm_provider)
    compression = max(compression_saved_tokens, 0) * input_cost
    prompt_caching = max(cache_read_input_tokens, 0) * max(input_cost - cache_read_cost, 0.0)

    usage = _usage_from_spend_log(usage_object)
    if usage is None or not model:
        return SavingsSpend(compression=compression, prompt_caching=prompt_caching)

    decision = routing_decision if isinstance(routing_decision, Mapping) else {}
    baseline_model = decision.get("savings_baseline_model")
    baseline_key = decision.get("savings_baseline_pricing_key")
    # Two inputs, because they fix different halves of the same problem and the
    # resolver needs both. `model_map_key` is the served model already resolved through
    # `base_model`, which is the only way an Azure deployment name reaches the cost map
    # at all; it is built without `router_model_id`, so it never carries a deployment's
    # own price overrides. `model_id` is the key those overrides are registered under.
    model_map = model_map_information if isinstance(model_map_information, Mapping) else {}
    mapped = model_map.get("model_map_key")
    resolved_model = mapped if isinstance(mapped, str) and mapped else model
    selected_key = cost_key(resolved_model, custom_llm_provider, model_id) or resolved_model
    autorouter = compute_autorouter_savings(
        baseline_model=baseline_model if isinstance(baseline_model, str) else None,
        selected_model=model,
        selected_provider=custom_llm_provider,
        usage=usage,
        # Absent means the router never recorded a shape, which is the conservative
        # reading: charge the cache write rather than claim a first turn's saving.
        conversation_continuing=decision.get("conversation_continuing") is not False,
        baseline_pricing_key=baseline_key if isinstance(baseline_key, str) else None,
        selected_pricing_key=selected_key,
    )
    return SavingsSpend(compression=compression, prompt_caching=prompt_caching, autorouter=autorouter)
