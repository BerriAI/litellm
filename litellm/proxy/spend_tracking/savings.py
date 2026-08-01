"""
Per-request cost-savings computation for the Cost Optimization dashboard.

Turns the token-level savings recorded on a request into dollar amounts using
the model's own pricing. Daily rollup rows are keyed by date and entity, not by
model, so the dollars have to be computed here (where the model and its prices
are known) and summed into the daily tables; tokens cannot be priced after they
have been aggregated across models.
"""

from typing import NamedTuple

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.llm_cost_calc.utils import generic_cost_per_token
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


def _cost_of_usage(model: _ModelIdentity, usage: Usage) -> float | None:
    """What ``usage`` costs on ``model``, or ``None`` when the model has no pricing."""
    try:
        prompt_cost, completion_cost = generic_cost_per_token(
            model=model.model, usage=usage, custom_llm_provider=model.provider
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


def _is_mid_conversation_switch(
    previous_model: _ModelIdentity | None, selected: _ModelIdentity, session_tracked: bool
) -> bool:
    """Whether some other model was already warm for this conversation.

    Three states, not two. An untracked request named no session at all, so nothing
    can say why its cache is cold and it stays on the conservative rule, which
    under-claims rather than inflates. A tracked request with no previous model is a
    genuine first turn, and one whose previous model is the selected one stayed put;
    neither is a switch, and both would have paid the same write on a single-model
    deployment. Only a tracked request served something else before is.
    """
    if not session_tracked:
        return True
    return previous_model is not None and previous_model != selected


def _baseline_usage(usage: Usage, is_switch: bool) -> Usage:
    """The same request as a single-model baseline would have met it.

    On a switch, the model that was already serving this conversation had the prompt
    cached, so whatever this request paid to write would have been a read on the
    baseline. The write is the switch's own cost and has to count against the saving,
    which is why the cache tokens move into the read bucket here.

    On a first turn nothing was cached anywhere, so the baseline would have paid the
    same write; leaving the usage alone lets both arms carry it and the saving comes
    out as the rate difference it really is. Charging the write to both cases, which
    is what having no discriminator forces, understates a genuine first turn to a
    few percent of its value and can render it as a loss.

    Only the cache buckets move. Every other field the request was priced on travels
    through untouched, audio and image and video counts among them, because the baseline
    is this same request served by a model that happened to be warm; naming the fields to
    keep instead would price the baseline on a request that never ran, and would go stale
    the next time a priced field is added.
    """
    cache_read, cache_creation = _cache_token_split(usage)
    details = usage.prompt_tokens_details
    if details is None or cache_creation <= 0 or not is_switch:
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
    previous_model: str | None = None,
    session_tracked: bool = False,
) -> float:
    """Net dollars the router saved, or cost, by serving this request on ``selected_model``.

    Signed on purpose. Switching models leaves the new one with a cold cache, so the
    request pays a cache-creation charge that staying on one model would not have
    incurred; when that charge outweighs the cheaper rates, routing lost money and the
    dashboard has to be able to say so. Zero when both sides resolve to the same
    deployment, or when either cannot be resolved or priced.

    ``previous_model`` is what this conversation was served last and ``session_tracked``
    whether the request named a session at all. Together they separate a switch from a
    first turn; without a session there is no discriminator, so every cold cache is
    charged as a switch, which understates rather than inflates.
    """
    # No provider argument for the baseline on purpose: it arrives from the routing
    # metadata as a single self-describing string, already qualified by the auto-router,
    # so there is no second field that could disagree with it.
    baseline = _resolve_model(baseline_model, None)
    selected = _resolve_model(selected_model, selected_provider)
    if baseline is None or selected is None or baseline == selected:
        return 0.0
    # Resolved, not string-compared: the previous model is recorded as the router's own
    # model-group name while the selected one arrives normalized from the spend log, so
    # raw equality reads `anthropic/claude-opus-5` as a switch away from `claude-opus-5`.
    is_switch = _is_mid_conversation_switch(_resolve_model(previous_model, None), selected, session_tracked)
    baseline_cost = _cost_of_usage(baseline, _baseline_usage(usage, is_switch=is_switch))
    selected_cost = _cost_of_usage(selected, usage)
    if baseline_cost is None or selected_cost is None:
        return 0.0
    return baseline_cost - selected_cost


def _usage_from_spend_log(usage_object: dict | None) -> Usage | None:
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
    baseline_model: str | None = None,
    usage_object: dict | None = None,
    previous_model: str | None = None,
    session_tracked: bool = False,
) -> SavingsSpend:
    """
    Dollar savings for one request, split by optimization driver.

    Compression savings price the tokens compression removed at the model's
    input rate. Prompt-caching savings price the cache-read tokens at the
    difference between the input rate and the discounted cache-read rate.
    Auto-router savings compare the served ``model`` against the counterfactual
    ``baseline_model`` and are zero unless the two differ; ``previous_model``
    tells a mid-conversation switch from a conversation's first turn.
    """
    input_cost, cache_read_cost = _input_and_cache_read_cost(model, custom_llm_provider)
    compression = max(compression_saved_tokens, 0) * input_cost
    prompt_caching = max(cache_read_input_tokens, 0) * max(input_cost - cache_read_cost, 0.0)

    usage = _usage_from_spend_log(usage_object)
    if usage is None or not model:
        return SavingsSpend(compression=compression, prompt_caching=prompt_caching)

    autorouter = compute_autorouter_savings(
        baseline_model=baseline_model,
        selected_model=model,
        selected_provider=custom_llm_provider,
        previous_model=previous_model,
        session_tracked=session_tracked,
        usage=usage,
    )
    return SavingsSpend(compression=compression, prompt_caching=prompt_caching, autorouter=autorouter)
