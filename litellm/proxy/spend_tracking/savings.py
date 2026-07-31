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


class SavingsSpend(NamedTuple):
    compression: float
    prompt_caching: float
    autorouter: float = 0.0


class _ModelRates(NamedTuple):
    input: float
    output: float
    cache_read: float
    cache_write: float


def _model_rates(model: str | None, custom_llm_provider: str | None) -> _ModelRates:
    """
    Per-token prices for a model, in its four billed dimensions.

    Falls open to all-zero rates when the model is unknown so savings degrade to
    zero rather than raising inside the spend writer. When a model has no separate
    cache-read price the cache-read rate mirrors the input rate, which yields zero
    caching savings; a missing cache-write price falls back to the input rate,
    which is what providers without a distinct cache-creation charge bill.
    """
    if not model:
        return _ModelRates(0.0, 0.0, 0.0, 0.0)
    try:
        info = litellm.get_model_info(model=model, custom_llm_provider=custom_llm_provider)
    except Exception as e:  # noqa: BLE001  # get_model_info raises bare Exception for unmapped models; degrade to zero savings
        verbose_proxy_logger.debug(
            "savings: no model info for provider=%s model=%s (%s)", custom_llm_provider, model, e
        )
        return _ModelRates(0.0, 0.0, 0.0, 0.0)
    input_cost = float(info.get("input_cost_per_token") or 0.0)
    output_cost = float(info.get("output_cost_per_token") or 0.0)
    cache_read = info.get("cache_read_input_token_cost")
    cache_write = info.get("cache_creation_input_token_cost")
    return _ModelRates(
        input=input_cost,
        output=output_cost,
        cache_read=input_cost if cache_read is None else float(cache_read),
        cache_write=input_cost if cache_write is None else float(cache_write),
    )


def compute_autorouter_savings(
    baseline_model: str | None,
    selected_model: str | None,
    baseline_provider: str | None,
    selected_provider: str | None,
    prompt_tokens: int,
    completion_tokens: int,
    cache_creation_input_tokens: int,
) -> float:
    """
    Net dollars saved by routing this request to ``selected_model`` instead of the
    counterfactual ``baseline_model``.

    The model-switch delta prices prompt tokens at each model's input rate and
    completion tokens at each model's output rate (output is typically several
    times the input rate, so pricing completions at the input rate materially
    understates the gap). The cache-write penalty is the cost of switching: a
    cold cache on the selected deployment forces a cache-creation charge that the
    baseline would not have incurred, priced at the selected model's cache-write
    rate. Cache-read discounts are deliberately excluded here; they are attributed
    to the prompt-caching driver, so folding them in would double-count.

    Returns zero when routing did not change the model or when pricing is unknown.
    Floored at zero so an escalation to a pricier model never reads as negative
    savings on the dashboard.
    """
    if not baseline_model or not selected_model or baseline_model == selected_model:
        return 0.0
    baseline = _model_rates(baseline_model, baseline_provider)
    selected = _model_rates(selected_model, selected_provider)
    baseline_cost = (prompt_tokens * baseline.input) + (completion_tokens * baseline.output)
    selected_cost = (prompt_tokens * selected.input) + (completion_tokens * selected.output)
    cache_write_penalty = max(cache_creation_input_tokens, 0) * selected.cache_write
    return max((baseline_cost - selected_cost) - cache_write_penalty, 0.0)


def compute_savings_spend(
    model: str | None,
    custom_llm_provider: str | None,
    compression_saved_tokens: int,
    cache_read_input_tokens: int,
    baseline_model: str | None = None,
    baseline_provider: str | None = None,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    cache_creation_input_tokens: int = 0,
) -> SavingsSpend:
    """
    Dollar savings for one request, split by optimization driver.

    Compression savings price the tokens compression removed at the model's
    input rate. Prompt-caching savings price the cache-read tokens at the
    difference between the input rate and the discounted cache-read rate.
    Auto-router savings compare the served ``model`` against the counterfactual
    ``baseline_model`` and are zero unless the two differ.
    """
    rates = _model_rates(model, custom_llm_provider)
    compression = max(compression_saved_tokens, 0) * rates.input
    prompt_caching = max(cache_read_input_tokens, 0) * max(rates.input - rates.cache_read, 0.0)
    autorouter = compute_autorouter_savings(
        baseline_model=baseline_model,
        selected_model=model,
        baseline_provider=baseline_provider,
        selected_provider=custom_llm_provider,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cache_creation_input_tokens=cache_creation_input_tokens,
    )
    return SavingsSpend(compression=compression, prompt_caching=prompt_caching, autorouter=autorouter)
