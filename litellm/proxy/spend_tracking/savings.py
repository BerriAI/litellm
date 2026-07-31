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
from litellm.types.utils import Usage


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


def _cost_of_usage(model: str, custom_llm_provider: str | None, usage: Usage) -> float | None:
    """What ``usage`` costs on ``model``, or ``None`` when the model has no pricing."""
    try:
        prompt_cost, completion_cost = generic_cost_per_token(
            model=model, usage=usage, custom_llm_provider=custom_llm_provider or ""
        )
    except Exception as e:  # noqa: BLE001  # get_model_info raises bare Exception for unmapped models; degrade to zero savings
        verbose_proxy_logger.debug(
            "savings: cannot price usage for provider=%s model=%s (%s)", custom_llm_provider, model, e
        )
        return None
    return prompt_cost + completion_cost


def compute_autorouter_savings(
    baseline_model: str | None,
    selected_model: str | None,
    baseline_provider: str | None,
    selected_provider: str | None,
    usage: Usage,
) -> float:
    """Net dollars saved by serving this request on ``selected_model`` rather than ``baseline_model``.

    Both arms price the same usage, so each token is charged once in its own
    dimension; ``prompt_tokens`` already includes the cache tokens. Zero when the
    model is unchanged or unpriced, and floored at zero on an escalation.
    """
    if not baseline_model or not selected_model or baseline_model == selected_model:
        return 0.0
    baseline_cost = _cost_of_usage(baseline_model, baseline_provider, usage)
    selected_cost = _cost_of_usage(selected_model, selected_provider, usage)
    if baseline_cost is None or selected_cost is None:
        return 0.0
    return max(baseline_cost - selected_cost, 0.0)


def _usage_from_spend_log(usage_object: dict | None) -> Usage | None:
    """Rebuild the request's ``Usage`` from the copy the spend log recorded."""
    if not usage_object:
        return None
    try:
        return Usage(**usage_object)
    except Exception as e:  # noqa: BLE001  # a malformed usage_object must not fail the daily spend write
        verbose_proxy_logger.debug("savings: unusable usage_object (%s)", e)
        return None


def compute_savings_spend(
    model: str | None,
    custom_llm_provider: str | None,
    compression_saved_tokens: int,
    cache_read_input_tokens: int,
    baseline_model: str | None = None,
    baseline_provider: str | None = None,
    usage_object: dict | None = None,
) -> SavingsSpend:
    """
    Dollar savings for one request, split by optimization driver.

    Compression savings price the tokens compression removed at the model's
    input rate. Prompt-caching savings price the cache-read tokens at the
    difference between the input rate and the discounted cache-read rate.
    Auto-router savings compare the served ``model`` against the counterfactual
    ``baseline_model`` and are zero unless the two differ.
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
        baseline_provider=baseline_provider,
        selected_provider=custom_llm_provider,
        usage=usage,
    )
    return SavingsSpend(compression=compression, prompt_caching=prompt_caching, autorouter=autorouter)
