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


def compute_savings_spend(
    model: str | None,
    custom_llm_provider: str | None,
    compression_saved_tokens: int,
    cache_read_input_tokens: int,
) -> SavingsSpend:
    """
    Dollar savings for one request, split by optimization driver.

    Compression savings price the tokens compression removed at the model's
    input rate. Prompt-caching savings price the cache-read tokens at the
    difference between the input rate and the discounted cache-read rate.
    """
    input_cost, cache_read_cost = _input_and_cache_read_cost(model, custom_llm_provider)
    compression = max(compression_saved_tokens, 0) * input_cost
    prompt_caching = max(cache_read_input_tokens, 0) * max(input_cost - cache_read_cost, 0.0)
    return SavingsSpend(compression=compression, prompt_caching=prompt_caching)
