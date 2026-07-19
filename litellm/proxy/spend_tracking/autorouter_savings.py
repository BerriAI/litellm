"""
Per-request autorouter cost-savings computation for the Cost Optimization
dashboard.

The autorouter (complexity router) picks a cheaper model per request instead of
sending everything to the most expensive configured candidate. The savings for
one request is the difference between what the baseline (most expensive
candidate) would have cost for the same token usage and what the routed model
actually cost. Baseline per-token prices are stamped onto the request metadata
at routing time; this module reads them back out of a parsed SpendLog
``metadata`` dict and prices the counterfactual.

The output-token count is the routed model's, not the baseline's (the baseline
was never called), so the dollar figure is an estimate, the same tradeoff the
compression and prompt-caching savings already make.
"""

from collections.abc import Mapping
from typing import NamedTuple

from pydantic import TypeAdapter, ValidationError

from litellm._logging import verbose_proxy_logger
from litellm.types.router_strategy.autorouter_savings import AutorouterSavingsMetadata

_AUTOROUTER_SAVINGS_METADATA_KEY = "autorouter_savings"
_autorouter_savings_adapter = TypeAdapter(AutorouterSavingsMetadata)


class AutorouterSavings(NamedTuple):
    savings_spend: float
    requests: int
    escalated_requests: int


_ZERO = AutorouterSavings(savings_spend=0.0, requests=0, escalated_requests=0)


def _parse_metadata(metadata: Mapping[str, object]) -> AutorouterSavingsMetadata | None:
    raw = metadata.get(_AUTOROUTER_SAVINGS_METADATA_KEY)
    if not isinstance(raw, Mapping):
        return None
    try:
        return _autorouter_savings_adapter.validate_python(dict(raw))
    except ValidationError as e:
        verbose_proxy_logger.debug("autorouter savings: malformed metadata (%s)", e)
        return None


def compute_autorouter_savings(
    metadata: Mapping[str, object],
    prompt_tokens: int,
    completion_tokens: int,
    actual_spend: float,
) -> AutorouterSavings:
    """
    Estimated dollar savings for one autorouter request, plus the request and
    escalation counters that feed the dashboard's escalation rate.

    Returns all-zero when the request was not routed by an autorouter (no
    autorouter metadata present). Savings are clamped at zero so a turn that
    happened to route to the baseline model, or to something pricier, never
    shows negative savings.
    """
    parsed = _parse_metadata(metadata)
    if parsed is None:
        return _ZERO
    baseline_cost = (
        max(prompt_tokens, 0) * parsed["baseline_input_cost_per_token"]
        + max(completion_tokens, 0) * parsed["baseline_output_cost_per_token"]
    )
    savings = max(baseline_cost - actual_spend, 0.0)
    return AutorouterSavings(
        savings_spend=savings,
        requests=1,
        escalated_requests=1 if parsed["escalated"] else 0,
    )
