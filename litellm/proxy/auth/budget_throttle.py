"""
Throttle a key after it exceeds its own ``max_budget`` instead of blocking it.

When a key opts in via ``throttle_on_budget_exceeded`` and a global
``budget_exceeded_throttle_percentage`` is configured, an over-budget key keeps
serving requests but at a reduced TPM/RPM (the configured percentage of its
configured limits). The decision (over budget + opted in) is made once during
auth; the scaling is recomputed from the key's original limits on every request
so it never compounds across requests.
"""

import math
from typing import Optional

import litellm
from litellm.proxy._types import UserAPIKeyAuth


def budget_throttle_percentage() -> Optional[float]:
    """
    The global throttle percentage, or None when throttling is disabled /
    misconfigured (in which case an over-budget key is hard-blocked, the safe
    default).
    """
    pct = litellm.budget_exceeded_throttle_percentage
    if not isinstance(pct, (int, float)) or isinstance(pct, bool):
        return None
    if not 0 < pct <= 1:
        return None
    return float(pct)


def should_throttle_budget_exceeded(valid_token: UserAPIKeyAuth) -> bool:
    """
    True when a key that exceeded its own ``max_budget`` should be throttled
    rather than blocked: it opted in, a valid global percentage is set, and the
    key has a TPM or RPM limit to scale down. A key with neither limit has
    nothing to throttle, so it stays hard-blocked (the safe default) rather than
    serving unlimited requests past its budget.
    """
    if (valid_token.metadata or {}).get("throttle_on_budget_exceeded") is not True:
        return False
    if valid_token.tpm_limit is None and valid_token.rpm_limit is None:
        return False
    return budget_throttle_percentage() is not None


def throttled_limit(limit: Optional[int], pct: Optional[float]) -> Optional[int]:
    """
    Scale a TPM/RPM limit to ``pct`` of its value, keeping a trickle of at least
    1 so a throttled key is slowed rather than fully locked out. An unset limit
    or unset percentage leaves the limit unchanged.
    """
    if limit is None or pct is None:
        return limit
    return max(1, math.floor(limit * pct))
