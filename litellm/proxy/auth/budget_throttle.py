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

from litellm.proxy._types import UserAPIKeyAuth


def budget_throttle_percentage() -> float | None:
    from litellm.proxy.auth import key_budget_context_adapter

    return key_budget_context_adapter.budget_throttle_percentage()


def should_throttle_budget_exceeded(valid_token: UserAPIKeyAuth) -> bool:
    from litellm.proxy.auth import key_budget_context_adapter

    return key_budget_context_adapter.should_throttle_budget_exceeded(valid_token)


def throttled_limit(limit: int | None, pct: float | None) -> int | None:
    """
    Scale a TPM/RPM limit to ``pct`` of its value, keeping a trickle of at least
    1 so a throttled key is slowed rather than fully locked out. An unset limit
    or unset percentage leaves the limit unchanged.
    """
    if limit is None or pct is None:
        return limit
    return max(1, math.floor(limit * pct))
