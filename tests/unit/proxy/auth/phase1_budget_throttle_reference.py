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
from typing import Final

import litellm
from litellm.proxy._types import UserAPIKeyAuth


def budget_throttle_percentage() -> float | None:

    pct: Final = litellm.budget_exceeded_throttle_percentage
    if not isinstance(pct, (int, float)) or isinstance(pct, bool):
        return None
    if not 0 < pct <= 1:
        return None
    return float(pct)


def should_throttle_budget_exceeded(valid_token: UserAPIKeyAuth) -> bool:

    if (valid_token.metadata or {}).get("throttle_on_budget_exceeded") is not True:  # pyright: ignore[reportUnknownMemberType]  # frozen Phase 1 accepts unvalidated legacy values
        return False
    if valid_token.tpm_limit is None and valid_token.rpm_limit is None:
        return False
    return budget_throttle_percentage() is not None


def throttled_limit(limit: int | None, pct: float | None) -> int | None:

    if limit is None or pct is None:
        return limit
    return max(1, math.floor(limit * pct))


__all__ = ()
