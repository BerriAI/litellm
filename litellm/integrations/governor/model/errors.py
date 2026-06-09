"""Typed degradation signal raised by tiers and converted to a verdict."""

from typing import Literal

Tier = Literal["L1", "L2", "L3"]


class TierDegraded(Exception):
    """A cache tier could not answer authoritatively.

    Policies never decide what to do on outage; they raise this and
    ``engine.degradation`` converts it to a reject or admit-degraded verdict per
    the policy's declared ``fail_mode``. ``reason`` is an open string so new
    failure shapes (``redis_unavailable``, ``redis_timeout_ambiguous``,
    ``evicted_mid_window``, ``reconcile_against_evicted``) need no enum edit.
    """

    def __init__(self, *, tier: Tier, reason: str) -> None:
        super().__init__(f"tier={tier} reason={reason}")
        self.tier: Tier = tier
        self.reason: str = reason
