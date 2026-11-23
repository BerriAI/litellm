from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal, TypeAlias

ROUTER_USAGE_TARGET: Final = "router_usage"
CounterKind: TypeAlias = Literal["tpm", "rpm", "itpm", "otpm", "tpd", "rpd"]
CounterScope: TypeAlias = Literal["global_router", "deployment"]
# The native store counts in i64, so a larger amount never crosses into it.
MAX_INCREMENT: Final = 2**63 - 1


@dataclass(frozen=True, slots=True)
class UsageCounter:
    """One deployment's count in one window: ``HH-MM`` for a per-minute kind, ``YYYY-MM-DD`` for a per-day one.
    ``global_router`` counters are the router core's, ``deployment`` ones those usage-based-routing-v2 and the
    model rate limit check share."""

    scope: CounterScope
    deployment_id: str
    model: str | None
    window: str
    kind: CounterKind

    @staticmethod
    def for_deployment(model_id: str, model: str, kind: CounterKind, window: str) -> UsageCounter:
        """A ``deployment`` counter. Callers spell the id and model with ``f"{value}"``, as main's
        ``f"{model_id}:{model}:..."`` keys did, so a missing one is ``"None"``."""
        return UsageCounter("deployment", model_id, model, window, kind)


@dataclass(frozen=True, slots=True)
class CounterIncrement:
    counter: UsageCounter
    amount: float
    ttl: float | None
