from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal, TypeAlias

ROUTER_USAGE_TARGET: Final = "router_usage"
CounterKind: TypeAlias = Literal["tpm", "rpm"]


@dataclass(frozen=True, slots=True)
class UsageCounter:
    deployment_id: str
    model: str | None
    minute: str
    kind: CounterKind


NativeCounter: TypeAlias = tuple[str, str | None, str, CounterKind]


def native_counter(counter: UsageCounter) -> NativeCounter:
    return (counter.deployment_id, counter.model, counter.minute, counter.kind)
