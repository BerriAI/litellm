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
