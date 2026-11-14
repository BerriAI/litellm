from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias


@dataclass(frozen=True, slots=True)
class DeploymentPin:
    model_group: str
    caller: str


@dataclass(frozen=True, slots=True)
class DeploymentSessionPin:
    model_group: str
    caller: str | None
    session_id: str


@dataclass(frozen=True, slots=True)
class TierSessionPin:
    router: str
    caller: str | None
    session_id: str


@dataclass(frozen=True, slots=True)
class TierModelPin:
    router: str
    tiers: str
    caller: str | None
    session_id: str
    tier: str
    replay_fallback: str | None


PinKey: TypeAlias = DeploymentPin | DeploymentSessionPin | TierSessionPin | TierModelPin
