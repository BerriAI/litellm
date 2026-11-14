from __future__ import annotations

from collections.abc import Mapping
from typing import TypeAlias

from typing_extensions import ReadOnly, TypedDict


class DeploymentHealthStateValue(TypedDict):
    is_healthy: ReadOnly[bool]
    timestamp: ReadOnly[float]
    reason: ReadOnly[str]


DeploymentHealthStates: TypeAlias = Mapping[str, DeploymentHealthStateValue]
