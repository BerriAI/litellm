"""Injectable clock, so expiry / proactive-refresh decisions are deterministic in tests.

`resolve()` never calls `datetime.now()` directly; it reads `Clock.now()`, and tests inject a
fixed clock. Part of the leaf-adapter surface the S0 chassis later bundles into `GatewayDeps`.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Protocol


class Clock(Protocol):
    def now(self) -> datetime: ...


class SystemClock:
    def now(self) -> datetime:
        return datetime.now(timezone.utc)
