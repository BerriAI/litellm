"""Injectable time source so tests pin both wall-clock and elapsed time."""

import time
from datetime import datetime, timezone
from typing import Protocol


class Clock(Protocol):
    def now(self) -> datetime: ...

    def monotonic_s(self) -> float: ...


class SystemClock:
    def now(self) -> datetime:
        return datetime.now(timezone.utc)

    def monotonic_s(self) -> float:
        return time.monotonic()
