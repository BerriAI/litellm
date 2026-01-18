"""Abstract destination interfaces for Focus export."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol


@dataclass(frozen=True)
class FocusTimeWindow:
    """Represents the span of data exported in a single batch."""

    start_time: datetime
    end_time: datetime
    frequency: str


class FocusDestination(Protocol):
    """Protocol for anything that can receive Focus export files."""

    async def deliver(
        self,
        *,
        content: bytes,
        time_window: FocusTimeWindow,
        filename: str,
    ) -> None:
        """Persist the serialized export for the provided time window."""
        ...
