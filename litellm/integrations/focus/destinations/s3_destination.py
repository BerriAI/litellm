"""S3 destination implementation for Focus export."""

from __future__ import annotations

from typing import Any, Optional

from .base import FocusDestination, FocusTimeWindow


class FocusS3Destination(FocusDestination):
    """Handles uploading serialized exports to S3 buckets."""

    def __init__(
        self,
        *,
        prefix: str,
        config: Optional[dict[str, Any]] = None,
    ) -> None:
        self.prefix = prefix.rstrip("/")
        self.config = config or {}

    async def deliver(
        self,
        *,
        content: bytes,
        time_window: FocusTimeWindow,
        filename: str,
    ) -> None:
        raise NotImplementedError

    def _build_object_key(self, *, time_window: FocusTimeWindow, filename: str) -> str:
        raise NotImplementedError
