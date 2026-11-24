from __future__ import annotations

from collections.abc import Awaitable
from typing import Protocol


class SensitiveRoutePinsHandle(Protocol):
    def get(self, tenant: str, session_id: str, default_ttl_seconds: float) -> Awaitable[str | None]: ...
    def set(self, tenant: str, session_id: str, model: str, ttl_seconds: float) -> Awaitable[None]: ...
