from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

WriteBack: TypeAlias = tuple[str, object, float | None]


@dataclass(frozen=True, slots=True)
class AuthCacheSettings:
    """``UserApiKeyCache``'s memory configuration: each partition's size and ``default_in_memory_ttl``."""

    memory_entries: int
    memory_ttl: float | None
