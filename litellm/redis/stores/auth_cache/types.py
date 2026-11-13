from __future__ import annotations

from dataclasses import dataclass
from typing import Final, TypeAlias

from litellm.rust_bridge._native import AuthObject

AUTH_CACHE_TARGET: Final = "auth_cache"

WriteBack: TypeAlias = tuple[AuthObject, object, float | None]


@dataclass(frozen=True, slots=True)
class AuthCacheSettings:
    """``UserApiKeyCache``'s memory configuration: each partition's size and ``default_in_memory_ttl``."""

    memory_entries: int
    memory_ttl: float | None
