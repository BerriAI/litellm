"""Per-worker cache of stored BYOK credentials, keyed so peer workers can evict it over the auth cache pub/sub."""

from dataclasses import dataclass
from typing import Final

from litellm.caching.in_memory_cache import InMemoryCache
from litellm.constants import MCP_BYOK_CREDENTIAL_CACHE_MAX_SIZE, MCP_BYOK_CREDENTIAL_CACHE_TTL_SECONDS

_CACHE_KEY_PREFIX: Final = "mcp_byok_credential"


@dataclass(frozen=True, slots=True)
class CachedByokCredential:
    credential: str | None


byok_credential_cache: Final = InMemoryCache(
    max_size_in_memory=MCP_BYOK_CREDENTIAL_CACHE_MAX_SIZE,
    default_ttl=MCP_BYOK_CREDENTIAL_CACHE_TTL_SECONDS,
)


def byok_credential_cache_key(user_id: str, server_id: str) -> str:
    return f"{_CACHE_KEY_PREFIX}:{user_id}:{server_id}"


def get_cached_byok_credential(user_id: str, server_id: str) -> CachedByokCredential | None:
    cached: Final = byok_credential_cache.get_cache(  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  # InMemoryCache is untyped
        byok_credential_cache_key(user_id, server_id)
    )
    return cached if isinstance(cached, CachedByokCredential) else None


def cache_byok_credential(user_id: str, server_id: str, credential: str | None) -> None:
    byok_credential_cache.set_cache(  # pyright: ignore[reportUnknownMemberType]  # InMemoryCache is untyped
        byok_credential_cache_key(user_id, server_id),
        CachedByokCredential(credential=credential),
    )
