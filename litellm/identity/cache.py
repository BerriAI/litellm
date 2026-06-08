"""Three-layer identity cache.

Layer 1 (per-process, ~5s TTL): bounded ``InMemoryCache`` inside the
``DualCache`` we wrap. Bounds revocation staleness without round-tripping
to Redis on every request.

Layer 2 (Redis, cross-replica): the ``redis_cache`` on the same
``DualCache``. Writes go to both layers; reads fall through.

Layer 3 (Prisma): not owned here. ``store.load_identity`` calls the DB
when both cache layers miss.

Cross-table fan-out is handled via *generation counters*: when a team or
user changes, the counter for that team/user is bumped. Cached
identities carry the team/user generation they were minted under, and a
read that finds a stale generation treats the entry as a miss. This is
cheaper than enumerating every key that references a given team.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from litellm.integrations.otel.model.spans import SpanRole
from litellm.integrations.otel.runtime import traced

if TYPE_CHECKING:
    from litellm.caching.dual_cache import DualCache
    from litellm.proxy._types import UserAPIKeyAuth


IDENTITY_KEY_PREFIX = "identity:v1"
IDENTITY_GENERATION_PREFIX = "identity:gen:v1"
DEFAULT_IDENTITY_TTL_SECONDS = 5


def identity_cache_key(token_hash: str) -> str:
    return f"{IDENTITY_KEY_PREFIX}:{token_hash}"


def team_generation_key(team_id: str) -> str:
    return f"{IDENTITY_GENERATION_PREFIX}:team:{team_id}"


def user_generation_key(user_id: str) -> str:
    return f"{IDENTITY_GENERATION_PREFIX}:user:{user_id}"


def org_generation_key(org_id: str) -> str:
    return f"{IDENTITY_GENERATION_PREFIX}:org:{org_id}"


def _generation_attr_key(scope: str) -> str:
    return f"identity_cache_generation_{scope}"


def _attach_generations(uak: "UserAPIKeyAuth", generations: dict) -> None:
    """Stash the generation counters this entry was minted under.

    Stored on the model's ``metadata`` so the value survives Pydantic
    serialization round-trips through Redis (``CacheCodec``).
    """
    if not isinstance(uak.metadata, dict):
        uak.metadata = {}
    uak.metadata["__identity_cache_generations__"] = generations


def _read_generations(uak: "UserAPIKeyAuth") -> dict:
    metadata = uak.metadata if isinstance(uak.metadata, dict) else {}
    raw = metadata.get("__identity_cache_generations__")
    return raw if isinstance(raw, dict) else {}


class IdentityCache:
    """Memory + Redis facade keyed by the hashed virtual key.

    Stores ``UserAPIKeyAuth`` because the legacy carrier is what the rest
    of the proxy consumes today, and ``UserApiKeyCache`` already knows
    how to round-trip it through ``CacheCodec``. Callers that want an
    ``IdentityContext`` view should call ``uak.to_identity_context()``
    at the read site.
    """

    def __init__(
        self,
        dual_cache: "DualCache",
        ttl_seconds: int = DEFAULT_IDENTITY_TTL_SECONDS,
    ) -> None:
        self._cache = dual_cache
        self._ttl_seconds = ttl_seconds

    @traced(
        "identity.cache.get",
        role=SpanRole.DB_CALL,
        attrs=lambda result: {
            "identity.cache.layer": ("miss" if result is None else "memory_or_redis"),
        },
    )
    async def get(self, token_hash: str) -> Optional["UserAPIKeyAuth"]:
        from litellm.proxy._types import UserAPIKeyAuth

        cache_key = identity_cache_key(token_hash)
        cached = await self._cache.async_get_cache(
            key=cache_key, model_type=UserAPIKeyAuth
        )
        if cached is None:
            return None
        if await self._is_stale(cached):
            await self._cache.async_delete_cache(cache_key)
            return None
        return cached

    @traced("identity.cache.set", role=SpanRole.DB_CALL)
    async def set(self, token_hash: str, uak: "UserAPIKeyAuth") -> None:
        generations = await self._snapshot_generations_for(uak)
        _attach_generations(uak, generations)
        await self._cache.async_set_cache(
            key=identity_cache_key(token_hash),
            value=uak,
            ttl=self._ttl_seconds,
        )

    async def delete(self, token_hash: str) -> None:
        await self._cache.async_delete_cache(identity_cache_key(token_hash))

    async def _is_stale(self, uak: "UserAPIKeyAuth") -> bool:
        stored = _read_generations(uak)
        if not stored:
            return False
        current = await self._snapshot_generations_for(uak)
        return any(stored.get(k) != current.get(k) for k in stored)

    async def _snapshot_generations_for(self, uak: "UserAPIKeyAuth") -> dict:
        scopes: list[tuple[str, str]] = []
        if uak.team_id:
            scopes.append(("team", team_generation_key(uak.team_id)))
        if uak.user_id:
            scopes.append(("user", user_generation_key(uak.user_id)))
        if uak.org_id:
            scopes.append(("org", org_generation_key(uak.org_id)))
        if not scopes:
            return {}

        values = await self._cache.async_batch_get_cache(
            keys=[key for _, key in scopes]
        )
        return {
            scope: (values[index] or 0) if index < len(values) else 0
            for index, (scope, _) in enumerate(scopes)
        }

    async def bump_generation(self, scope_key: str) -> None:
        await self._cache.async_increment_cache(key=scope_key, value=1)


_identity_cache: Optional[IdentityCache] = None


def get_identity_cache(dual_cache: Optional["DualCache"] = None) -> IdentityCache:
    """Return the process-wide ``IdentityCache``, building it on first call.

    When ``dual_cache`` is omitted, the proxy's module-level cache is used.
    The first call wins; later calls ignore the argument so every consumer
    in a process shares one instance.
    """
    global _identity_cache
    if _identity_cache is not None:
        return _identity_cache

    if dual_cache is None:
        from litellm.proxy.proxy_server import user_api_key_cache as _proxy_cache

        dual_cache = _proxy_cache

    _identity_cache = IdentityCache(dual_cache=dual_cache)
    return _identity_cache
