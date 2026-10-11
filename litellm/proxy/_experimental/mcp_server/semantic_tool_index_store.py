"""
Redis-shared vector store for the MCP semantic tool index.

Lets every pod reuse embeddings another pod already computed instead of
re-embedding every tool description at startup.
"""

import base64
import sys
from array import array
from collections.abc import Mapping, Sequence
from hashlib import sha256
from typing import TYPE_CHECKING, Final, Protocol

from pydantic import TypeAdapter

from litellm._internal_context import with_service_target
from litellm._logging import verbose_logger
from litellm.proxy.db.db_transaction_queue.pod_lock_manager import PodLockManager

if TYPE_CHECKING:
    from litellm.caching.redis_cache import RedisCache

_KEY_PREFIX: Final = "litellm:mcp_semantic_tool_index:v1:"
_BUILD_LOCK_ID: Final = "mcp_semantic_tool_index_build"
_BUILD_LOCK_TTL_S: Final = 300
_VECTOR_TTL_S: Final = 30 * 24 * 60 * 60
_PUT_CHUNK_SIZE: Final = 50
_CACHE_TARGET: Final = "mcp_semantic_tool_index"


def tool_vector_key(embedding_identity: str, text: str) -> str:
    """Cache key for one embedded description; identity changes invalidate every entry."""
    return _KEY_PREFIX + sha256((embedding_identity + "\0" + text).encode("utf-8")).hexdigest()


class ToolVectorStore(Protocol):
    """Shared embedding cache plus a cross-pod build lock."""

    async def get_many(self, keys: Sequence[str]) -> Mapping[str, tuple[float, ...]]:
        """Return only the keys that hit."""
        ...

    async def put_many(self, vectors: Mapping[str, Sequence[float]]) -> None: ...

    async def try_acquire_build_lock(self) -> bool: ...

    async def release_build_lock(self) -> None: ...


class RedisToolVectorStore:
    """ToolVectorStore over RedisCache; every Redis failure degrades to a local embed."""

    def __init__(self, redis_cache: "RedisCache", lock_manager: PodLockManager) -> None:
        self._redis_cache = redis_cache
        self._lock_manager = lock_manager
        self._degraded = False

    @with_service_target(_CACHE_TARGET)
    async def get_many(self, keys: Sequence[str]) -> Mapping[str, tuple[float, ...]]:
        try:
            raw: Final = TypeAdapter(Mapping[str, object]).validate_python(
                await self._redis_cache.async_batch_get_cache(list(keys))  # pyright: ignore[reportUnknownMemberType]  # RedisCache declares a partially untyped signature
            )
        except Exception as e:  # noqa: BLE001  # any redis failure must degrade to a local embed, never raise
            self._degraded = True
            verbose_logger.warning("MCP semantic tool index store get_many failed; embedding locally: %s", e)
            return {}
        return {key: vector for key in keys if (vector := _decode_vector(raw.get(key))) is not None}

    @with_service_target(_CACHE_TARGET)
    async def put_many(self, vectors: Mapping[str, Sequence[float]]) -> None:
        try:
            entries: Final = tuple((key, _encode_vector(vector)) for key, vector in vectors.items())
            for start in range(0, len(entries), _PUT_CHUNK_SIZE):
                await self._redis_cache.async_set_cache_pipeline(  # pyright: ignore[reportUnknownMemberType]  # RedisCache declares a partially untyped signature
                    entries[start : start + _PUT_CHUNK_SIZE], ttl=_VECTOR_TTL_S
                )
        except Exception as e:  # noqa: BLE001  # any redis failure must degrade to a local embed, never raise
            self._degraded = True
            verbose_logger.warning("MCP semantic tool index store put_many failed; vectors stay local: %s", e)

    async def try_acquire_build_lock(self) -> bool:
        # PodLockManager returns False on Redis errors too; once degraded, embed locally instead of waiting.
        try:
            acquired: Final = await self._lock_manager.acquire_lock(
                _BUILD_LOCK_ID, ttl=_BUILD_LOCK_TTL_S, allow_reentrant=True
            )
            return bool(acquired) or self._degraded
        except Exception as e:  # noqa: BLE001  # any redis failure must degrade to a local embed, never raise
            self._degraded = True
            verbose_logger.warning("MCP semantic tool index store lock acquire failed; embedding locally: %s", e)
            return True

    async def release_build_lock(self) -> None:
        try:
            await self._lock_manager.release_lock(_BUILD_LOCK_ID)
        except Exception as e:  # noqa: BLE001  # any redis failure must degrade to a local embed, never raise
            verbose_logger.warning("MCP semantic tool index store lock release failed: %s", e)


def _encode_vector(vector: Sequence[float]) -> str:
    packed: Final = array("f", vector)
    if sys.byteorder == "big":
        packed.byteswap()
    return base64.b64encode(packed.tobytes()).decode("ascii")


def _decode_vector(value: object) -> tuple[float, ...] | None:
    try:
        if not isinstance(value, (str, bytes, bytearray)):
            return None
        raw: Final = base64.b64decode(value)
        vector: Final = array("f")
        vector.frombytes(raw)
        if sys.byteorder == "big":
            vector.byteswap()
        return tuple(vector)
    except Exception:  # noqa: BLE001  # corrupt payloads are treated as a plain cache miss
        return None
