from __future__ import annotations

import json
from typing import TYPE_CHECKING, Generic, Optional, cast

from litellm.proxy.auth_v2.sessions.base import SessionValue

if TYPE_CHECKING:
    from redis.asyncio import Redis


class RedisSessionStore(Generic[SessionValue]):
    """Redis-backed store. Shared across workers; Redis enforces the TTL."""

    def __init__(self, client: "Redis", namespace: str, default_ttl: int) -> None:
        self._client = client
        self._namespace = namespace
        self._default_ttl = default_ttl

    def _key(self, key: str) -> str:
        return f"{self._namespace}:{key}"

    def _ttl(self, ttl_seconds: Optional[int]) -> int:
        return self._default_ttl if ttl_seconds is None else ttl_seconds

    async def get(self, key: str) -> Optional[SessionValue]:
        raw = await self._client.get(self._key(key))
        return cast(SessionValue, json.loads(raw)) if raw is not None else None

    async def set(
        self, key: str, value: SessionValue, ttl_seconds: Optional[int] = None
    ) -> None:
        await self._client.set(
            self._key(key), json.dumps(value), ex=self._ttl(ttl_seconds)
        )

    async def pop(self, key: str) -> Optional[SessionValue]:
        raw = await self._client.getdel(self._key(key))
        return cast(SessionValue, json.loads(raw)) if raw is not None else None

    async def delete(self, key: str) -> None:
        await self._client.delete(self._key(key))

    async def add_if_absent(self, key: str, ttl_seconds: Optional[int] = None) -> bool:
        added = await self._client.set(
            self._key(key), "1", nx=True, ex=self._ttl(ttl_seconds)
        )
        return bool(added)
