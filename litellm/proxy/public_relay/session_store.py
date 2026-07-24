from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol, TypeVar, cast  # noqa: TID251, RUF100  # DualCache returns backend clients dynamically.

from pydantic import BaseModel

from litellm.caching import DualCache
from litellm.proxy.public_relay.config import PublicRelaySettings
from litellm.proxy.public_relay.security import hash_session_token


class VerificationRecord(BaseModel):
    code_hash: str
    purpose: str
    email: str


class PortalSession(BaseModel):
    account_id: str
    user_id: str
    email: str
    session_version: int
    csrf_token: str


CachedModel = TypeVar("CachedModel", bound=BaseModel)


class RelayRedis(Protocol):
    async def async_increment(self, *, key: str, value: float, ttl: int) -> float: ...

    async def async_set_cache(self, *, key: str, value: str, ttl: int) -> bool | None: ...

    async def async_get_cache(self, *, key: str) -> object: ...

    async def async_delete_cache(self, *, key: str) -> bool: ...


@dataclass(frozen=True, slots=True)
class RelayCache:
    cache: DualCache
    settings: PublicRelaySettings

    def require_redis(self) -> None:
        if self.cache.redis_cache is None:
            raise RuntimeError("public relay requires a shared Redis connection")

    def redis(self) -> RelayRedis:
        self.require_redis()
        value = self.cache.redis_cache
        if value is None:
            raise RuntimeError("public relay requires a shared Redis connection")
        return cast(RelayRedis, value)  # cast-ok: the callable Redis methods are checked before use.

    async def enforce_limit(self, key: str, limit: int, ttl_seconds: int) -> None:
        count = await self.redis().async_increment(key=key, value=1, ttl=ttl_seconds)
        if int(count) > limit:
            raise PermissionError("rate limit exceeded")

    async def put_verification(self, record: VerificationRecord) -> None:
        await self.redis().async_set_cache(
            key=_verification_key(record.purpose, record.email),
            value=record.model_dump_json(),
            ttl=self.settings.verification_ttl_seconds,
        )

    async def get_verification(self, purpose: str, email: str) -> VerificationRecord | None:
        raw = await self.redis().async_get_cache(key=_verification_key(purpose, email))
        return _validate_cached(raw, VerificationRecord)

    async def delete_verification(self, purpose: str, email: str) -> None:
        await self.redis().async_delete_cache(key=_verification_key(purpose, email))

    async def create_session(self, token: str, session: PortalSession) -> None:
        token_hash = hash_session_token(self.settings.session_secret, token)
        await self.redis().async_set_cache(
            key=f"public-relay:session:{token_hash}",
            value=session.model_dump_json(),
            ttl=self.settings.session_ttl_seconds,
        )

    async def get_session(self, token: str) -> PortalSession | None:
        token_hash = hash_session_token(self.settings.session_secret, token)
        raw = await self.redis().async_get_cache(key=f"public-relay:session:{token_hash}")
        return _validate_cached(raw, PortalSession)

    async def delete_session(self, token: str) -> None:
        token_hash = hash_session_token(self.settings.session_secret, token)
        await self.redis().async_delete_cache(key=f"public-relay:session:{token_hash}")


def _verification_key(purpose: str, email: str) -> str:
    return f"public-relay:verification:{purpose}:{email}"


def _validate_cached(raw: object, model: type[CachedModel]) -> CachedModel | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        return model.model_validate_json(raw)
    if isinstance(raw, bytes):
        return model.model_validate_json(raw.decode())
    if isinstance(raw, dict):
        return model.model_validate(raw)
    if isinstance(raw, (int, float, bool)):
        return model.model_validate(json.loads(str(raw)))
    return None
