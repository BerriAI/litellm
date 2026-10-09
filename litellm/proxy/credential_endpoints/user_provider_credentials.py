"""DB and cache helpers for per-user provider connections.

A row in ``LiteLLM_UserProviderCredentials`` binds one LiteLLM user to one
admin-named LLM credential (e.g. the user's own GitHub token behind a
``github_copilot`` per-user credential). ``credential_b64`` always stores the
payload encrypted; the request-time cache carries only ciphertext plus a
negative marker so an unconnected user does not hit the DB per request.
"""

import json
from collections.abc import Mapping, Sequence
from typing import (
    TYPE_CHECKING,
    Final,
    Protocol,
    cast,  # noqa: TID251  # narrows the untyped Redis cache to the str-keyed Protocol
)

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from litellm._internal_context import with_service_target
from litellm._logging import verbose_proxy_logger
from litellm.caching.dual_cache import DualCache

if TYPE_CHECKING:
    from litellm.caching.redis_cache import RedisCache
from litellm.constants import (
    GITHUB_COPILOT_USER_CREDENTIAL_CACHE_TTL_SECONDS,
    USER_PROVIDER_CREDENTIAL_CACHE_PREFIX,
    USER_PROVIDER_CREDENTIAL_NOT_CONNECTED,
)
from litellm.proxy.common_utils.encrypt_decrypt_utils import (
    decrypt_value_helper,
    encrypt_value_helper,
)
from litellm.repositories.chunked_in import find_many_in
from litellm.repositories.prisma_protocols import TableActions
from litellm.repositories.table_repositories import PrismaTableRepository

if TYPE_CHECKING:
    from prisma import models as prisma_models

    from litellm.proxy.utils import PrismaClient

_NOT_CONNECTED: Final = USER_PROVIDER_CREDENTIAL_NOT_CONNECTED


class GithubCopilotUserConnectionPayload(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    access_token: str = Field(repr=False)
    github_login: str


class _UserProviderCredentialsRepository(PrismaTableRepository["prisma_models.LiteLLM_UserProviderCredentials"]):
    table_name = "litellm_userprovidercredentials"


def _table(
    prisma_client: "PrismaClient",
) -> TableActions["prisma_models.LiteLLM_UserProviderCredentials"]:
    return _UserProviderCredentialsRepository(prisma_client, use_writer=True).table


def _cache_key(user_id: str, credential_name: str) -> str:
    pair: Final = json.dumps([user_id, credential_name], separators=(",", ":"))
    return f"{USER_PROVIDER_CREDENTIAL_CACHE_PREFIX}:{pair}"


def _encode(payload: GithubCopilotUserConnectionPayload) -> str:
    return encrypt_value_helper(payload.model_dump_json())


def decode_user_provider_credential(stored: str) -> GithubCopilotUserConnectionPayload | None:
    decrypted: Final = decrypt_value_helper(
        value=stored,
        key="user_provider_credential",
        exception_type="debug",
        return_original_value=False,
    )
    if decrypted is None:
        return None
    try:
        return GithubCopilotUserConnectionPayload.model_validate_json(decrypted)
    except ValidationError:
        return None


async def upsert_user_provider_credential(
    prisma_client: "PrismaClient",
    user_id: str,
    credential_name: str,
    provider: str,
    payload: GithubCopilotUserConnectionPayload,
) -> None:
    await _table(prisma_client).upsert(
        where={"user_id_credential_name": {"user_id": user_id, "credential_name": credential_name}},
        data={
            "create": {
                "user_id": user_id,
                "credential_name": credential_name,
                "provider": provider,
                "credential_b64": _encode(payload),
            },
            "update": {"credential_b64": _encode(payload), "provider": provider},
        },
    )


async def get_user_provider_credential(
    prisma_client: "PrismaClient",
    user_id: str,
    credential_name: str,
) -> GithubCopilotUserConnectionPayload | None:
    row: Final = await _table(prisma_client).find_unique(
        where={"user_id_credential_name": {"user_id": user_id, "credential_name": credential_name}}
    )
    if row is None:
        return None
    return decode_user_provider_credential(row.credential_b64)


async def delete_user_provider_credential(
    prisma_client: "PrismaClient",
    user_id: str,
    credential_name: str,
) -> GithubCopilotUserConnectionPayload | None:
    existing: Final = await get_user_provider_credential(prisma_client, user_id, credential_name)
    await _table(prisma_client).delete_many(where={"user_id": user_id, "credential_name": credential_name})
    return existing


async def delete_user_provider_credentials_for_credential(
    prisma_client: "PrismaClient",
    credential_name: str,
) -> Sequence[str]:
    """Delete every user connection under ``credential_name``; returns the
    affected user_ids so callers can invalidate their cache entries."""
    rows: Final = await _table(prisma_client).find_many(where={"credential_name": credential_name})
    user_ids: Final = tuple(dict.fromkeys(row.user_id for row in rows))
    await _table(prisma_client).delete_many(where={"credential_name": credential_name})
    return user_ids


async def list_user_provider_credentials(
    prisma_client: "PrismaClient",
    user_id: str,
) -> Sequence["prisma_models.LiteLLM_UserProviderCredentials"]:
    return await _table(prisma_client).find_many(where={"user_id": user_id})


async def list_user_provider_credentials_for_credential(
    prisma_client: "PrismaClient",
    credential_name: str,
) -> Sequence["prisma_models.LiteLLM_UserProviderCredentials"]:
    return await _table(prisma_client).find_many(where={"credential_name": credential_name})


class _StringKeyCache(Protocol):
    async def async_get_cache(
        self,
        key: str,
        **kwargs: object,  # kwargs-ok: RedisCache accepts optional cache kwargs
    ) -> object: ...

    async def async_set_cache(
        self,
        key: str,
        value: str,
        **kwargs: object,  # kwargs-ok: RedisCache accepts ttl/nx kwargs
    ) -> None: ...

    async def async_delete_cache(
        self,
        key: str,
        **kwargs: object,  # kwargs-ok: RedisCache accepts optional cache kwargs
    ) -> None: ...


def _string_cache(cache: "RedisCache") -> _StringKeyCache:
    return cast(_StringKeyCache, cache)  # cast-ok: RedisCache exposes the str-keyed cache protocol


async def _try_cache_get(token_cache: "RedisCache", key: str) -> object:
    """Redis errors must read as a plain miss: the DB is the source of truth."""
    try:
        return await _string_cache(token_cache).async_get_cache(key)
    except Exception:  # noqa: BLE001  # a Redis outage must fall back to the database, never reject a connected user
        verbose_proxy_logger.warning("aget_user_provider_tokens: Redis get failed; falling back to the database")
        return None


async def _try_cache_set(token_cache: "RedisCache", key: str, value: str, nx: bool = False) -> None:
    try:
        await _string_cache(token_cache).async_set_cache(
            key, value, nx=nx, ttl=GITHUB_COPILOT_USER_CREDENTIAL_CACHE_TTL_SECONDS
        )
    except Exception:  # noqa: BLE001  # caching is best-effort; a Redis outage is not worth failing the request
        verbose_proxy_logger.warning("aget_user_provider_tokens: Redis set failed; skipping the cache write")


@with_service_target("user_provider_connections")
async def invalidate_user_provider_credential_cache(
    cache: DualCache,
    user_id: str,
    credential_name: str,
) -> bool:
    """Write the not-connected tombstone rather than deleting: a read in flight
    before the disconnect must not fill the token back in behind it (fills are
    set-if-absent). Returns False when Redis is attached and the write fails, so
    the disconnect can refuse to drop the row while a stale token might linger."""
    token_cache: Final = cache.redis_cache
    if token_cache is None:
        return True
    try:
        await _string_cache(token_cache).async_set_cache(
            _cache_key(user_id, credential_name),
            _NOT_CONNECTED,
            ttl=GITHUB_COPILOT_USER_CREDENTIAL_CACHE_TTL_SECONDS,
        )
    except Exception:  # noqa: BLE001  # caller decides whether a Redis outage aborts the disconnect
        verbose_proxy_logger.warning("invalidate_user_provider_credential_cache: Redis tombstone failed")
        return False
    # async_set_cache swallows client errors internally, so a write that never
    # landed looks identical to a success. The tombstone is only trusted when the
    # key reads back as _NOT_CONNECTED; anything else means revoke failed.
    readback: Final = await _try_cache_get(token_cache, _cache_key(user_id, credential_name))
    if readback != _NOT_CONNECTED:
        verbose_proxy_logger.warning("invalidate_user_provider_credential_cache: tombstone not visible after write")
        return False
    return True


@with_service_target("user_provider_connections")
async def drop_user_provider_credential_cache(
    cache: DualCache,
    user_id: str,
    credential_name: str,
) -> None:
    """Connect path: delete the key outright so the next read fills fresh (a
    tombstone would linger for the TTL and hide the new connection)."""
    token_cache: Final = cache.redis_cache
    if token_cache is None:
        return
    try:
        await _string_cache(token_cache).async_delete_cache(_cache_key(user_id, credential_name))
    except Exception:  # noqa: BLE001  # a Redis outage must not fail a connect
        verbose_proxy_logger.warning("drop_user_provider_credential_cache: Redis delete failed")


@with_service_target("user_provider_connections")
async def set_user_provider_credential_cache(
    cache: DualCache,
    user_id: str,
    credential_name: str,
    payload: GithubCopilotUserConnectionPayload,
) -> bool:
    """Connect path: overwrite the key with the new connection's ciphertext so a
    stale set-if-absent fill from a pre-connect read cannot resurrect a
    not-connected marker. The overwrite is verified by decoding the key back
    (async_set_cache swallows write errors); on a mismatch the key is deleted,
    and False means a stale entry survived both attempts."""
    token_cache: Final = cache.redis_cache
    if token_cache is None:
        return True
    key: Final = _cache_key(user_id, credential_name)
    try:
        await _string_cache(token_cache).async_set_cache(
            key,
            _encode(payload),
            ttl=GITHUB_COPILOT_USER_CREDENTIAL_CACHE_TTL_SECONDS,
        )
        readback: Final = await _try_cache_get(token_cache, key)
        if isinstance(readback, str) and decode_user_provider_credential(readback) == payload:
            return True
        verbose_proxy_logger.warning(
            "set_user_provider_credential_cache: Redis overwrite not visible; falling back to delete"
        )
    except Exception:  # noqa: BLE001  # a Redis outage must not fail a connect
        verbose_proxy_logger.warning("set_user_provider_credential_cache: Redis set failed; falling back to delete")
    try:
        await _string_cache(token_cache).async_delete_cache(key)
    except Exception:  # noqa: BLE001  # a Redis outage must not fail a connect
        verbose_proxy_logger.warning("set_user_provider_credential_cache: Redis delete failed")
    after_delete: Final = await _try_cache_get(token_cache, key)
    if after_delete is None:
        return True
    if isinstance(after_delete, str) and decode_user_provider_credential(after_delete) == payload:
        return True
    verbose_proxy_logger.warning("set_user_provider_credential_cache: stale entry survived delete")
    return False


@with_service_target("user_provider_connections")
async def aget_user_provider_tokens(
    prisma_client: "PrismaClient",
    cache: DualCache,
    user_id: str,
    credential_names: Sequence[str],
) -> Mapping[str, str]:
    """Map each per-user credential name to the caller's stored GitHub token.

    Cached values are the stored ciphertext or the ``_NOT_CONNECTED`` marker;
    plaintext tokens only ever live in the returned dict. Redis is the only
    cache layer used: without it there is no shared invalidation, so a worker
    serving a stale local entry after a disconnect is worse than a DB read."""
    token_cache: Final = cache.redis_cache
    names: Final = tuple(dict.fromkeys(credential_names))
    if not names:
        return {}
    cached: Final[dict[str, str]] = {}  # mutable-ok: accumulates hits and DB reads
    misses: Final[list[str]] = []  # mutable-ok: accumulates cache misses
    for name in names:
        value = await _try_cache_get(token_cache, _cache_key(user_id, name)) if token_cache is not None else None
        if value == _NOT_CONNECTED:
            continue
        if isinstance(value, str) and value:
            cached[name] = value
        else:
            misses.append(name)
    if not misses:
        return {
            name: token
            for name, ciphertext in cached.items()
            if (token := _token_from_ciphertext(ciphertext)) is not None
        }

    try:
        rows: Final = await find_many_in(
            _table(prisma_client),
            "credential_name",
            misses,
            where={"user_id": user_id},
        )
    except Exception:  # noqa: BLE001  # a DB outage reads as not connected so the caller gets a 401, never a 500
        verbose_proxy_logger.exception("aget_user_provider_tokens: DB read failed for user_id=%s", user_id)
        return {}
    found: Final = {row.credential_name: row.credential_b64 for row in rows}
    for name in misses:
        if token_cache is not None:
            await _try_cache_set(token_cache, _cache_key(user_id, name), found.get(name, _NOT_CONNECTED), nx=True)
        if name in found:
            cached[name] = found[name]
    return {
        name: token for name, ciphertext in cached.items() if (token := _token_from_ciphertext(ciphertext)) is not None
    }


def _token_from_ciphertext(ciphertext: str) -> str | None:
    payload: Final = decode_user_provider_credential(ciphertext)
    return payload.access_token if payload is not None else None
