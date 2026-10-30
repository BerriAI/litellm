"""DB and cache helpers for per-user provider connections.

A row in ``LiteLLM_UserProviderCredentials`` binds one LiteLLM user to one
admin-named LLM credential (e.g. the user's own GitHub token behind a
``github_copilot`` per-user credential). ``credential_b64`` always stores the
payload encrypted; the request-time cache carries only ciphertext plus a
negative marker so an unconnected user does not hit the DB per request.
"""

from collections.abc import Mapping, Sequence
from datetime import timedelta
from typing import TYPE_CHECKING, Final

from litellm._internal_context import with_service_target
from litellm._logging import verbose_proxy_logger
from litellm.caching.dual_cache import DualCache

if TYPE_CHECKING:
    from litellm.caching.redis_cache import RedisCache
from litellm.constants import (
    GITHUB_COPILOT_USER_CREDENTIAL_CACHE_TTL_SECONDS,
    USER_PROVIDER_CREDENTIAL_NOT_CONNECTED,
)
from litellm.redis.stores.user_provider_credentials import (
    GithubCopilotUserConnectionPayload,
    InvalidCredentialValue,
    TransportUserProviderCredentialStore,
    UserProviderCredentialStore,
    decode_credential,
    encode_credential,
    user_provider_credential_key,
)
from litellm.redis.transport.contract import RedisFailure
from litellm.repositories.chunked_in import find_many_in
from litellm.repositories.prisma_protocols import TableActions
from litellm.repositories.table_repositories import PrismaTableRepository

if TYPE_CHECKING:
    from prisma import models as prisma_models

    from litellm.proxy.utils import PrismaClient

_NOT_CONNECTED: Final = USER_PROVIDER_CREDENTIAL_NOT_CONNECTED
_cache_key: Final = user_provider_credential_key


class _UserProviderCredentialsRepository(PrismaTableRepository["prisma_models.LiteLLM_UserProviderCredentials"]):
    table_name = "litellm_userprovidercredentials"


def _table(
    prisma_client: "PrismaClient",
) -> TableActions["prisma_models.LiteLLM_UserProviderCredentials"]:
    return _UserProviderCredentialsRepository(prisma_client, use_writer=True).table


def _encode(payload: GithubCopilotUserConnectionPayload) -> str:
    return encode_credential(payload)


def decode_user_provider_credential(stored: str) -> GithubCopilotUserConnectionPayload | None:
    return decode_credential(stored)


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


def _store(cache: "RedisCache") -> UserProviderCredentialStore:
    return TransportUserProviderCredentialStore(cache.transport())


async def _try_cache_get(
    token_cache: "RedisCache", user_id: str, credential_name: str
) -> str | None | InvalidCredentialValue:
    result: Final = await _store(token_cache).read(user_id, credential_name)
    return None if isinstance(result, RedisFailure) else result


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
    await _store(token_cache).disconnect(
        user_id, credential_name, ttl=timedelta(seconds=GITHUB_COPILOT_USER_CREDENTIAL_CACHE_TTL_SECONDS)
    )
    readback: Final = await _try_cache_get(token_cache, user_id, credential_name)
    return readback == _NOT_CONNECTED


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
    await _store(token_cache).delete(user_id, credential_name)


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
    write: Final = await _store(token_cache).connect(
        user_id,
        credential_name,
        payload,
        ttl=timedelta(seconds=GITHUB_COPILOT_USER_CREDENTIAL_CACHE_TTL_SECONDS),
    )
    if not (isinstance(write, RedisFailure) and write.kind == "decode"):
        readback: Final = await _try_cache_get(token_cache, user_id, credential_name)
        if isinstance(readback, str) and decode_user_provider_credential(readback) == payload:
            return True
    await _store(token_cache).delete(user_id, credential_name)
    after_delete: Final = await _try_cache_get(token_cache, user_id, credential_name)
    if after_delete is None:
        return True
    return isinstance(after_delete, str) and decode_user_provider_credential(after_delete) == payload


@with_service_target("user_provider_connections")
async def aget_user_provider_credentials(
    prisma_client: "PrismaClient",
    cache: DualCache,
    user_id: str,
    credential_names: Sequence[str],
) -> Mapping[str, Mapping[str, str]]:
    """Map each per-user credential name to the caller's stored GitHub connection.

    Cached values are the stored ciphertext or the ``_NOT_CONNECTED`` marker;
    plaintext connection data only ever lives in the returned mapping. Redis is the only
    cache layer used: without it there is no shared invalidation, so a worker
    serving a stale local entry after a disconnect is worse than a DB read."""
    token_cache: Final = cache.redis_cache
    names: Final = tuple(dict.fromkeys(credential_names))
    if not names:
        return {}
    cached: Final[dict[str, str]] = {}  # mutable-ok: accumulates hits and DB reads
    misses: Final[list[str]] = []  # mutable-ok: accumulates cache misses
    for name in names:
        if (
            value := await _try_cache_get(token_cache, user_id, name) if token_cache is not None else None
        ) == _NOT_CONNECTED:
            continue
        if isinstance(value, str) and value:
            cached[name] = value
        else:
            misses.append(name)
    if not misses:
        return {
            name: connection
            for name, ciphertext in cached.items()
            if (connection := _connection_from_ciphertext(ciphertext)) is not None
        }

    try:
        rows: Final = await find_many_in(
            _table(prisma_client),
            "credential_name",
            misses,
            where={"user_id": user_id},
        )
    except Exception:  # noqa: BLE001  # a DB outage reads as not connected so the caller gets a 401, never a 500
        verbose_proxy_logger.exception("aget_user_provider_credentials: DB read failed")
        return {}
    found: Final = {row.credential_name: row.credential_b64 for row in rows}
    for name in misses:
        if token_cache is not None:
            await _store(token_cache).fill(
                user_id, name, found.get(name), ttl=timedelta(seconds=GITHUB_COPILOT_USER_CREDENTIAL_CACHE_TTL_SECONDS)
            )
        if name in found:
            cached[name] = found[name]
    return {
        name: found_connection
        for name, ciphertext in cached.items()
        if (found_connection := _connection_from_ciphertext(ciphertext)) is not None
    }


def _connection_from_ciphertext(ciphertext: str) -> Mapping[str, str] | None:
    payload: Final = decode_user_provider_credential(ciphertext)
    if payload is None:
        return None
    return {"access_token": payload.access_token, "github_host": payload.github_host}
