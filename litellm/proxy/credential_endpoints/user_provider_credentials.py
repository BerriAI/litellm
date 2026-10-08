"""DB and cache helpers for per-user provider connections.

A row in ``LiteLLM_UserProviderCredentials`` binds one LiteLLM user to one
admin-named LLM credential (e.g. the user's own GitHub token behind a
``github_copilot`` per-user credential). ``credential_b64`` always stores the
payload encrypted; the request-time cache carries only ciphertext plus a
negative marker so an unconnected user does not hit the DB per request.
"""

import json
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Final

from pydantic import BaseModel, ConfigDict, Field

from litellm._internal_context import with_service_target
from litellm._logging import verbose_proxy_logger
from litellm.caching.dual_cache import DualCache
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
    model_config = ConfigDict(extra="ignore")

    access_token: str = Field(repr=False)
    github_login: str


class _UserProviderCredentialsRepository(PrismaTableRepository["prisma_models.LiteLLM_UserProviderCredentials"]):
    table_name = "litellm_userprovidercredentials"


def _table(
    prisma_client: "PrismaClient",
) -> TableActions["prisma_models.LiteLLM_UserProviderCredentials"]:
    return _UserProviderCredentialsRepository(prisma_client).table


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
    except Exception:
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


@with_service_target("user_provider_connections")
async def invalidate_user_provider_credential_cache(
    cache: DualCache,
    user_id: str,
    credential_name: str,
) -> None:
    await cache.async_delete_cache(_cache_key(user_id, credential_name))


@with_service_target("user_provider_connections")
async def aget_user_provider_tokens(
    prisma_client: "PrismaClient",
    cache: DualCache,
    user_id: str,
    credential_names: Sequence[str],
) -> Mapping[str, str]:
    """Map each per-user credential name to the caller's stored GitHub token.

    Cached values are the stored ciphertext or the ``_NOT_CONNECTED`` marker;
    plaintext tokens only ever live in the returned dict."""
    names: Final = tuple(dict.fromkeys(credential_names))
    if not names:
        return {}
    cached: dict[str, str] = {}  # mutable-ok: accumulates hits and DB reads
    misses: list[str] = []  # mutable-ok: accumulates cache misses
    for name in names:
        value = await cache.async_get_cache(_cache_key(user_id, name))
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
    except Exception:
        verbose_proxy_logger.exception("aget_user_provider_tokens: DB read failed for user_id=%s", user_id)
        return {}
    found: dict[str, str] = {}  # mutable-ok: accumulates rows
    for row in rows:
        found[row.credential_name] = row.credential_b64
    for name in misses:
        await cache.async_set_cache(
            _cache_key(user_id, name),
            found.get(name, _NOT_CONNECTED),
            ttl=GITHUB_COPILOT_USER_CREDENTIAL_CACHE_TTL_SECONDS,
        )
        if name in found:
            cached[name] = found[name]
    return {
        name: token for name, ciphertext in cached.items() if (token := _token_from_ciphertext(ciphertext)) is not None
    }


def _token_from_ciphertext(ciphertext: str) -> str | None:
    payload: Final = decode_user_provider_credential(ciphertext)
    return payload.access_token if payload is not None else None
