"""Experimental secrets vault.

Lets a LiteLLM user keep named secrets on the gateway and read them back with
the same key they already use for LLM calls. Every route is scoped to the
calling key's ``user_id``; no role can read another user's secrets through it.
"""

from typing import TYPE_CHECKING, Annotated, Final, TypeAlias

from fastapi import APIRouter, Depends, HTTPException, Path, Response

from litellm.constants import UI_SESSION_TOKEN_TEAM_ID
from litellm.models.vault import (
    VAULT_SECRET_NAME_PATTERN,
    VaultSecretDeleteResponse,
    VaultSecretListResponse,
    VaultSecretMetadata,
    VaultSecretRevealed,
    VaultSecretWrite,
)
from litellm.proxy._types import CommonProxyErrors, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.utils import handle_exception_on_proxy

from .vault_secrets import (
    decrypt_vault_secret,
    delete_vault_secret,
    get_vault_secret,
    list_vault_secrets,
    mark_vault_secret_accessed,
    upsert_vault_secret,
)

if TYPE_CHECKING:
    from prisma import models as prisma_models

    from litellm.proxy.utils import PrismaClient

router: Final = APIRouter(tags=["[experimental] secrets vault"])

_NO_STORE: Final = "no-store"
AuthenticatedKey: TypeAlias = Annotated[
    UserAPIKeyAuth,
    Depends(user_api_key_auth),  # pyright: ignore[reportArgumentType]  # declared as object | callable
]

SecretName: TypeAlias = Annotated[
    str,
    Path(description="Name of the secret", pattern=VAULT_SECRET_NAME_PATTERN),
]


def _authenticated_user_id(user_api_key_dict: UserAPIKeyAuth) -> str:
    user_id: Final = user_api_key_dict.user_id
    if not isinstance(user_id, str) or not user_id:
        raise HTTPException(
            status_code=401,
            detail="The secrets vault needs a key that belongs to a user",
        )
    if user_api_key_dict.impersonated_by is not None:
        raise HTTPException(status_code=403, detail="The secrets vault cannot be used while impersonating a user")
    if user_api_key_dict.team_id not in (None, UI_SESSION_TOKEN_TEAM_ID):
        raise HTTPException(
            status_code=403,
            detail="The secrets vault cannot be used with a team key. Use a personal key instead",
        )
    return user_id


def _connected(prisma_client: "PrismaClient | None") -> "PrismaClient":
    if prisma_client is None:
        raise HTTPException(
            status_code=500,
            detail={"error": CommonProxyErrors.db_not_connected_error.value},
        )
    return prisma_client


def _metadata(row: "prisma_models.LiteLLM_VaultSecrets") -> VaultSecretMetadata:
    return VaultSecretMetadata(
        secret_name=row.secret_name,
        description=row.description,
        created_at=row.created_at,
        updated_at=row.updated_at,
        last_accessed_at=row.last_accessed_at,
    )


@router.get(
    "/vault/secrets",
    response_model=VaultSecretListResponse,
)
async def list_secrets(
    user_api_key_dict: AuthenticatedKey,
) -> VaultSecretListResponse:
    """[Experimental] List the calling user's secrets. Values are never returned here."""
    from litellm.proxy.proxy_server import prisma_client

    try:
        user_id: Final = _authenticated_user_id(user_api_key_dict)
        rows: Final = await list_vault_secrets(_connected(prisma_client), user_id)
        return VaultSecretListResponse(secrets=tuple(_metadata(row) for row in rows))
    except Exception as e:  # noqa: BLE001  # endpoint boundary: every failure becomes the proxy error contract
        raise handle_exception_on_proxy(e)


@router.put(
    "/vault/secrets/{secret_name}",
    response_model=VaultSecretMetadata,
)
async def put_secret(
    secret_name: SecretName,
    body: VaultSecretWrite,
    user_api_key_dict: AuthenticatedKey,
) -> VaultSecretMetadata:
    """[Experimental] Create the calling user's secret, or replace its value and description."""
    from litellm.proxy.proxy_server import prisma_client

    try:
        user_id: Final = _authenticated_user_id(user_api_key_dict)
        row: Final = await upsert_vault_secret(
            _connected(prisma_client),
            user_id=user_id,
            secret_name=secret_name,
            value=body.value,
            description=body.description,
        )
        return _metadata(row)
    except Exception as e:  # noqa: BLE001  # endpoint boundary: every failure becomes the proxy error contract
        raise handle_exception_on_proxy(e)


@router.get(
    "/vault/secrets/{secret_name}",
    response_model=VaultSecretRevealed,
)
async def reveal_secret(
    secret_name: SecretName,
    fastapi_response: Response,
    user_api_key_dict: AuthenticatedKey,
) -> VaultSecretRevealed:
    """[Experimental] Return the decrypted value of one of the calling user's secrets."""
    from litellm.proxy.proxy_server import prisma_client

    try:
        user_id: Final = _authenticated_user_id(user_api_key_dict)
        client: Final = _connected(prisma_client)
        row: Final = await get_vault_secret(client, user_id, secret_name)
        if row is None:
            raise HTTPException(status_code=404, detail=f"Secret '{secret_name}' not found")
        value: Final = decrypt_vault_secret(row.value_b64)
        if value is None:
            raise HTTPException(
                status_code=500,
                detail=f"Secret '{secret_name}' could not be decrypted with the current salt key",
            )
        accessed_at: Final = await mark_vault_secret_accessed(client, user_id, secret_name, row.updated_at)
        fastapi_response.headers["Cache-Control"] = _NO_STORE  # rebind-ok: FastAPI only exposes response headers here
        return VaultSecretRevealed(
            secret_name=row.secret_name,
            description=row.description,
            created_at=row.created_at,
            updated_at=row.updated_at,
            last_accessed_at=accessed_at,
            value=value,
        )
    except Exception as e:  # noqa: BLE001  # endpoint boundary: every failure becomes the proxy error contract
        raise handle_exception_on_proxy(e)


@router.delete(
    "/vault/secrets/{secret_name}",
    response_model=VaultSecretDeleteResponse,
)
async def delete_secret(
    secret_name: SecretName,
    user_api_key_dict: AuthenticatedKey,
) -> VaultSecretDeleteResponse:
    """[Experimental] Delete one of the calling user's secrets. Idempotent."""
    from litellm.proxy.proxy_server import prisma_client

    try:
        user_id: Final = _authenticated_user_id(user_api_key_dict)
        deleted: Final = await delete_vault_secret(_connected(prisma_client), user_id, secret_name)
        return VaultSecretDeleteResponse(secret_name=secret_name, deleted=deleted)
    except Exception as e:  # noqa: BLE001  # endpoint boundary: every failure becomes the proxy error contract
        raise handle_exception_on_proxy(e)
