"""
SECURE SHARE

Endpoints backing the dashboard "Secure Share" feature: proxy admins share a
credential with an internal user over a temporary, end-to-end-encrypted link.

The browser encrypts the secret with AES-256-GCM under a key derived from an
admin-chosen password (PBKDF2-SHA256). Only the resulting ciphertext and the
crypto parameters needed to decrypt it (PBKDF2 salt, GCM iv) reach the server,
alongside an expiry. The plaintext secret and the password never leave the
client, so a database or server compromise yields ciphertext without the
password required to open it.

Links are one-time: a successful GET returns the ciphertext once and deletes
the row, so the encrypted payload is exposed to the network at most once.

POST   /secure_share/create        - store an encrypted share (proxy admin only)
GET    /secure_share/{share_id}     - fetch and consume a share once (admins + internal users)
DELETE /secure_share/{share_id}     - revoke a share early (proxy admin only)
"""

import base64
import binascii
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Final

from fastapi import APIRouter, Depends, HTTPException, Path, status
from pydantic import BaseModel, Field, field_validator

from litellm.proxy._types import (
    CommonProxyErrors,
    LitellmUserRoles,
    UserAPIKeyAuth,
)
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.repositories.table_repositories import SecureShareRepository

router = APIRouter()

MAX_CIPHERTEXT_BYTES: Final[int] = 256 * 1024

_READ_ALLOWED_ROLES: Final[frozenset[str]] = frozenset(
    {
        LitellmUserRoles.PROXY_ADMIN.value,
        LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY.value,
        LitellmUserRoles.INTERNAL_USER.value,
        LitellmUserRoles.INTERNAL_USER_VIEW_ONLY.value,
    }
)


class SecureShareExpiry(str, Enum):
    ONE_HOUR = "1h"
    SIX_HOURS = "6h"
    ONE_DAY = "1d"
    SEVEN_DAYS = "7d"

    @property
    def duration(self) -> timedelta:
        return {
            SecureShareExpiry.ONE_HOUR: timedelta(hours=1),
            SecureShareExpiry.SIX_HOURS: timedelta(hours=6),
            SecureShareExpiry.ONE_DAY: timedelta(days=1),
            SecureShareExpiry.SEVEN_DAYS: timedelta(days=7),
        }[self]


def _validate_b64(value: str, *, field_name: str, max_bytes: int) -> str:
    try:
        raw = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as e:
        raise ValueError(f"{field_name} must be base64-encoded") from e
    if len(raw) == 0:
        raise ValueError(f"{field_name} must not be empty")
    if len(raw) > max_bytes:
        raise ValueError(f"{field_name} exceeds the {max_bytes}-byte limit")
    return value


class SecureShareCreateRequest(BaseModel):
    ciphertext: str = Field(description="base64 AES-256-GCM ciphertext of the secret")
    salt: str = Field(description="base64 PBKDF2 salt used to derive the AES key")
    iv: str = Field(description="base64 AES-GCM initialization vector")
    expiry: SecureShareExpiry = Field(description="how long the share stays retrievable")

    @field_validator("ciphertext")
    @classmethod
    def _check_ciphertext(cls, value: str) -> str:
        return _validate_b64(value, field_name="ciphertext", max_bytes=MAX_CIPHERTEXT_BYTES)

    @field_validator("salt")
    @classmethod
    def _check_salt(cls, value: str) -> str:
        return _validate_b64(value, field_name="salt", max_bytes=64)

    @field_validator("iv")
    @classmethod
    def _check_iv(cls, value: str) -> str:
        return _validate_b64(value, field_name="iv", max_bytes=64)


class SecureShareCreateResponse(BaseModel):
    share_id: str
    expires_at: datetime


class SecureShareGetResponse(BaseModel):
    share_id: str
    ciphertext: str
    salt: str
    iv: str
    expires_at: datetime
    created_by: str


def _secure_share_repository() -> SecureShareRepository:
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=500, detail={"error": CommonProxyErrors.db_not_connected_error.value})
    return SecureShareRepository(prisma_client)


def _require_proxy_admin(user_api_key_dict: UserAPIKeyAuth) -> None:
    if user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN.value:
        raise HTTPException(status_code=403, detail={"error": CommonProxyErrors.not_allowed_access.value})


def _require_read_access(user_api_key_dict: UserAPIKeyAuth) -> None:
    if user_api_key_dict.user_role not in _READ_ALLOWED_ROLES:
        raise HTTPException(
            status_code=403,
            detail={"error": "Only proxy admins and internal users can view secure shares."},
        )


@router.post(
    "/secure_share/create",
    tags=["Secure Share"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=SecureShareCreateResponse,
)
async def create_secure_share(
    request: SecureShareCreateRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),  # noqa: B008  # FastAPI resolves auth from this default
) -> SecureShareCreateResponse:
    _require_proxy_admin(user_api_key_dict)

    from litellm.proxy.proxy_server import litellm_proxy_admin_name

    repository = _secure_share_repository()
    expires_at = datetime.now(timezone.utc) + request.expiry.duration
    created = await repository.table.create(
        data={
            "ciphertext": request.ciphertext,
            "salt": request.salt,
            "iv": request.iv,
            "expires_at": expires_at,
            "created_by": user_api_key_dict.user_id or litellm_proxy_admin_name,
        }
    )
    return SecureShareCreateResponse.model_validate(created, from_attributes=True)


@router.get(
    "/secure_share/{share_id}",
    tags=["Secure Share"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=SecureShareGetResponse,
)
async def get_secure_share(
    share_id: str = Path(description="id returned by /secure_share/create"),
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),  # noqa: B008  # FastAPI resolves auth from this default
) -> SecureShareGetResponse:
    _require_read_access(user_api_key_dict)

    repository = _secure_share_repository()
    row = await repository.table.find_unique(where={"share_id": share_id})
    if row is None:
        raise HTTPException(status_code=404, detail={"error": "Secure share not found."})

    share = SecureShareGetResponse.model_validate(row, from_attributes=True)
    await repository.table.delete(where={"share_id": share_id})
    if _is_expired(share.expires_at):
        raise HTTPException(status_code=status.HTTP_410_GONE, detail={"error": "Secure share has expired."})
    return share


@router.delete(
    "/secure_share/{share_id}",
    tags=["Secure Share"],
    dependencies=[Depends(user_api_key_auth)],
)
async def delete_secure_share(
    share_id: str = Path(description="id returned by /secure_share/create"),
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),  # noqa: B008  # FastAPI resolves auth from this default
) -> dict[str, str]:
    _require_proxy_admin(user_api_key_dict)

    repository = _secure_share_repository()
    deleted = await repository.table.delete(where={"share_id": share_id})
    if deleted is None:
        raise HTTPException(status_code=404, detail={"error": "Secure share not found."})
    return {"share_id": share_id, "status": "deleted"}


def _is_expired(expires_at: datetime) -> bool:
    reference = expires_at if expires_at.tzinfo is not None else expires_at.replace(tzinfo=timezone.utc)
    return reference <= datetime.now(timezone.utc)
