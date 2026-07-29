"""
Self-service password reset endpoints for internal (non-SSO) users.

/user/forgot_password
/user/reset_password/validate
/user/reset_password
"""

import asyncio
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional, cast

from fastapi import APIRouter, HTTPException, Request

from litellm._logging import verbose_proxy_logger
from litellm.models.user import LiteLLM_UserTable
from litellm.proxy._types import CommonProxyErrors
from litellm.proxy.auth.network import TrustedProxyConfig, resolve_client_ip
from litellm.proxy.auth.trusted_proxy_utils import get_trusted_proxy_cidrs
from litellm.proxy.utils import get_proxy_base_url, hash_password, hash_token, send_email
from litellm.repositories.password_reset_token_repository import (
    PasswordResetTokenRepository,
)
from litellm.repositories.user_repository import UserRepository
from litellm.types.proxy.management_endpoints.password_reset_endpoints import (
    ForgotPasswordRequest,
    ResetPasswordRequest,
)

router = APIRouter()

_GENERIC_FORGOT_PASSWORD_MESSAGE = "If an account exists for this email, a password reset link has been sent."
_GENERIC_INVALID_TOKEN_MESSAGE = "This link is invalid or has expired."
_MAX_REQUESTS_PER_EMAIL_PER_HOUR = 3
_MAX_REQUESTS_PER_IP_PER_HOUR = 10
_RESET_TOKEN_TTL_MINUTES = 30
_RATE_LIMIT_WINDOW_SECONDS = 3600


async def _send_reset_email_safely(receiver_email: str, subject: str, html: str) -> None:
    try:
        await send_email(receiver_email=receiver_email, subject=subject, html=html)
    except ValueError as e:
        verbose_proxy_logger.warning("Password reset email not sent, SMTP misconfigured: %s", e)


@router.post("/user/forgot_password", include_in_schema=False)
async def forgot_password(data: ForgotPasswordRequest, request: Request):
    from litellm.proxy.proxy_server import prisma_client, user_api_key_cache

    if prisma_client is None:
        raise HTTPException(status_code=500, detail={"error": CommonProxyErrors.db_not_connected_error.value})

    cidrs = get_trusted_proxy_cidrs()
    resolved_ip, _ = resolve_client_ip(
        request, TrustedProxyConfig(use_forwarded_for=bool(cidrs), trusted_proxy_cidrs=cidrs)
    )
    client_ip = resolved_ip or "unknown"
    email_count = await user_api_key_cache.async_increment_cache(
        key=f"password_reset_rl:email:{data.email.lower()}", value=1, ttl=_RATE_LIMIT_WINDOW_SECONDS
    )
    ip_count = await user_api_key_cache.async_increment_cache(
        key=f"password_reset_rl:ip:{client_ip}", value=1, ttl=_RATE_LIMIT_WINDOW_SECONDS
    )

    if (email_count is not None and email_count > _MAX_REQUESTS_PER_EMAIL_PER_HOUR) or (
        ip_count is not None and ip_count > _MAX_REQUESTS_PER_IP_PER_HOUR
    ):
        verbose_proxy_logger.warning("Password reset rate limit exceeded for ip=%s", client_ip)
        raise HTTPException(status_code=429, detail={"error": "Too many requests. Please try again later."})

    user_obj = cast(
        Optional[LiteLLM_UserTable],
        await UserRepository(prisma_client).table.find_first(
            where={"user_email": {"equals": data.email, "mode": "insensitive"}}
        ),
    )

    if user_obj is None or user_obj.password is None:
        verbose_proxy_logger.warning("Password reset requested for an unknown or SSO-only email")
        return {"message": _GENERIC_FORGOT_PASSWORD_MESSAGE}

    configured_base_url = get_proxy_base_url()
    if configured_base_url is None:
        verbose_proxy_logger.warning(
            "Password reset email not sent: PROXY_BASE_URL is not configured, refusing to derive "
            "the reset link from the request Host header"
        )
        return {"message": _GENERIC_FORGOT_PASSWORD_MESSAGE}

    now = datetime.now(timezone.utc)
    token_repo = PasswordResetTokenRepository(prisma_client)
    await token_repo.invalidate_unused_for_user(user_id=user_obj.user_id, now=now)

    raw_token = secrets.token_urlsafe(32)
    await token_repo.create(
        data={
            "token_hash": hash_token(raw_token),
            "user_id": user_obj.user_id,
            "requested_ip": client_ip,
            "expires_at": now + timedelta(minutes=_RESET_TOKEN_TTL_MINUTES),
        }
    )

    reset_base_url = configured_base_url.rstrip("/") + "/ui/reset-password"
    reset_link = f"{reset_base_url}?token={raw_token}"

    asyncio.create_task(
        _send_reset_email_safely(
            receiver_email=data.email,
            subject="Reset your LiteLLM password",
            html=(
                f"<p>Click the link below to reset your password. This link expires in "
                f"{_RESET_TOKEN_TTL_MINUTES} minutes.</p><p><a href='{reset_link}'>{reset_link}</a></p>"
            ),
        )
    )

    return {"message": _GENERIC_FORGOT_PASSWORD_MESSAGE}


@router.get("/user/reset_password/validate", include_in_schema=False)
async def validate_reset_password_token(token: str):
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=500, detail={"error": CommonProxyErrors.db_not_connected_error.value})

    token_repo = PasswordResetTokenRepository(prisma_client)
    now = datetime.now(timezone.utc)
    token_row = await token_repo.find_valid_by_hash(token_hash=hash_token(token), now=now)

    if token_row is None:
        raise HTTPException(status_code=400, detail={"error": _GENERIC_INVALID_TOKEN_MESSAGE})

    user_obj = await UserRepository(prisma_client).table.find_unique(where={"user_id": token_row.user_id})
    if user_obj is None:
        raise HTTPException(status_code=400, detail={"error": _GENERIC_INVALID_TOKEN_MESSAGE})

    return {"user_email": user_obj.user_email}


@router.post("/user/reset_password", include_in_schema=False)
async def reset_password(data: ResetPasswordRequest):
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=500, detail={"error": CommonProxyErrors.db_not_connected_error.value})

    token_hash = hash_token(data.token)
    now = datetime.now(timezone.utc)

    token_row = await PasswordResetTokenRepository(prisma_client).find_valid_by_hash(token_hash=token_hash, now=now)
    if token_row is None:
        raise HTTPException(status_code=400, detail={"error": _GENERIC_INVALID_TOKEN_MESSAGE})

    hashed_pw = hash_password(data.new_password)

    async with prisma_client.db.tx() as tx:
        updated_count = await tx.litellm_passwordresettoken.update_many(
            where={"token_hash": token_hash, "used_at": None},
            data={"used_at": now},
        )
        if updated_count == 0:
            raise HTTPException(status_code=400, detail={"error": _GENERIC_INVALID_TOKEN_MESSAGE})

        user_obj = await tx.litellm_usertable.update(where={"user_id": token_row.user_id}, data={"password": hashed_pw})
        if user_obj is None:
            raise HTTPException(status_code=400, detail={"error": _GENERIC_INVALID_TOKEN_MESSAGE})

        await tx.litellm_passwordresettoken.update_many(
            where={"user_id": token_row.user_id, "used_at": None},
            data={"used_at": now},
        )

    return {"message": "Password reset successfully. Please log in with your new password."}
