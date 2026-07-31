"""
Self-service password reset endpoints for internal (non-SSO) users.

/user/forgot_password
/user/reset_password/validate
/user/reset_password
"""

import asyncio
import secrets
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, HTTPException, Request

from litellm._logging import verbose_proxy_logger
from litellm.constants import UI_SESSION_TOKEN_TEAM_ID
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
    ValidateResetPasswordTokenRequest,
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


async def _revoke_ui_sessions_for_user(tx, user_id: str) -> list:
    """Delete the user's dashboard UI-session keys inside the reset transaction.

    Stops an already-issued session cookie from continuing to authenticate after
    a password reset. Scoped to `team_id == UI_SESSION_TOKEN_TEAM_ID` so the
    user's own personal API keys (used for LLM calls, not the dashboard) are left
    untouched. Returns the deleted rows so callers can evict them from the auth
    cache outside the transaction.
    """
    ui_session_tokens = await tx.litellm_verificationtoken.find_many(
        where={"user_id": user_id, "team_id": UI_SESSION_TOKEN_TEAM_ID}
    )
    if ui_session_tokens:
        await tx.litellm_verificationtoken.delete_many(where={"user_id": user_id, "team_id": UI_SESSION_TOKEN_TEAM_ID})
    return ui_session_tokens


@router.post("/user/forgot_password", include_in_schema=False)
async def forgot_password(data: ForgotPasswordRequest, request: Request):
    from litellm.proxy.proxy_server import prisma_client, spend_counter_cache

    if prisma_client is None:
        raise HTTPException(status_code=500, detail={"error": CommonProxyErrors.db_not_connected_error.value})

    cidrs = get_trusted_proxy_cidrs()
    resolved_ip, _ = resolve_client_ip(
        request, TrustedProxyConfig(use_forwarded_for=bool(cidrs), trusted_proxy_cidrs=cidrs)
    )
    client_ip = resolved_ip or "unknown"
    # spend_counter_cache (unlike user_api_key_cache) is always Redis-backed when Redis is
    # configured, regardless of the enable_redis_auth_cache flag, so these counters stay
    # accurate across multi-worker/multi-pod deployments instead of being process-local.
    email_count = await spend_counter_cache.async_increment_cache(
        key=f"password_reset_rl:email:{data.email.lower()}", value=1, ttl=_RATE_LIMIT_WINDOW_SECONDS
    )
    ip_count = await spend_counter_cache.async_increment_cache(
        key=f"password_reset_rl:ip:{client_ip}", value=1, ttl=_RATE_LIMIT_WINDOW_SECONDS
    )

    if (email_count is not None and email_count > _MAX_REQUESTS_PER_EMAIL_PER_HOUR) or (
        ip_count is not None and ip_count > _MAX_REQUESTS_PER_IP_PER_HOUR
    ):
        verbose_proxy_logger.warning("Password reset rate limit exceeded for ip=%s", client_ip)
        raise HTTPException(status_code=429, detail={"error": "Too many requests. Please try again later."})

    user_obj = await UserRepository(prisma_client).table.find_first(
        where={"user_email": {"equals": data.email, "mode": "insensitive"}}
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
    # The token lives in the URL fragment, not the query string: fragments are never sent to
    # the server, so they don't land in access logs, Referer headers, or (server-side) browser
    # history entries the way a `?token=` query param would.
    reset_link = f"{reset_base_url}#token={raw_token}"

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


@router.post("/user/reset_password/validate", include_in_schema=False)
async def validate_reset_password_token(data: ValidateResetPasswordTokenRequest):
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=500, detail={"error": CommonProxyErrors.db_not_connected_error.value})

    token_repo = PasswordResetTokenRepository(prisma_client)
    now = datetime.now(timezone.utc)
    token_row = await token_repo.find_valid_by_hash(token_hash=hash_token(data.token), now=now)

    if token_row is None:
        raise HTTPException(status_code=400, detail={"error": _GENERIC_INVALID_TOKEN_MESSAGE})

    user_obj = await UserRepository(prisma_client).table.find_unique(where={"user_id": token_row.user_id})
    if user_obj is None:
        raise HTTPException(status_code=400, detail={"error": _GENERIC_INVALID_TOKEN_MESSAGE})

    return {"user_email": user_obj.user_email}


@router.post("/user/reset_password", include_in_schema=False)
async def reset_password(data: ResetPasswordRequest):
    from litellm.proxy.proxy_server import prisma_client, proxy_logging_obj, user_api_key_cache

    if prisma_client is None:
        raise HTTPException(status_code=500, detail={"error": CommonProxyErrors.db_not_connected_error.value})

    token_hash = hash_token(data.token)
    now = datetime.now(timezone.utc)

    token_row = await PasswordResetTokenRepository(prisma_client).find_valid_by_hash(token_hash=token_hash, now=now)
    if token_row is None:
        raise HTTPException(status_code=400, detail={"error": _GENERIC_INVALID_TOKEN_MESSAGE})

    hashed_pw = hash_password(data.new_password)

    async with prisma_client.db.tx() as tx:
        # Re-check expiry inside the atomic claim: without it, a token that expires between
        # find_valid_by_hash above and this update still gets claimed, since the predicate
        # would otherwise only check token_hash/used_at.
        updated_count = await tx.litellm_passwordresettoken.update_many(
            where={"token_hash": token_hash, "used_at": None, "expires_at": {"gt": now}},
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

        revoked_ui_sessions = await _revoke_ui_sessions_for_user(tx, token_row.user_id)

    for session_token in revoked_ui_sessions:
        user_api_key_cache.delete_cache(key=session_token.token)
        await proxy_logging_obj.internal_usage_cache.dual_cache.async_delete_cache(key=session_token.token)

    return {"message": "Password reset successfully. Please log in with your new password."}
