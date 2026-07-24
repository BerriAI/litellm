from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, Response, status

from litellm._logging import verbose_proxy_logger
from litellm.proxy.public_relay.api_types import (
    ApiKeyResponse,
    LoginRequest,
    MessageResponse,
    PasswordResetRequest,
    RegisterRequest,
    SessionResponse,
    VerificationCodeRequest,
)
from litellm.proxy.public_relay.email_delivery import VerificationEmail, send_verification_email
from litellm.proxy.public_relay.repository import (
    create_account,
    get_account_by_email,
    update_password,
)
from litellm.proxy.public_relay.runtime import (
    SESSION_COOKIE,
    PortalContext,
    account_response,
    database,
    relay_cache,
    remote_ip,
    require_portal_write,
    settings,
)
from litellm.proxy.public_relay.security import (
    hash_password,
    hash_verification_code,
    new_session_credentials,
    new_verification_code,
    normalize_email,
    verification_code_matches,
    verify_account_password,
)
from litellm.proxy.public_relay.session_store import PortalSession, RelayCache, VerificationRecord
from litellm.proxy.public_relay.turnstile import TurnstileVerifier

router = APIRouter(prefix="/v1/public/auth", tags=["public relay authentication"])


@router.post("/register/code", response_model=MessageResponse)
async def register_code(
    payload: VerificationCodeRequest,
    request: Request,
    background_tasks: BackgroundTasks,
) -> MessageResponse:
    await _send_code(payload, request, background_tasks, "register")
    return MessageResponse(message="If the address is eligible, a verification code has been sent")


@router.post("/password-reset/code", response_model=MessageResponse)
async def password_reset_code(
    payload: VerificationCodeRequest,
    request: Request,
    background_tasks: BackgroundTasks,
) -> MessageResponse:
    await _send_code(payload, request, background_tasks, "password-reset")
    return MessageResponse(message="If the account exists, a verification code has been sent")


@router.post("/register", response_model=SessionResponse)
async def register(payload: RegisterRequest, request: Request, response: Response) -> SessionResponse:
    value = settings()
    email = _normalized_email(payload.email)
    await _verify_turnstile(value.turnstile_verify_url, payload.turnstile_token, remote_ip(request))
    cache = relay_cache(value)
    await _enforce(cache, f"public-relay:register:ip:{remote_ip(request)}", 10, 3600)
    await _validate_code(cache, value.session_secret, "register", email, payload.code)
    if await get_account_by_email(database(), email) is not None:
        raise _authentication_failure(status.HTTP_400_BAD_REQUEST)
    try:
        created = await create_account(database(), email, hash_password(payload.password))
    except Exception as exc:
        raise _authentication_failure(status.HTTP_400_BAD_REQUEST) from exc
    await cache.delete_verification("register", email)
    credentials = new_session_credentials()
    await cache.create_session(
        credentials.token,
        PortalSession(
            account_id=created.account.account_id,
            user_id=created.account.user_id,
            email=email,
            session_version=created.account.session_version,
            csrf_token=credentials.csrf_token,
        ),
    )
    _set_session_cookie(response, credentials.token, value.session_ttl_seconds)
    return SessionResponse(
        account=account_response(created.account),
        csrf_token=credentials.csrf_token,
        default_key=ApiKeyResponse(
            key_id=created.key_id,
            alias="Default",
            key=created.raw_key,
            log_content=True,
        ),
    )


@router.post("/login", response_model=SessionResponse)
async def login(payload: LoginRequest, request: Request, response: Response) -> SessionResponse:
    value = settings()
    email = _normalized_email(payload.email)
    await _verify_turnstile(value.turnstile_verify_url, payload.turnstile_token, remote_ip(request))
    cache = relay_cache(value)
    await _enforce(cache, f"public-relay:login:ip:{remote_ip(request)}", 30, 900)
    await _enforce(cache, f"public-relay:login:email:{email}", 10, 900)
    account = await get_account_by_email(database(), email)
    password_valid = verify_account_password(payload.password, account.password if account is not None else None)
    if account is None or account.status != "ACTIVE" or not password_valid:
        raise _authentication_failure()
    credentials = new_session_credentials()
    await cache.create_session(
        credentials.token,
        PortalSession(
            account_id=account.account_id,
            user_id=account.user_id,
            email=account.normalized_email,
            session_version=account.session_version,
            csrf_token=credentials.csrf_token,
        ),
    )
    _set_session_cookie(response, credentials.token, value.session_ttl_seconds)
    return SessionResponse(account=account_response(account), csrf_token=credentials.csrf_token)


@router.post("/password-reset", response_model=MessageResponse)
async def password_reset(payload: PasswordResetRequest, request: Request) -> MessageResponse:
    value = settings()
    email = _normalized_email(payload.email)
    await _verify_turnstile(value.turnstile_verify_url, payload.turnstile_token, remote_ip(request))
    cache = relay_cache(value)
    await _enforce(cache, f"public-relay:reset:ip:{remote_ip(request)}", 10, 3600)
    await _validate_code(cache, value.session_secret, "password-reset", email, payload.code)
    account = await get_account_by_email(database(), email)
    if account is None:
        raise _authentication_failure(status.HTTP_400_BAD_REQUEST)
    try:
        password_hash = hash_password(payload.password)
    except ValueError as exc:
        raise _authentication_failure(status.HTTP_400_BAD_REQUEST) from exc
    await update_password(database(), account, password_hash)
    await cache.delete_verification("password-reset", email)
    return MessageResponse(message="Password updated")


@router.post("/logout", response_model=MessageResponse)
async def logout(
    response: Response,
    context: Annotated[PortalContext, Depends(require_portal_write)],
) -> MessageResponse:
    value = settings()
    await relay_cache(value).delete_session(context.token)
    response.delete_cookie(SESSION_COOKIE, path="/", secure=True, httponly=True, samesite="lax")
    return MessageResponse(message="Signed out")


async def _send_code(
    payload: VerificationCodeRequest,
    request: Request,
    background_tasks: BackgroundTasks,
    purpose: str,
) -> None:
    value = settings()
    email = _normalized_email(payload.email)
    ip_address = remote_ip(request)
    await _verify_turnstile(value.turnstile_verify_url, payload.turnstile_token, ip_address)
    cache = relay_cache(value)
    await _enforce(cache, f"public-relay:code:ip:{ip_address}", 20, 3600)
    await _enforce(
        cache,
        f"public-relay:code:email:{purpose}:{email}",
        1,
        value.verification_resend_seconds,
    )
    account = await get_account_by_email(database(), email)
    eligible = (purpose == "register" and account is None) or (purpose == "password-reset" and account is not None)
    if not eligible:
        return
    code = new_verification_code()
    await cache.put_verification(
        VerificationRecord(
            code_hash=hash_verification_code(value.session_secret, purpose, email, code),
            purpose=purpose,
            email=email,
        )
    )
    background_tasks.add_task(_deliver_code, VerificationEmail(receiver=email, code=code, purpose=purpose))


async def _deliver_code(message: VerificationEmail) -> None:
    try:
        await send_verification_email(message)
    except Exception:  # noqa: BLE001  # Background delivery failures must not expose account eligibility.
        verbose_proxy_logger.exception("Failed to deliver public relay verification email")


async def _validate_code(cache: RelayCache, secret: bytes, purpose: str, email: str, code: str) -> None:
    await _enforce(cache, f"public-relay:verify:{purpose}:{email}", cache.settings.verification_max_attempts, 600)
    record = await cache.get_verification(purpose, email)
    if record is None or not verification_code_matches(secret, purpose, email, code, record.code_hash):
        raise _authentication_failure(status.HTTP_400_BAD_REQUEST)


async def _verify_turnstile(verify_url: str | None, token: str, ip_address: str) -> None:
    if verify_url is None or not await TurnstileVerifier(verify_url).verify(token, ip_address):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": {"message": "Human verification failed", "type": "invalid_request_error"}},
        )


async def _enforce(cache: RelayCache, key: str, limit: int, ttl_seconds: int) -> None:
    try:
        await cache.enforce_limit(key, limit, ttl_seconds)
    except PermissionError as exc:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={"error": {"message": "Too many requests", "type": "rate_limit_error"}},
        ) from exc


def _normalized_email(value: str) -> str:
    try:
        return normalize_email(value)
    except ValueError as exc:
        raise _authentication_failure(status.HTTP_400_BAD_REQUEST) from exc


def _authentication_failure(status_code: int = status.HTTP_401_UNAUTHORIZED) -> HTTPException:
    return HTTPException(
        status_code=status_code,
        detail={"error": {"message": "Unable to authenticate", "type": "authentication_error"}},
    )


def _set_session_cookie(response: Response, token: str, max_age: int) -> None:
    response.set_cookie(
        SESSION_COOKIE,
        token,
        max_age=max_age,
        path="/",
        secure=True,
        httponly=True,
        samesite="lax",
    )
