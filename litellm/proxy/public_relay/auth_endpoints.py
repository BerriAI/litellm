from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status

from litellm.proxy.public_relay.api_types import (
    ActivateRequest,
    ApiKeyResponse,
    LoginRequest,
    MessageResponse,
    PasswordResetRequest,
    SessionResponse,
)
from litellm.proxy.public_relay.db_types import AccountRow
from litellm.proxy.public_relay.repository import (
    activate_account,
    get_account_by_email,
    reset_password_with_token,
)
from litellm.proxy.public_relay.runtime import (
    SESSION_COOKIE,
    PortalContext,
    account_response,
    database,
    relay_store,
    remote_ip,
    require_portal_write,
    settings,
)
from litellm.proxy.public_relay.security import (
    hash_auth_token,
    hash_password,
    new_session_credentials,
    normalize_email,
    verify_account_password,
)
from litellm.proxy.public_relay.session_store import PortalSession, RelayStore

router = APIRouter(prefix="/v1/public/auth", tags=["public relay authentication"])


@router.post("/activate", response_model=SessionResponse)
async def activate(payload: ActivateRequest, response: Response) -> SessionResponse:
    value = settings()
    try:
        created = await activate_account(
            database(),
            hash_auth_token(value.session_secret, payload.token),
            hash_password(payload.password),
        )
    except (PermissionError, ValueError) as exc:
        raise _authentication_failure(status.HTTP_400_BAD_REQUEST) from exc
    credentials = new_session_credentials()
    await relay_store(value).create_session(
        credentials.token,
        _portal_session(created.account, credentials.csrf_token),
    )
    _set_session_cookie(response, credentials.token, value.session_ttl_seconds)
    return SessionResponse(
        account=account_response(created.account),
        csrf_token=credentials.csrf_token,
        default_key=ApiKeyResponse(
            key_id=created.key_id,
            alias="Default",
            key=created.raw_key,
            log_content=False,
        ),
    )


@router.post("/login", response_model=SessionResponse)
async def login(payload: LoginRequest, request: Request, response: Response) -> SessionResponse:
    value = settings()
    try:
        email = normalize_email(payload.email)
    except ValueError as exc:
        raise _authentication_failure() from exc
    store = relay_store(value)
    await _enforce(store, f"login:ip:{remote_ip(request)}", 30, 900)
    await _enforce(store, f"login:email:{email}", 10, 900)
    account = await get_account_by_email(database(), email)
    password_valid = verify_account_password(payload.password, account.password if account is not None else None)
    if account is None or account.status != "ACTIVE" or not password_valid:
        raise _authentication_failure()
    credentials = new_session_credentials()
    await store.create_session(credentials.token, _portal_session(account, credentials.csrf_token))
    _set_session_cookie(response, credentials.token, value.session_ttl_seconds)
    return SessionResponse(account=account_response(account), csrf_token=credentials.csrf_token)


@router.post("/password-reset", response_model=MessageResponse)
async def password_reset(payload: PasswordResetRequest) -> MessageResponse:
    value = settings()
    try:
        await reset_password_with_token(
            database(),
            hash_auth_token(value.session_secret, payload.token),
            hash_password(payload.password),
        )
    except (PermissionError, ValueError) as exc:
        raise _authentication_failure(status.HTTP_400_BAD_REQUEST) from exc
    return MessageResponse(message="Password updated")


@router.post("/logout", response_model=MessageResponse)
async def logout(
    response: Response,
    context: Annotated[PortalContext, Depends(require_portal_write)],
) -> MessageResponse:
    value = settings()
    await relay_store(value).delete_session(context.token)
    response.delete_cookie(SESSION_COOKIE, path="/", secure=True, httponly=True, samesite="lax")
    return MessageResponse(message="Signed out")


def _portal_session(account: AccountRow, csrf_token: str) -> PortalSession:
    return PortalSession(
        account_id=account.account_id,
        user_id=account.user_id,
        email=account.normalized_email,
        session_version=account.session_version,
        csrf_token=csrf_token,
    )


async def _enforce(store: RelayStore, key: str, limit: int, window_seconds: int) -> None:
    try:
        await store.enforce_limit(key, limit, window_seconds)
    except PermissionError as exc:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={"error": {"message": "Too many requests", "type": "rate_limit_error"}},
        ) from exc


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
