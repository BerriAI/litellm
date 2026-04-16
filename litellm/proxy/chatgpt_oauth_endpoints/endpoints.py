"""
Admin-only endpoints that let the LiteLLM UI sign in to the ChatGPT / Codex
backend using the OpenAI device-code OAuth flow and persist the resulting
tokens in ``LiteLLM_CredentialsTable``.

The browser-based PKCE flow redirects to ``127.0.0.1:1455`` on the user's
machine, which does not work for a remotely-hosted UI — device code is the
correct flow here: the user's browser visits ``auth.openai.com/codex/device``,
enters a code, and the proxy polls the token endpoint in the background.

Endpoints:
  POST /chatgpt/oauth/start       — start a new flow; returns user_code + verification URL
  GET  /chatgpt/oauth/status      — poll session status (pending/success/error)
  POST /chatgpt/oauth/cancel      — abandon an in-flight flow

Session state is held in-memory per proxy replica. For a multi-replica
deploy, enable sticky sessions for the UI → proxy path or use a
single-replica admin node.
"""

import asyncio
import threading
import time
import uuid
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.credential_accessor import CredentialAccessor
from litellm.llms.chatgpt.authenticator import Authenticator
from litellm.llms.chatgpt.common_utils import (
    CHATGPT_DEVICE_VERIFY_URL,
    ChatGPTAuthError,
)
from litellm.llms.chatgpt.db_authenticator import (
    CREDENTIAL_TYPE,
    DBAuthenticator,
    persist_credential_to_db,
)
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.types.utils import CredentialItem

router = APIRouter(prefix="/chatgpt/oauth", tags=["chatgpt oauth"])

SESSION_TTL_SECONDS = 20 * 60
SESSIONS_MAX_SIZE = 100

_sessions: Dict[str, Dict[str, Any]] = {}
_sessions_lock = threading.Lock()


class StartRequest(BaseModel):
    credential_name: str


class StartResponse(BaseModel):
    session_id: str
    user_code: str
    verification_url: str
    interval: int


class StatusResponse(BaseModel):
    status: str  # "pending" | "success" | "error" | "cancelled"
    credential_name: Optional[str] = None
    message: Optional[str] = None


class RefreshRequest(BaseModel):
    credential_name: str


class RefreshResponse(BaseModel):
    credential_name: str
    expires_at: Optional[int] = None


def _require_admin(user_api_key_dict: UserAPIKeyAuth) -> None:
    role = getattr(user_api_key_dict, "user_role", None)
    if role not in (
        LitellmUserRoles.PROXY_ADMIN,
        LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY,
    ):
        raise HTTPException(
            status_code=403,
            detail="Only proxy admins may initiate ChatGPT OAuth flows.",
        )


def _purge_expired_sessions() -> None:
    now = time.time()
    with _sessions_lock:
        expired = [k for k, v in _sessions.items() if v.get("expires_at", 0) < now]
        for k in expired:
            del _sessions[k]


def _update_session(session_id: str, **fields: Any) -> None:
    with _sessions_lock:
        entry = _sessions.get(session_id)
        if entry is None:
            return
        entry.update(fields)


def _get_session(session_id: str) -> Optional[Dict[str, Any]]:
    with _sessions_lock:
        entry = _sessions.get(session_id)
        return dict(entry) if entry else None


@router.post(
    "/start",
    response_model=StartResponse,
    dependencies=[Depends(user_api_key_auth)],
)
async def start_oauth(
    body: StartRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> StartResponse:
    _require_admin(user_api_key_dict)
    _purge_expired_sessions()

    with _sessions_lock:
        if len(_sessions) >= SESSIONS_MAX_SIZE:
            raise HTTPException(
                status_code=429,
                detail="Too many in-flight OAuth sessions. Retry shortly.",
            )

    authenticator = Authenticator()
    try:
        device_code = authenticator._request_device_code()
    except ChatGPTAuthError as exc:
        raise HTTPException(status_code=502, detail=exc.message)

    session_id = uuid.uuid4().hex
    session = {
        "status": "pending",
        "credential_name": body.credential_name,
        "started_by": user_api_key_dict.user_id,
        "started_at": time.time(),
        "expires_at": time.time() + SESSION_TTL_SECONDS,
        "device_code": device_code,
        "cancelled": False,
    }
    with _sessions_lock:
        _sessions[session_id] = session

    thread = threading.Thread(
        target=_run_device_code_flow,
        args=(session_id, body.credential_name, device_code, authenticator),
        daemon=True,
        name=f"chatgpt-oauth-{session_id[:8]}",
    )
    thread.start()

    return StartResponse(
        session_id=session_id,
        user_code=device_code["user_code"],
        verification_url=CHATGPT_DEVICE_VERIFY_URL,
        interval=int(device_code.get("interval", "5")),
    )


@router.get(
    "/status",
    response_model=StatusResponse,
    dependencies=[Depends(user_api_key_auth)],
)
async def oauth_status(
    session_id: str,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> StatusResponse:
    _require_admin(user_api_key_dict)
    entry = _get_session(session_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="Unknown or expired session_id")
    return StatusResponse(
        status=entry["status"],
        credential_name=entry.get("credential_name"),
        message=entry.get("message"),
    )


@router.post(
    "/cancel",
    dependencies=[Depends(user_api_key_auth)],
)
async def oauth_cancel(
    session_id: str,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> Dict[str, bool]:
    _require_admin(user_api_key_dict)
    with _sessions_lock:
        entry = _sessions.get(session_id)
        if entry is None:
            raise HTTPException(status_code=404, detail="Unknown or expired session_id")
        entry["cancelled"] = True
        if entry["status"] == "pending":
            entry["status"] = "cancelled"
            entry["message"] = "Cancelled by user"
    return {"success": True}


@router.post(
    "/refresh",
    response_model=RefreshResponse,
    dependencies=[Depends(user_api_key_auth)],
)
async def oauth_refresh(
    body: RefreshRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> RefreshResponse:
    """
    Force a refresh of the OAuth access token using the stored refresh
    token. Rotates the refresh token too if the IdP issues a new one.
    """
    _require_admin(user_api_key_dict)
    db_auth = DBAuthenticator(credential_name=body.credential_name)
    auth_data = db_auth._read_auth_file()
    if not auth_data or not auth_data.get("refresh_token"):
        raise HTTPException(
            status_code=404,
            detail=(
                f"No refresh_token stored for credential '{body.credential_name}'. "
                "Re-run the sign-in flow."
            ),
        )
    try:
        db_auth._refresh_tokens(auth_data["refresh_token"])
    except ChatGPTAuthError as exc:
        raise HTTPException(status_code=502, detail=exc.message)
    # _refresh_tokens already wrote back via DBAuthenticator._write_auth_file,
    # which scheduled the DB persist. Re-read to get the canonical expires_at.
    fresh = db_auth._read_auth_file() or {}
    return RefreshResponse(
        credential_name=body.credential_name,
        expires_at=fresh.get("expires_at"),
    )


def _run_device_code_flow(
    session_id: str,
    credential_name: str,
    device_code: Dict[str, str],
    authenticator: Authenticator,
) -> None:
    """
    Background worker: polls for the authorization code, exchanges it for
    tokens, then upserts the credential into the in-memory cache and DB.
    """
    try:
        auth_code = authenticator._poll_for_authorization_code(device_code)
        if _get_session(session_id).get("cancelled"):  # type: ignore[union-attr]
            return
        tokens = authenticator._exchange_code_for_tokens(auth_code)
    except ChatGPTAuthError as exc:
        _update_session(session_id, status="error", message=exc.message)
        return
    except Exception as exc:  # pragma: no cover - defensive
        verbose_proxy_logger.exception("Unexpected error in ChatGPT OAuth flow")
        _update_session(session_id, status="error", message=str(exc))
        return

    auth_record = authenticator._build_auth_record(tokens)
    credential_values = {k: str(v) for k, v in auth_record.items() if v is not None}
    item = CredentialItem(
        credential_name=credential_name,
        credential_values=credential_values,
        credential_info={
            "type": CREDENTIAL_TYPE,
            "custom_llm_provider": "chatgpt",
        },
    )
    CredentialAccessor.upsert_credentials([item])

    try:
        asyncio.run(persist_credential_to_db(item))
    except Exception as exc:
        verbose_proxy_logger.error(
            "Failed to persist ChatGPT OAuth credential %s: %s",
            credential_name,
            exc,
        )
        _update_session(
            session_id,
            status="error",
            message=f"Tokens obtained but DB persist failed: {exc}",
        )
        return

    _update_session(session_id, status="success", message=None)
