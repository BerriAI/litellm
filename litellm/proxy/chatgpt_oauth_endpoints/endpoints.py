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

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.credential_accessor import CredentialAccessor
from litellm.secret_managers.main import get_secret_bool
from litellm.llms.chatgpt.authenticator import Authenticator
from litellm.llms.chatgpt.common_utils import (
    CHATGPT_DEVICE_VERIFY_URL,
    ChatGPTAuthError,
)
from litellm.llms.chatgpt.db_authenticator import (
    CREDENTIAL_TYPE,
    DBAuthenticator,
    _register_proxy_main_loop,
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


def _require_store_model_in_db() -> None:
    """
    Fail fast if ``STORE_MODEL_IN_DB`` is not set.

    OAuth tokens get encrypted + upserted into ``LiteLLM_CredentialsTable``
    by this flow, but the proxy only reloads that table into
    ``litellm.credential_list`` on startup when ``STORE_MODEL_IN_DB=True``
    (see ``proxy_config.get_credentials`` wiring in ``proxy_server.py``).
    Without the env var the write succeeds but nothing reads it back — the
    credential appears to vanish on the next restart, and request-time
    ``api_key: oauth:<name>`` resolution fails because the name is no
    longer in the in-memory cache. Refuse up front so admins don't sit
    through the 15-minute device-code poll only to hit silent data loss.
    """
    if not get_secret_bool("STORE_MODEL_IN_DB", False):
        raise HTTPException(
            status_code=400,
            detail=(
                "OAuth sign-in requires STORE_MODEL_IN_DB=True so the "
                "proxy can reload credentials from the database on "
                "restart. Set the env var (or equivalent in your "
                "deployment config) and restart before trying again."
            ),
        )


def _require_admin(user_api_key_dict: UserAPIKeyAuth) -> None:
    # These endpoints write to LiteLLM_CredentialsTable (start → insert,
    # refresh → rotate). PROXY_ADMIN_VIEW_ONLY must not reach them.
    role = getattr(user_api_key_dict, "user_role", None)
    if role != LitellmUserRoles.PROXY_ADMIN:
        raise HTTPException(
            status_code=403,
            detail="Only PROXY_ADMIN may initiate ChatGPT OAuth flows.",
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
)
async def start_oauth(
    body: StartRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> StartResponse:
    _require_admin(user_api_key_dict)
    _require_store_model_in_db()
    _purge_expired_sessions()

    # Atomically reserve a slot so concurrent callers cannot all pass the
    # capacity check and blow past SESSIONS_MAX_SIZE.
    session_id = uuid.uuid4().hex
    now = time.time()
    with _sessions_lock:
        if len(_sessions) >= SESSIONS_MAX_SIZE:
            raise HTTPException(
                status_code=429,
                detail="Too many in-flight OAuth sessions. Retry shortly.",
            )
        _sessions[session_id] = {
            "status": "starting",
            "credential_name": body.credential_name,
            "started_by": user_api_key_dict.user_id,
            "started_at": now,
            "expires_at": now + SESSION_TTL_SECONDS,
            "cancelled": False,
        }

    # Capture the proxy's main event loop so the DBAuthenticator refresh
    # path can schedule its fire-and-forget persist safely across threads
    # (prevents the cross-event-loop errors you hit if you `asyncio.run`
    # an async prisma call from a worker thread).
    loop = asyncio.get_running_loop()
    _register_proxy_main_loop(loop)

    authenticator = Authenticator()
    try:
        device_code = await loop.run_in_executor(
            None, authenticator._request_device_code
        )
    except ChatGPTAuthError as exc:
        with _sessions_lock:
            _sessions.pop(session_id, None)
        raise HTTPException(status_code=502, detail=exc.message)

    with _sessions_lock:
        entry = _sessions.get(session_id)
        if entry is None:
            # Cancelled or evicted while the device-code call was in flight.
            raise HTTPException(status_code=410, detail="Session was cancelled")
        entry["status"] = "pending"
        entry["device_code"] = device_code

    # Run the remaining poll → exchange → persist flow as a task on the main
    # loop (was previously a thread doing ``asyncio.run``, which built a new
    # loop and collided with prisma_client's main-loop-bound primitives).
    # Blocking IO inside the task runs through ``loop.run_in_executor``.
    asyncio.create_task(
        _run_device_code_flow_async(
            session_id, body.credential_name, device_code, authenticator
        )
    )

    return StartResponse(
        session_id=session_id,
        user_code=device_code["user_code"],
        verification_url=CHATGPT_DEVICE_VERIFY_URL,
        interval=int(device_code.get("interval", "5")),
    )


@router.get(
    "/status",
    response_model=StatusResponse,
)
async def oauth_status(
    session_id: str = Query(
        ..., description="Session ID returned by /chatgpt/oauth/start"
    ),
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
)
async def oauth_cancel(
    session_id: str = Query(
        ..., description="Session ID returned by /chatgpt/oauth/start"
    ),
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


async def _run_device_code_flow_async(
    session_id: str,
    credential_name: str,
    device_code: Dict[str, str],
    authenticator: Authenticator,
) -> None:
    """
    Background async task: polls for the authorization code, exchanges it
    for tokens, then upserts the credential into the in-memory cache and
    DB. Runs on the proxy's main event loop so the DB persist step shares
    a loop with ``prisma_client``.
    """
    loop = asyncio.get_running_loop()
    try:
        auth_code = await loop.run_in_executor(
            None, authenticator._poll_for_authorization_code, device_code
        )
        session_snapshot = _get_session(session_id)
        if session_snapshot is None or session_snapshot.get("cancelled"):
            return
        tokens = await loop.run_in_executor(
            None, authenticator._exchange_code_for_tokens, auth_code
        )
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
        await persist_credential_to_db(item)
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
