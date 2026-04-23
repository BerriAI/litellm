"""
Admin-only endpoints that let the LiteLLM UI sign in to GitHub Copilot using
GitHub's device-code OAuth flow, and persist the resulting ``access_token``
to ``LiteLLM_CredentialsTable``.

Mirrors ``litellm/proxy/chatgpt_oauth_endpoints/`` — the only user-facing
differences are Copilot-specific verification URLs and the ``refresh``
endpoint semantics (refreshes the short-lived Copilot API key rather than
the long-lived GitHub OAuth token, since the latter does not expire).

Endpoints:
  POST /copilot/oauth/start      — start a new flow; returns user_code + verification URL
  GET  /copilot/oauth/status     — poll session status (pending/success/error)
  POST /copilot/oauth/cancel     — abandon an in-flight flow
  POST /copilot/oauth/refresh    — force a Copilot API key refresh for a stored credential
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
from litellm.llms.github_copilot.authenticator import Authenticator
from litellm.llms.github_copilot.common_utils import GithubCopilotError
from litellm.llms.github_copilot.db_authenticator import (
    CREDENTIAL_TYPE,
    DBAuthenticator,
    _register_proxy_main_loop,
    persist_credential_to_db,
)
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.types.utils import CredentialItem

router = APIRouter(prefix="/copilot/oauth", tags=["copilot oauth"])

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
    api_key_expires_at: Optional[int] = None


def _require_store_model_in_db() -> None:
    """
    Fail fast if ``STORE_MODEL_IN_DB`` is not set — see the chatgpt
    endpoints for the full rationale. Without this env var the OAuth
    credential write succeeds but the proxy won't reload it on restart,
    causing silent data loss after the user sits through the device-code
    poll.
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
            detail="Only PROXY_ADMIN may initiate Copilot OAuth flows.",
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
    # path can schedule persist work cross-thread via run_coroutine_threadsafe.
    loop = asyncio.get_running_loop()
    _register_proxy_main_loop(loop)

    authenticator = Authenticator()
    try:
        device_code_info = await loop.run_in_executor(
            None, authenticator._get_device_code
        )
    except GithubCopilotError as exc:
        with _sessions_lock:
            _sessions.pop(session_id, None)
        raise HTTPException(status_code=502, detail=exc.message)

    with _sessions_lock:
        entry = _sessions.get(session_id)
        if entry is None:
            raise HTTPException(status_code=410, detail="Session was cancelled")
        entry["status"] = "pending"
        entry["device_code_info"] = device_code_info

    # Run the remaining flow as a task on the main loop so the DB persist
    # step shares an event loop with prisma_client.
    asyncio.create_task(
        _run_device_code_flow_async(
            session_id, body.credential_name, device_code_info, authenticator
        )
    )

    return StartResponse(
        session_id=session_id,
        user_code=device_code_info["user_code"],
        verification_url=device_code_info["verification_uri"],
        interval=int(device_code_info.get("interval", 5)),
    )


@router.get(
    "/status",
    response_model=StatusResponse,
)
async def oauth_status(
    session_id: str = Query(
        ..., description="Session ID returned by /copilot/oauth/start"
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
        ..., description="Session ID returned by /copilot/oauth/start"
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
    Force a refresh of the Copilot API key derived from the stored GitHub
    access token. The GitHub access token itself does not have a short TTL;
    this endpoint is primarily a health-check / "force new session key"
    action surfaced in the UI.
    """
    _require_admin(user_api_key_dict)
    db_auth = DBAuthenticator(credential_name=body.credential_name)
    try:
        info = db_auth.force_refresh_api_key()
    except GithubCopilotError as exc:
        raise HTTPException(status_code=502, detail=exc.message)
    expires_at = info.get("expires_at")
    return RefreshResponse(
        credential_name=body.credential_name,
        api_key_expires_at=int(expires_at) if expires_at is not None else None,
    )


async def _run_device_code_flow_async(
    session_id: str,
    credential_name: str,
    device_code_info: Dict[str, str],
    authenticator: Authenticator,
) -> None:
    """
    Background async task: polls GitHub for the access token, then upserts
    the credential into the in-memory cache and awaits the DB persist on
    the same event loop as ``prisma_client``.
    """
    loop = asyncio.get_running_loop()
    try:
        access_token = await loop.run_in_executor(
            None,
            authenticator._poll_for_access_token,
            device_code_info["device_code"],
        )
        session_snapshot = _get_session(session_id)
        if session_snapshot is None or session_snapshot.get("cancelled"):
            return
    except GithubCopilotError as exc:
        _update_session(session_id, status="error", message=exc.message)
        return
    except Exception as exc:  # pragma: no cover - defensive
        verbose_proxy_logger.exception("Unexpected error in Copilot OAuth flow")
        _update_session(session_id, status="error", message=str(exc))
        return

    item = CredentialItem(
        credential_name=credential_name,
        credential_values={"access_token": access_token},
        credential_info={
            "type": CREDENTIAL_TYPE,
            "custom_llm_provider": "github_copilot",
        },
    )
    CredentialAccessor.upsert_credentials([item])
    # Invalidate any cached Copilot API key tied to an old access token.
    DBAuthenticator._api_key_cache.pop(credential_name, None)

    try:
        await persist_credential_to_db(item)
    except Exception as exc:
        verbose_proxy_logger.error(
            "Failed to persist Copilot OAuth credential %s: %s",
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
