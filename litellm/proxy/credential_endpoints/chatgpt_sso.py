"""
ChatGPT SSO endpoints for device code OAuth flow.

The backend is fully stateless during the auth flow. The client holds the
device_auth_id and user_code between requests — nothing is stored in the DB
until the tokens are successfully obtained.

Flow:
  1. POST /credentials/chatgpt/initiate
       Returns device_auth_id + user_code + verification_uri + poll_interval_ms.
       No DB write.

  2. POST /credentials/chatgpt/status  (client calls repeatedly)
       Body: { device_auth_id, user_code }
       Polls OpenAI's device token endpoint. When authorization_code is received,
       exchanges it for tokens automatically.
       Returns: { status: "pending"|"complete"|"failed", refresh_token?, account_id?, error? }
       On complete, returns the refresh_token — client stores it as a named credential.
       No DB write — client decides what to do with the token.

Design: no session_id, no pending/failed DB rows, no cancel endpoint.
Multi-worker safe: device codes are validated by OpenAI, not by LiteLLM.
"""

from typing import Literal, Optional
from urllib.parse import quote

import httpx as _httpx
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel

from litellm._logging import verbose_proxy_logger
from litellm.llms.custom_httpx.http_handler import get_async_httpx_client
from litellm.types.llms.custom_http import httpxSpecialProvider
from litellm.llms.chatgpt.authenticator import Authenticator
from litellm.llms.chatgpt.common_utils import (
    CHATGPT_AUTH_BASE,
    CHATGPT_CLIENT_ID,
    CHATGPT_DEVICE_CODE_URL,
    CHATGPT_DEVICE_TOKEN_URL,
    CHATGPT_DEVICE_VERIFY_URL,
    CHATGPT_OAUTH_TOKEN_URL,
)
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

router = APIRouter()


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class InitiateResponse(BaseModel):
    device_auth_id: str
    user_code: str
    verification_uri: str
    poll_interval_ms: int


class StatusRequest(BaseModel):
    device_auth_id: str
    user_code: str


class StatusResponse(BaseModel):
    status: Literal["pending", "complete", "failed"]
    refresh_token: Optional[str] = None  # complete only
    account_id: Optional[str] = None  # complete only
    error: Optional[str] = None  # failed only


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/credentials/chatgpt/initiate",
    dependencies=[Depends(user_api_key_auth)],
    tags=["credential management"],
    response_model=InitiateResponse,
)
async def chatgpt_initiate(
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Start the ChatGPT Device Code flow.

    Calls OpenAI's device code endpoint and returns device_auth_id, user_code,
    verification_uri, and poll_interval_ms to the client. Nothing is stored
    in the database.
    """
    async_client = get_async_httpx_client(llm_provider=httpxSpecialProvider.SSO_HANDLER)
    try:
        resp = await async_client.post(
            CHATGPT_DEVICE_CODE_URL,
            json={"client_id": CHATGPT_CLIENT_ID},
        )
        resp.raise_for_status()
    except Exception as e:
        verbose_proxy_logger.error(f"ChatGPT device code request failed: {e}")
        raise HTTPException(
            status_code=502,
            detail={"error": f"ChatGPT device code request failed: {e}"},
        )
    resp_json = resp.json()

    device_auth_id = resp_json.get("device_auth_id")
    user_code = resp_json.get("user_code") or resp_json.get("usercode")
    poll_interval_seconds = int(resp_json.get("interval", 5))

    if not all([device_auth_id, user_code]):
        raise HTTPException(
            status_code=502,
            detail={"error": "OpenAI response missing required fields"},
        )

    return InitiateResponse(
        device_auth_id=device_auth_id,
        user_code=user_code,
        verification_uri=CHATGPT_DEVICE_VERIFY_URL,
        poll_interval_ms=poll_interval_seconds * 1000,
    )


@router.post(
    "/credentials/chatgpt/status",
    dependencies=[Depends(user_api_key_auth)],
    tags=["credential management"],
    response_model=StatusResponse,
)
async def chatgpt_status(
    request: Request,
    fastapi_response: Response,
    body: StatusRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Poll the ChatGPT Device Code flow for completion.

    Makes a single attempt to check if the user has authorized the device code.
    OpenAI's device token endpoint returns 403/404 while pending.
    When the authorization_code is received, this endpoint automatically
    exchanges it for tokens and returns the refresh_token.

    - 403/404 → {"status": "pending"}
    - success → exchanges code → {"status": "complete", "refresh_token": "...", "account_id": "..."}
    - any other error → {"status": "failed", "error": "..."}

    No DB write is made — the client holds the refresh_token and decides
    what to do with it (store as named credential or use inline).
    """
    async_client = get_async_httpx_client(llm_provider=httpxSpecialProvider.SSO_HANDLER)

    # Step 1: Poll for authorization code
    # OpenAI returns 403/404 while the user hasn't authorized yet.
    # The litellm httpx wrapper calls raise_for_status() automatically,
    # so we catch HTTPStatusError and check the status code ourselves.
    try:
        resp = await async_client.post(
            CHATGPT_DEVICE_TOKEN_URL,
            json={
                "device_auth_id": body.device_auth_id,
                "user_code": body.user_code,
            },
        )
    except _httpx.HTTPStatusError as e:
        if e.response.status_code in (403, 404):
            verbose_proxy_logger.debug("ChatGPT: authorization_pending")
            return StatusResponse(status="pending")
        verbose_proxy_logger.error(f"ChatGPT token poll failed: {e}")
        return StatusResponse(status="failed", error=str(e))
    except Exception as e:
        verbose_proxy_logger.error(f"ChatGPT token poll failed: {e}")
        return StatusResponse(status="failed", error=str(e))

    if resp.status_code in (403, 404):
        verbose_proxy_logger.debug("ChatGPT: authorization_pending")
        return StatusResponse(status="pending")

    if resp.status_code != 200:
        error_text = resp.text[:200] if resp.text else f"HTTP {resp.status_code}"
        verbose_proxy_logger.warning(f"ChatGPT device code poll unexpected response: {error_text}")
        return StatusResponse(status="failed", error=error_text)

    resp_json = resp.json()
    verbose_proxy_logger.debug(f"ChatGPT token poll response keys: {list(resp_json.keys())}")

    authorization_code = resp_json.get("authorization_code")
    code_verifier = resp_json.get("code_verifier")
    if not authorization_code or not code_verifier:
        verbose_proxy_logger.debug("ChatGPT: response 200 but missing authorization_code/code_verifier")
        return StatusResponse(status="pending")

    # Step 2: Exchange authorization code for tokens
    # Use a plain httpx client for the exchange — the litellm wrapper adds
    # raise_for_status hooks that swallow the error body we need for debugging,
    # and may add headers that interfere with the OAuth token endpoint.
    import httpx as _raw_httpx

    try:
        redirect_uri = f"{CHATGPT_AUTH_BASE}/deviceauth/callback"
        exchange_body = (
            "grant_type=authorization_code"
            f"&code={quote(authorization_code, safe='')}"
            f"&redirect_uri={quote(redirect_uri, safe='')}"
            f"&client_id={quote(CHATGPT_CLIENT_ID, safe='')}"
            f"&code_verifier={quote(code_verifier, safe='')}"
        )
        async with _raw_httpx.AsyncClient() as raw_client:
            token_resp = await raw_client.post(
                CHATGPT_OAUTH_TOKEN_URL,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                content=exchange_body,
            )
        if not token_resp.is_success:
            verbose_proxy_logger.error(
                f"ChatGPT token exchange returned {token_resp.status_code}: {token_resp.text[:500]}"
            )
            return StatusResponse(status="failed", error=f"Token exchange failed: HTTP {token_resp.status_code}")
        token_data = token_resp.json()
    except Exception as e:
        verbose_proxy_logger.error(f"ChatGPT token exchange failed: {e}")
        return StatusResponse(status="failed", error=f"Token exchange failed: {e}")

    refresh_token = token_data.get("refresh_token")
    verbose_proxy_logger.info(
        f"ChatGPT token exchange result: keys={list(token_data.keys())}, "
        f"has_refresh={bool(refresh_token)}, refresh_len={len(refresh_token or '')}, "
        f"has_access={bool(token_data.get('access_token'))}"
    )
    if not refresh_token:
        return StatusResponse(
            status="failed",
            error="Token exchange response missing refresh_token",
        )

    # Derive account_id from id_token or access_token
    id_token = token_data.get("id_token")
    access_token = token_data.get("access_token")
    account_id = Authenticator._extract_account_id(id_token or access_token)

    verbose_proxy_logger.info("ChatGPT device code flow completed successfully")
    return StatusResponse(
        status="complete",
        refresh_token=refresh_token,
        account_id=account_id,
    )
