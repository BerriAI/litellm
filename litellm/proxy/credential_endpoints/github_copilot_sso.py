"""
GitHub Copilot SSO endpoints for device code OAuth flow.

The backend is fully stateless during the auth flow. The client holds the
device_code between requests — nothing is stored in the DB until the GitHub
token is successfully validated.

Flow:
  1. POST /credentials/github_copilot/initiate
       Returns device_code + user_code + verification_uri + poll_interval_ms + expires_in.
       No DB write.

  2. POST /credentials/github_copilot/status  (client calls repeatedly)
       Body: { device_code }
       Returns: { status: "pending"|"complete"|"failed", access_token?, retry_after_ms?, error? }
       On slow_down, returns retry_after_ms — client must wait that long before retrying.
       No DB write — client decides what to do with the token.

Design: no session_id, no pending/failed DB rows, no cancel endpoint.
Multi-worker safe: device_code is validated by GitHub, not by LiteLLM.
"""

from typing import Literal, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from litellm._logging import verbose_proxy_logger
from litellm.llms.custom_httpx.http_handler import get_async_httpx_client
from litellm.types.llms.custom_http import httpxSpecialProvider
from litellm.llms.github_copilot.authenticator import (
    GITHUB_ACCESS_TOKEN_URL,
    GITHUB_CLIENT_ID,
    GITHUB_DEVICE_CODE_URL,
    Authenticator,
)
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

router = APIRouter()


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class InitiateResponse(BaseModel):
    device_code: str
    user_code: str
    verification_uri: str
    poll_interval_ms: int
    expires_in: int


class StatusRequest(BaseModel):
    device_code: str


class StatusResponse(BaseModel):
    status: Literal["pending", "complete", "failed"]
    access_token: Optional[str] = None  # complete only
    retry_after_ms: Optional[int] = None  # pending + slow_down only; client must wait this long before retrying
    error: Optional[str] = None  # failed only


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/credentials/github_copilot/initiate",
    tags=["credential management"],
    response_model=InitiateResponse,
)
async def github_copilot_initiate(
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Start the GitHub Device Code flow.

    Calls GitHub and returns device_code, user_code, verification_uri,
    poll_interval_ms, and expires_in to the client. Nothing is stored in
    the database.
    """
    headers = Authenticator.get_github_headers()
    async_client = get_async_httpx_client(llm_provider=httpxSpecialProvider.SSO_HANDLER)
    try:
        resp = await async_client.post(
            GITHUB_DEVICE_CODE_URL,
            headers=headers,
            json={"client_id": GITHUB_CLIENT_ID, "scope": "read:user"},
        )
        resp.raise_for_status()
    except Exception as e:
        verbose_proxy_logger.error(f"GitHub device code request failed: {e}")
        raise HTTPException(
            status_code=502,
            detail={"error": f"GitHub device code request failed: {e}"},
        )
    resp_json = resp.json()

    device_code = resp_json.get("device_code")
    user_code = resp_json.get("user_code")
    verification_uri = resp_json.get("verification_uri")
    poll_interval_seconds = int(resp_json.get("interval", 5))
    expires_in = int(resp_json.get("expires_in", 900))

    if not all([device_code, user_code, verification_uri]):
        raise HTTPException(
            status_code=502,
            detail={"error": "GitHub response missing required fields"},
        )

    return InitiateResponse(
        device_code=device_code,
        user_code=user_code,
        verification_uri=verification_uri,
        poll_interval_ms=poll_interval_seconds * 1000,
        expires_in=expires_in,
    )


@router.post(
    "/credentials/github_copilot/status",
    tags=["credential management"],
    response_model=StatusResponse,
)
async def github_copilot_status(
    body: StatusRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Poll the GitHub Device Code flow for completion.

    Makes a single attempt to exchange the device_code for an access_token.

    - authorization_pending → {"status": "pending"}
    - slow_down → {"status": "pending", "retry_after_ms": <GitHub-reported interval in ms>}
      The client MUST wait retry_after_ms before calling again, or GitHub will
      keep increasing the required interval.
    - success → {"status": "complete", "access_token": "ghu_xxx"}
    - any other error → {"status": "failed", "error": "..."}

    No DB write is made — the client holds the access_token and decides
    what to do with it (store as named credential or use inline).
    """
    headers = Authenticator.get_github_headers()
    async_client = get_async_httpx_client(llm_provider=httpxSpecialProvider.SSO_HANDLER)
    try:
        resp = await async_client.post(
            GITHUB_ACCESS_TOKEN_URL,
            headers=headers,
            json={
                "client_id": GITHUB_CLIENT_ID,
                "device_code": body.device_code,
                "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
            },
        )
        resp.raise_for_status()
        resp_json = resp.json()
    except Exception as e:
        verbose_proxy_logger.error(f"GitHub token poll failed: {e}")
        return StatusResponse(status="failed", error=str(e))

    verbose_proxy_logger.debug(f"GitHub token poll response: {resp_json}")

    if "access_token" in resp_json:
        verbose_proxy_logger.info("GitHub Copilot device code flow completed successfully")
        return StatusResponse(status="complete", access_token=resp_json["access_token"])

    error_code = resp_json.get("error", "")

    if error_code == "slow_down":
        interval = resp_json.get("interval")
        if interval is None:
            verbose_proxy_logger.warning("GitHub slow_down with no interval — treating as failed")
            return StatusResponse(status="failed", error="GitHub slow_down with no interval reported")
        interval_ms = int(interval) * 1000
        verbose_proxy_logger.warning(
            f"GitHub slow_down — telling client to wait {interval_ms}ms before retrying"
        )
        return StatusResponse(status="pending", retry_after_ms=interval_ms)

    if error_code == "authorization_pending":
        verbose_proxy_logger.debug("GitHub Copilot: authorization_pending")
        return StatusResponse(status="pending")

    # Any other error (expired_token, access_denied, etc.)
    error_description = resp_json.get("error_description", error_code)
    verbose_proxy_logger.warning(
        f"GitHub Copilot device code flow failed: error={error_code!r}, "
        f"description={error_description!r}"
    )
    return StatusResponse(status="failed", error=error_description)
