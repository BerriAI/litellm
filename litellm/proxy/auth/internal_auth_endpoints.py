"""
Internal data-plane auth endpoints.

This module exposes the authentication seam used by the Rust ai-gateway
(the data plane). The Rust gateway terminates client connections and needs to
validate the virtual keys it receives WITHOUT reimplementing LiteLLM's key
validation logic. Instead, it calls back into this Python control plane to
verify a key and get the resolved ``UserAPIKeyAuth`` object.

Security model:
  - These endpoints are gated by a DEDICATED data-plane secret
    (``LITELLM_DATA_PLANE_KEY``), NOT the proxy master key. The data plane is a
    distinct trust boundary from proxy admins, so it gets its own credential
    that can be rotated independently and never grants admin access.
  - The data-plane key is compared in constant time to avoid leaking the secret
    via timing side-channels.
"""

import hmac
import os
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

DATA_PLANE_KEY_ENV_VAR = "LITELLM_DATA_PLANE_KEY"
DATA_PLANE_KEY_HEADER = "X-LiteLLM-Data-Plane-Key"


def require_data_plane_key(request: Request) -> None:
    """
    FastAPI dependency that authenticates a request from the Rust data plane.

    Reads the ``X-LiteLLM-Data-Plane-Key`` header and compares it, in constant
    time, against the ``LITELLM_DATA_PLANE_KEY`` environment variable.

    This is intentionally a SEPARATE secret from ``LITELLM_MASTER_KEY`` — the
    data plane is its own trust boundary and must not be granted master-key
    privileges.

    Raises:
        HTTPException 500: if ``LITELLM_DATA_PLANE_KEY`` is unset/empty
            (misconfiguration — fail closed rather than allow unauthenticated
            access).
        HTTPException 401: if the header is missing or does not match.
    """
    expected_key: Optional[str] = os.getenv(DATA_PLANE_KEY_ENV_VAR)
    if not expected_key:
        raise HTTPException(status_code=500, detail="data-plane auth not configured")

    provided_key: Optional[str] = request.headers.get(DATA_PLANE_KEY_HEADER)
    if not provided_key or not hmac.compare_digest(provided_key, expected_key):
        raise HTTPException(status_code=401, detail="invalid data-plane key")


class VerifyKeyRequest(BaseModel):
    api_key: str
    # The actual route the gateway is serving this key on (e.g. "/v1/realtime").
    # REQUIRED and not defaulted: the gateway always sends its own request path,
    # so validation runs route/model restrictions against the real route — never a
    # hardcoded guess, and never this internal endpoint's path (which a normal
    # virtual key isn't allowed to call).
    route: str


def _synthetic_request(route: str, api_key: str) -> Request:
    """
    Build a minimal ASGI request standing in for the client's real call, so
    ``user_api_key_auth`` evaluates the key against the intended data-plane
    ``route`` (and an empty body) instead of this internal endpoint's path.
    """

    async def receive() -> Dict[str, Any]:
        return {"type": "http.request", "body": b"", "more_body": False}

    scope = {
        "type": "http",
        "method": "POST",
        "path": route,
        "raw_path": route.encode(),
        "headers": [(b"authorization", f"Bearer {api_key}".encode())],
        "query_string": b"",
        "scheme": "http",
        "client": ("127.0.0.1", 0),
        "server": ("127.0.0.1", 4000),
    }
    return Request(scope, receive)


router = APIRouter()


@router.post(
    "/internal/v1/auth/verify",
    dependencies=[Depends(require_data_plane_key)],
)
async def verify_key(body: VerifyKeyRequest) -> Dict[str, Any]:
    """
    Verify a virtual key on behalf of the Rust ai-gateway (data plane).

    The Rust gateway forwards the client's virtual key here; this delegates to
    the proxy's existing ``user_api_key_auth`` validation and returns the
    resolved ``UserAPIKeyAuth`` as JSON so the data plane can enforce the same
    limits the control plane would.

    Gated by ``require_data_plane_key`` (the dedicated data-plane secret, not
    the master key).

    On any auth failure the response is a 401 with a minimal body so internals
    are not leaked to the caller.
    """
    from litellm.proxy._types import ProxyException
    from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

    # user_api_key_auth expects the value exactly as the Authorization header
    # arrives — i.e. WITH the "Bearer " prefix (it strips it itself). The gateway
    # sends the bare key, so normalize: add the prefix unless already present.
    bearer_key = (
        body.api_key if body.api_key.startswith("Bearer ") else f"Bearer {body.api_key}"
    )
    synthetic_request = _synthetic_request(route=body.route, api_key=bearer_key)
    try:
        auth = await user_api_key_auth(request=synthetic_request, api_key=bearer_key)
    except (ProxyException, HTTPException):
        # Expected auth failures (invalid / expired / over-budget / blocked) → 401
        # with a minimal body so internals aren't leaked. Unexpected errors (e.g. a
        # DB outage) are deliberately NOT caught here: they surface as 500 rather
        # than masquerading as "invalid key". The data plane still fails closed —
        # it rejects any non-200 from this endpoint.
        raise HTTPException(status_code=401, detail="invalid api key")

    return auth.model_dump(exclude_none=True, mode="json")
