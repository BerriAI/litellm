"""
Authentication endpoints consumed by the Rust data-plane gateway.

The Rust gateway terminates client connections and needs to validate virtual
keys without reimplementing LiteLLM's proxy auth logic. It calls this internal
control-plane route to verify the key against the requested data-plane route
and model.
"""

import hmac
import json
import os
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from litellm.proxy._types import ProxyException
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

DATA_PLANE_KEY_ENV_VAR = "LITELLM_DATA_PLANE_KEY"
DATA_PLANE_KEY_HEADER = "X-LiteLLM-Data-Plane-Key"

router = APIRouter(prefix="/v1/rust_control_plane", tags=["rust control plane"])


def require_data_plane_key(request: Request) -> None:
    """
    Authenticate requests from the Rust data plane with a dedicated secret.

    This intentionally uses ``LITELLM_DATA_PLANE_KEY`` instead of the proxy
    master key: the data plane is a separate trust boundary and must not get
    admin privileges.
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
    # so validation runs route/model restrictions against the real route.
    route: str
    # Forwarded so user_api_key_auth's model access checks run for key, team,
    # and access-group restrictions.
    model: Optional[str] = None


def _synthetic_request(
    route: str, authorization_header: str, model: Optional[str]
) -> Request:
    """
    Build a minimal ASGI request for user_api_key_auth to validate the key
    against the data-plane route and model instead of this internal endpoint.
    """
    body = json.dumps({"model": model} if model is not None else {}).encode()

    async def receive() -> dict[str, Any]:
        return {"type": "http.request", "body": body, "more_body": False}

    scope = {
        "type": "http",
        "method": "POST",
        "path": route,
        "raw_path": route.encode(),
        "headers": [
            (b"authorization", authorization_header.encode()),
            (b"content-type", b"application/json"),
        ],
        "query_string": b"",
        "scheme": "http",
        "client": ("127.0.0.1", 0),
        "server": ("127.0.0.1", 4000),
    }
    request = Request(scope, receive)
    # Admission checks only. Realtime spend is reconciled later through callback
    # logs, so optimistic reservation here would have no request lifecycle to
    # release it.
    request.state.skip_budget_reservation = True
    return request


@router.post(
    "/authentication",
    dependencies=[Depends(require_data_plane_key)],
    include_in_schema=False,
)
async def verify_key(body: VerifyKeyRequest) -> dict[str, Any]:
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
    bearer_key = (
        body.api_key if body.api_key.startswith("Bearer ") else f"Bearer {body.api_key}"
    )
    synthetic_request = _synthetic_request(
        route=body.route, authorization_header=bearer_key, model=body.model
    )
    try:
        auth = await user_api_key_auth(request=synthetic_request, api_key=bearer_key)
    except (ProxyException, HTTPException) as exc:
        status_code = getattr(exc, "status_code", None) or getattr(exc, "code", None)
        try:
            is_server_error = status_code is not None and int(status_code) >= 500
        except (TypeError, ValueError):
            is_server_error = False
        if is_server_error:
            raise
        raise HTTPException(status_code=401, detail="invalid api key")

    return auth.model_dump(exclude_none=True, mode="json")
