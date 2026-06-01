import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import httpx
from fastapi import Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from starlette.background import BackgroundTask

from litellm.proxy.auth.user_api_key_auth import UserAPIKeyAuth, user_api_key_auth
from litellm.proxy.managed_agents_endpoints.endpoints import (
    _assert_owner_or_admin,
    router,
)
from litellm.proxy.managed_agents_endpoints.harness_client import (
    expand_message,
    harness_send_message,
)
from litellm.proxy.managed_agents_endpoints.types import MessageIn

HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
}

# Headers that authenticate the caller to the proxy. Must NOT be forwarded
# to the sandbox container (it runs untrusted user code and could exfiltrate
# the caller's LiteLLM API key or session cookie).
SENSITIVE_HEADERS = {
    "authorization",
    "cookie",
    "x-api-key",
    "x-litellm-api-key",
}


async def _touch_session(prisma_client, session_id: str) -> None:
    try:
        await prisma_client.db.litellm_managedagentsessiontable.update(
            where={"session_id": session_id},
            data={"last_seen_at": datetime.now(timezone.utc)},
        )
    except Exception:
        pass


# Module-level shared client. Reused across all passthrough requests so we
# get HTTP connection-pool keepalive (one TCP+TLS handshake per sandbox host
# instead of one per chat message). Lazily initialized inside the running
# event loop. Closed on proxy shutdown via close_passthrough_http_client.
_HTTP_CLIENT: Optional[httpx.AsyncClient] = None
_HTTP_CLIENT_LOCK = asyncio.Lock()


async def _get_http_client() -> httpx.AsyncClient:
    global _HTTP_CLIENT
    if _HTTP_CLIENT is None or _HTTP_CLIENT.is_closed:
        async with _HTTP_CLIENT_LOCK:
            if _HTTP_CLIENT is None or _HTTP_CLIENT.is_closed:
                _HTTP_CLIENT = httpx.AsyncClient(
                    timeout=httpx.Timeout(connect=10, read=None, write=None, pool=10),
                    limits=httpx.Limits(
                        max_connections=200, max_keepalive_connections=50
                    ),
                )
    return _HTTP_CLIENT


async def close_passthrough_http_client() -> None:
    """Called from proxy shutdown to drain pooled connections."""
    global _HTTP_CLIENT
    if _HTTP_CLIENT is not None and not _HTTP_CLIENT.is_closed:
        await _HTTP_CLIENT.aclose()
    _HTTP_CLIENT = None


@router.post("/sessions/{session_id}/message")
async def session_message(
    session_id: str,
    body: MessageIn,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> Dict[str, Any]:
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=500, detail="prisma client not available")

    row = await prisma_client.db.litellm_managedagentsessiontable.find_unique(
        where={"session_id": session_id},
        include={"agent": True},
    )
    if row is None:
        raise HTTPException(status_code=404, detail="session not found")
    _assert_owner_or_admin(user_api_key_dict, row.created_by, "session", session_id)
    if (
        row.status != "ready"
        or row.sandbox_url is None
        or row.harness_session_id is None
    ):
        raise HTTPException(
            status_code=409,
            detail=f"session not ready (status={row.status})",
        )

    parts = expand_message(body.text, body.parts)

    client = await _get_http_client()
    try:
        result = await harness_send_message(
            row.sandbox_url,
            row.harness_session_id,
            client,
            model=row.agent.model,
            parts=parts,
        )
    except httpx.HTTPStatusError as e:
        raise HTTPException(e.response.status_code, e.response.text)
    except httpx.HTTPError as e:
        raise HTTPException(502, f"upstream error: {e}")

    asyncio.create_task(_touch_session(prisma_client, session_id))
    return result


@router.get("/sessions/{session_id}/events")
async def session_events(
    session_id: str,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> StreamingResponse:
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=500, detail="prisma client not available")

    row = await prisma_client.db.litellm_managedagentsessiontable.find_unique(
        where={"session_id": session_id}
    )
    if row is None:
        raise HTTPException(status_code=404, detail="session not found")
    _assert_owner_or_admin(user_api_key_dict, row.created_by, "session", session_id)
    if row.status != "ready" or row.sandbox_url is None:
        raise HTTPException(
            status_code=409,
            detail=f"session not ready (status={row.status})",
        )

    client = await _get_http_client()
    req = client.build_request("GET", f"{row.sandbox_url}/event", timeout=None)
    try:
        upstream = await client.send(req, stream=True)
    except httpx.HTTPError as e:
        raise HTTPException(502, f"upstream error: {e}")

    resp_headers = {
        k: v for k, v in upstream.headers.items() if k.lower() not in HOP_BY_HOP
    }

    return StreamingResponse(
        upstream.aiter_raw(),
        status_code=upstream.status_code,
        headers=resp_headers,
        media_type=upstream.headers.get("content-type", "text/event-stream"),
        background=BackgroundTask(upstream.aclose),
    )


@router.api_route(
    "/sessions/{session_id}/raw/{path:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
)
async def session_raw_proxy(
    session_id: str,
    path: str,
    request: Request,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> StreamingResponse:
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=500, detail="prisma client not available")

    row = await prisma_client.db.litellm_managedagentsessiontable.find_unique(
        where={"session_id": session_id}
    )
    if row is None:
        raise HTTPException(status_code=404, detail="session not found")
    _assert_owner_or_admin(user_api_key_dict, row.created_by, "session", session_id)
    if row.status != "ready" or row.sandbox_url is None:
        raise HTTPException(
            status_code=409,
            detail=f"session not ready (status={row.status})",
        )

    target = f"{row.sandbox_url}/{path}"
    fwd_headers = {
        k: v
        for k, v in request.headers.items()
        if k.lower() not in HOP_BY_HOP
        and k.lower() not in SENSITIVE_HEADERS
        and k.lower() != "host"
    }
    body = await request.body()

    client = await _get_http_client()
    req = client.build_request(
        method=request.method,
        url=target,
        params=request.query_params,
        headers=fwd_headers,
        content=body if body else None,
    )
    try:
        upstream = await client.send(req, stream=True)
    except httpx.HTTPError as e:
        raise HTTPException(502, f"upstream error: {e}")

    asyncio.create_task(_touch_session(prisma_client, session_id))

    resp_headers = {
        k: v for k, v in upstream.headers.items() if k.lower() not in HOP_BY_HOP
    }

    return StreamingResponse(
        upstream.aiter_raw(),
        status_code=upstream.status_code,
        headers=resp_headers,
        background=BackgroundTask(upstream.aclose),
    )
