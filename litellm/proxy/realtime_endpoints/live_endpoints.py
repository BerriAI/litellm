"""GPT-Live sessions over WebRTC: the HTTP route that creates one and the sideband that joins it."""

import asyncio
import json
import time
from collections.abc import Mapping
from typing import Annotated, Final

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, Response, WebSocket
from fastapi import status as http_status
from fastapi.responses import JSONResponse

from litellm.litellm_core_utils.core_helpers import as_str_mapping
from litellm.llms.custom_httpx.http_handler import get_async_httpx_client
from litellm.llms.openai.live.handler import LiveAttach
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.auth_checks import can_key_call_resolved_model
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth, user_api_key_auth_live_websocket
from litellm.proxy.common_utils.encrypt_decrypt_utils import decrypt_value_helper, encrypt_value_helper
from litellm.proxy.realtime_endpoints.sideband import SidebandClientWebSocket, keep_sideband_task
from litellm.realtime_api.main import live_session_upstream
from litellm.types.llms.custom_http import httpxSpecialProvider

router: Final = APIRouter()

_LIVE_SESSION_TOKEN_VERSION: Final = "live_v1"
_LIVE_SESSION_TOKEN_TTL_SECONDS: Final = 24 * 60 * 60
_LIVE_SESSIONS_PATH: Final = "/v1/live/sessions"
_HTTP_SCHEMES: Final[Mapping[str, str]] = {"wss": "https", "ws": "http"}


def _session_token(upstream_session_id: str, model: str, user_api_key_dict: UserAPIKeyAuth) -> str:
    return encrypt_value_helper(
        json.dumps(
            {
                "v": _LIVE_SESSION_TOKEN_VERSION,
                "session_id": upstream_session_id,
                "model": model,
                "key": user_api_key_dict.token or "",
                "team_id": user_api_key_dict.team_id or "",
                "expires_at": int(time.time()) + _LIVE_SESSION_TOKEN_TTL_SECONDS,
            },
            separators=(",", ":"),
        )
    )


def _attachable_session(session_token: str, user_api_key_dict: UserAPIKeyAuth) -> tuple[str, str] | None:
    """The upstream session id and model behind a session token, when this caller may join it."""
    decrypted: Final = decrypt_value_helper(value=session_token, key="live_session_id", exception_type="debug")
    if decrypted is None:
        return None
    try:
        payload: Final = as_str_mapping(json.loads(decrypted))
    except ValueError:
        return None
    if payload is None or payload.get("v") != _LIVE_SESSION_TOKEN_VERSION:
        return None
    session_id: Final = payload.get("session_id")
    model: Final = payload.get("model")
    expires_at: Final = payload.get("expires_at")
    if not isinstance(session_id, str) or not isinstance(model, str) or not isinstance(expires_at, int):
        return None
    if expires_at < time.time():
        return None
    same_key: Final = bool(payload.get("key")) and payload.get("key") == user_api_key_dict.token
    same_team: Final = bool(payload.get("team_id")) and payload.get("team_id") == user_api_key_dict.team_id
    return (session_id, model) if same_key or same_team else None


def _start_live_billing_sideband(session_id: str, model: str, user_api_key_dict: UserAPIKeyAuth) -> None:
    """Meter a WebRTC session from the proxy's own sideband, since its audio and usage bypass the proxy."""
    from litellm.proxy.proxy_server import (
        _route_realtime_websocket_session,  # pyright: ignore[reportPrivateUsage]  # the WebSocket session route owns live metering
    )

    keep_sideband_task(
        asyncio.create_task(
            _route_realtime_websocket_session(
                SidebandClientWebSocket(_LIVE_SESSIONS_PATH),  # pyright: ignore[reportArgumentType]  # only the socket surface the session route reads
                user_api_key_dict.model_copy(),
                route_model=model,
                model=None,
                intent=None,
                guardrails=None,
                request_path=_LIVE_SESSIONS_PATH,
                live_session_start={"type": "session.start", "session": {"model": model}},
                live_attach=LiveAttach(session_id=session_id, metered=True),
            )
        )
    )


async def _requested_session(request: Request) -> tuple[Mapping[str, object], Mapping[str, object], str] | None:
    """The request body, its session, and the model the session asks for."""
    try:
        body: Final = as_str_mapping(await request.json())
    except ValueError:
        return None
    session: Final = None if body is None else as_str_mapping(body.get("session"))
    model: Final = None if session is None else session.get("model")
    if body is None or session is None:
        return None
    return (body, session, model) if isinstance(model, str) and model else None


def _created_session(response: httpx.Response) -> tuple[Mapping[str, object], Mapping[str, object], str] | None:
    """The provider's answer, its session, and the session id, or None when the answer names no session."""
    try:
        created: Final = as_str_mapping(response.json())
    except ValueError:
        return None
    session: Final = None if created is None else as_str_mapping(created.get("session"))
    session_id: Final = None if session is None else session.get("id")
    if created is None or session is None:
        return None
    return (created, session, session_id) if isinstance(session_id, str) else None


@router.post("/v1/live/sessions", tags=["realtime"])
@router.post("/live/sessions", tags=["realtime"], include_in_schema=False)
@router.post("/openai/v1/live/sessions", tags=["realtime"], include_in_schema=False)
async def create_live_session(
    request: Request,
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
) -> Response:
    """
    Create a GPT-Live session over WebRTC. The body carries the session and the browser's SDP offer,
    and the response carries the SDP answer. The returned `session.id` attaches a sideband on
    `/v1/live/sessions/{session_id}/attach`.
    """
    from litellm.proxy.proxy_server import llm_model_list, llm_router

    requested: Final = await _requested_session(request)
    if requested is None:
        raise HTTPException(status_code=http_status.HTTP_400_BAD_REQUEST, detail="session.model is required")
    body, session, model = requested
    if llm_router is None:
        raise HTTPException(status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR, detail="No router is configured")
    await can_key_call_resolved_model(
        model=model, valid_token=user_api_key_dict, llm_model_list=llm_model_list, llm_router=llm_router
    )

    deployment: Final = as_str_mapping(await llm_router.select(model, {}))
    litellm_params: Final = None if deployment is None else as_str_mapping(deployment.get("litellm_params"))
    resolved: Final = None if litellm_params is None else await live_session_upstream(litellm_params)
    if resolved is None:
        raise HTTPException(
            status_code=http_status.HTTP_400_BAD_REQUEST,
            detail="Live sessions over WebRTC require an OpenAI or Azure deployment",
        )
    provider_model, upstream = resolved
    scheme, _, rest = upstream.url.partition("://")
    upstream_response: Final = await get_async_httpx_client(llm_provider=httpxSpecialProvider.PassThroughEndpoint).post(
        f"{_HTTP_SCHEMES.get(scheme, scheme)}://{rest}",
        headers={**upstream.headers, "Content-Type": "application/json"},
        json={**body, "session": {**session, "model": provider_model}},
    )
    created_parts: Final = _created_session(upstream_response) if upstream_response.is_success else None
    if created_parts is None:
        return Response(
            content=upstream_response.content,
            status_code=upstream_response.status_code,
            media_type=upstream_response.headers.get("content-type", "application/json"),
        )
    created, created_session, upstream_session_id = created_parts
    _start_live_billing_sideband(upstream_session_id, model, user_api_key_dict)
    return JSONResponse(
        status_code=upstream_response.status_code,
        content={
            **created,
            "session": {
                **created_session,
                "id": _session_token(upstream_session_id, model, user_api_key_dict),
                "model": model,
            },
        },
    )


@router.websocket("/v1/live/sessions/{session_id}/attach")
@router.websocket("/live/sessions/{session_id}/attach")
@router.websocket("/openai/v1/live/sessions/{session_id}/attach")
async def attach_live_session(
    websocket: WebSocket,
    session_id: str,
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth_live_websocket)],
) -> None:
    """Join a WebRTC session as a sideband. Usage is billed once, by the proxy's own sideband."""
    from litellm.proxy.proxy_server import (
        _release_realtime_budget_reservation,  # pyright: ignore[reportPrivateUsage]  # the upgrade auth reserved budget for this socket
        _release_realtime_max_parallel_slot,  # pyright: ignore[reportPrivateUsage]  # the upgrade auth took a parallel slot for this socket
        _route_realtime_websocket_session,  # pyright: ignore[reportPrivateUsage]  # the WebSocket session route relays live sessions
    )

    attachable: Final = _attachable_session(session_id, user_api_key_dict)
    try:
        await websocket.accept()
        if attachable is None:
            await websocket.close(code=1008, reason="Unknown session")
    except BaseException:
        await _release_realtime_budget_reservation(user_api_key_dict)
        await _release_realtime_max_parallel_slot(user_api_key_dict)
        raise
    if attachable is None:
        await _release_realtime_budget_reservation(user_api_key_dict)
        await _release_realtime_max_parallel_slot(user_api_key_dict)
        return
    upstream_session_id, model = attachable
    await _route_realtime_websocket_session(
        websocket,
        user_api_key_dict,
        route_model=model,
        model=None,
        intent=None,
        guardrails=None,
        request_path=websocket.url.path,
        live_session_start={"type": "session.start", "session": {"model": model}},
        live_attach=LiveAttach(session_id=upstream_session_id, metered=False),
    )
