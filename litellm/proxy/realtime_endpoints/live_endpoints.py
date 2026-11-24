"""GPT-Live sessions over WebRTC: the HTTP route that creates one and the sideband that joins it."""

import json
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Annotated, Final

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, Response, WebSocket
from fastapi import status as http_status
from fastapi.responses import JSONResponse

from litellm.litellm_core_utils.core_helpers import as_str_mapping
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,  # pyright: ignore[reportUnknownVariableType]  # untyped dict params
)
from litellm.llms.openai.live.handler import LiveAttach, LiveUpstream
from litellm.proxy._types import ProxyException, UserAPIKeyAuth
from litellm.proxy.auth.auth_checks import can_key_call_resolved_model
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth, user_api_key_auth_live_websocket
from litellm.proxy.common_utils.encrypt_decrypt_utils import decrypt_value_helper, encrypt_value_helper
from litellm.proxy.realtime_endpoints.live_delegation import (
    authorize_live_delegation,
    deployment_delegation,
    live_client_frame,
)
from litellm.proxy.realtime_endpoints.sideband import join_sideband, webrtc_calls_are_metered
from litellm.realtime_api.main import live_guardrails_configured, live_session_upstream
from litellm.types.llms.custom_http import httpxSpecialProvider

router: Final = APIRouter()

_LIVE_SESSION_TOKEN_VERSION: Final = "live_v1"
_LIVE_SESSION_TOKEN_TTL_SECONDS: Final = 24 * 60 * 60
_LIVE_SESSIONS_PATH: Final = "/v1/live/sessions"
_HTTP_SCHEMES: Final[Mapping[str, str]] = {"wss": "https", "ws": "http"}


@dataclass(frozen=True, slots=True)
class _LiveSession:
    """A session that runs upstream: its provider id, the model group asked for, and the deployment that serves it."""

    session_id: str
    model: str
    deployment_id: str


def _session_token(session: _LiveSession, user_api_key_dict: UserAPIKeyAuth) -> str:
    return encrypt_value_helper(
        json.dumps(
            {
                "v": _LIVE_SESSION_TOKEN_VERSION,
                "session_id": session.session_id,
                "model": session.model,
                "deployment_id": session.deployment_id,
                "key": user_api_key_dict.token or "",
                "expires_at": int(time.time()) + _LIVE_SESSION_TOKEN_TTL_SECONDS,
            },
            separators=(",", ":"),
        )
    )


def _attachable_session(session_token: str, user_api_key_dict: UserAPIKeyAuth) -> _LiveSession | None:
    """The session behind a session token, when this caller may join it."""
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
    deployment_id: Final = payload.get("deployment_id")
    expires_at: Final = payload.get("expires_at")
    if not isinstance(session_id, str) or not isinstance(model, str) or not isinstance(expires_at, int):
        return None
    if not isinstance(deployment_id, str):
        return None
    if expires_at < time.time():
        return None
    if not payload.get("key") or payload.get("key") != user_api_key_dict.token:
        return None
    return _LiveSession(session_id=session_id, model=model, deployment_id=deployment_id)


async def _start_live_billing_sideband(session: _LiveSession, user_api_key_dict: UserAPIKeyAuth) -> bool:
    """Meter a WebRTC session from the proxy's own sideband, since its audio and usage bypass the proxy.
    False when the sideband could not join the session."""
    from litellm.proxy.proxy_server import (
        _route_realtime_websocket_session,  # pyright: ignore[reportPrivateUsage]  # the WebSocket session route owns live metering
    )

    return await join_sideband(
        _LIVE_SESSIONS_PATH,
        lambda client, metered_seconds: _route_realtime_websocket_session(
            client,  # pyright: ignore[reportArgumentType]  # only the socket surface the session route reads
            user_api_key_dict.model_copy(),
            route_model=session.model,
            model=None,
            intent=None,
            guardrails=None,
            request_path=_LIVE_SESSIONS_PATH,
            live_session_start={"type": "session.start", "session": {"model": session.model}},
            live_attach=LiveAttach(
                session_id=session.session_id,
                metered=True,
                metered_seconds=metered_seconds,
                on_metered=client.bill_seconds,
            ),
            pinned_deployment_id=session.deployment_id,
        ),
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


async def _post_session(upstream: LiveUpstream, body: Mapping[str, object]) -> httpx.Response:
    """Create the session on the provider. A provider error comes back as its response, for the caller to read."""
    scheme, _, rest = upstream.url.partition("://")
    http_client: Final = get_async_httpx_client(llm_provider=httpxSpecialProvider.PassThroughEndpoint)
    try:
        return await http_client.post(  # pyright: ignore[reportUnknownMemberType]  # untyped dict params
            f"{_HTTP_SCHEMES.get(scheme, scheme)}://{rest}",
            headers={**upstream.headers, "Content-Type": "application/json"},
            json=dict(body),
        )
    except httpx.HTTPStatusError as error:
        return error.response


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
    from litellm.proxy.litellm_pre_call_utils import (
        add_litellm_data_to_request,  # pyright: ignore[reportUnknownVariableType]  # untyped dict params
    )
    from litellm.proxy.proxy_server import (
        general_settings,  # pyright: ignore[reportUnknownVariableType]  # untyped proxy settings dict
        llm_model_list,  # pyright: ignore[reportUnknownVariableType]  # untyped proxy model list
        llm_router,
        proxy_config,
        version,
    )

    requested: Final = await _requested_session(request)
    if requested is None:
        raise HTTPException(status_code=http_status.HTTP_400_BAD_REQUEST, detail="session.model is required")
    body, session, model = requested
    transport: Final = as_str_mapping(body.get("transport"))
    if transport is None or transport.get("type") != "webrtc":
        raise HTTPException(
            status_code=http_status.HTTP_400_BAD_REQUEST, detail="Only the webrtc transport is supported"
        )
    if llm_router is None:
        raise HTTPException(status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR, detail="No router is configured")
    await can_key_call_resolved_model(
        model=model,
        valid_token=user_api_key_dict,
        llm_model_list=llm_model_list,  # pyright: ignore[reportUnknownArgumentType]  # untyped proxy model list
        llm_router=llm_router,
    )
    await authorize_live_delegation(session, user_api_key_dict)
    request_data: Final = as_str_mapping(
        await add_litellm_data_to_request(  # pyright: ignore[reportUnknownArgumentType]  # returns an untyped dict
            data={"model": model},
            request=request,
            general_settings=general_settings,  # pyright: ignore[reportUnknownArgumentType]  # untyped proxy settings dict
            user_api_key_dict=user_api_key_dict,
            version=version,
            proxy_config=proxy_config,
        )
    )
    if request_data is None or live_guardrails_configured(request_data):
        raise HTTPException(
            status_code=http_status.HTTP_400_BAD_REQUEST, detail="Guardrails are not supported on OpenAI Live sessions"
        )

    deployment: Final = as_str_mapping(await llm_router.select(model, dict(request_data)))
    litellm_params: Final = None if deployment is None else as_str_mapping(deployment.get("litellm_params"))
    resolved: Final = None if litellm_params is None else await live_session_upstream(litellm_params)
    if resolved is None:
        raise HTTPException(
            status_code=http_status.HTTP_400_BAD_REQUEST,
            detail="Live sessions over WebRTC require an OpenAI or Azure deployment",
        )
    provider_model, upstream = resolved
    provider_session: Final = deployment_delegation(session, litellm_params or {}, user_api_key_dict)
    upstream_response: Final = await _post_session(
        upstream, {**body, "session": {**provider_session, "model": provider_model}}
    )
    created_parts: Final = _created_session(upstream_response) if upstream_response.is_success else None
    if created_parts is None:
        return Response(
            content=upstream_response.content,
            status_code=upstream_response.status_code,
            media_type=upstream_response.headers.get("content-type", "application/json"),
        )
    created, created_session, upstream_session_id = created_parts
    model_info: Final = None if deployment is None else as_str_mapping(deployment.get("model_info"))
    deployment_id: Final = None if model_info is None else model_info.get("id")
    live_session: Final = _LiveSession(
        session_id=upstream_session_id,
        model=model,
        deployment_id=deployment_id if isinstance(deployment_id, str) else "",
    )
    metered: Final = webrtc_calls_are_metered(general_settings)  # pyright: ignore[reportUnknownArgumentType]  # untyped proxy settings dict
    if metered and not await _start_live_billing_sideband(live_session, user_api_key_dict):
        raise HTTPException(
            status_code=http_status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The proxy could not join this session to meter it, so the session was not started",
        )
    return JSONResponse(
        status_code=upstream_response.status_code,
        content={
            **created,
            "session": {
                **created_session,
                "id": _session_token(live_session, user_api_key_dict),
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
        _authorize_live_session_model,  # pyright: ignore[reportPrivateUsage]  # the same model check a Live WebSocket session gets
        _reject_realtime_session,  # pyright: ignore[reportPrivateUsage]  # closes the socket and releases its holds
        _release_realtime_budget_reservation,  # pyright: ignore[reportPrivateUsage]  # the upgrade auth reserved budget for this socket
        _release_realtime_max_parallel_slot,  # pyright: ignore[reportPrivateUsage]  # the upgrade auth took a parallel slot for this socket
        _route_realtime_websocket_session,  # pyright: ignore[reportPrivateUsage]  # the WebSocket session route relays live sessions
        live_session_model_authorizer,
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
    upgrade_auth: Final = user_api_key_dict.model_copy()
    await _release_realtime_budget_reservation(upgrade_auth)
    try:
        session_auth: Final = await _authorize_live_session_model(
            live_session_model_authorizer(), websocket, attachable.model, None
        )
    except (ProxyException, HTTPException):
        await _reject_realtime_session(websocket, upgrade_auth, code=1008, reason="Session not allowed")
        return
    except BaseException:
        await _release_realtime_max_parallel_slot(upgrade_auth)
        raise
    await _route_realtime_websocket_session(
        websocket,
        session_auth,
        route_model=attachable.model,
        model=None,
        intent=None,
        guardrails=None,
        request_path=websocket.url.path,
        live_session_start={"type": "session.start", "session": {"model": attachable.model}},
        live_attach=LiveAttach(session_id=attachable.session_id, metered=False),
        live_client_frame_gate=lambda client_frame, provider: live_client_frame(client_frame, provider, session_auth),
        pinned_deployment_id=attachable.deployment_id,
    )
