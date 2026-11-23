"""Controls on one GPT-Live session: inbound accept and reject, call control, fork, recording, and sideband attach."""

import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Annotated, Final

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, Response, WebSocket
from fastapi import status as http_status
from starlette.datastructures import URL, Headers
from starlette.responses import StreamingResponse
from starlette.types import Receive, Scope, Send

from litellm.litellm_core_utils.core_helpers import as_object_list, as_str_mapping, loads_json_object
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,  # pyright: ignore[reportUnknownVariableType]  # untyped dict params
)
from litellm.llms.openai.live.handler import LiveAttach
from litellm.proxy._types import LitellmUserRoles, ProxyException, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth, user_api_key_auth_live_websocket
from litellm.proxy.common_utils.custom_openapi_spec import inline_request_body
from litellm.proxy.realtime_endpoints.live_delegation import (
    authorize_live_delegation,
    deployment_delegation,
    live_client_frame,
)
from litellm.proxy.realtime_endpoints.live_endpoints import (
    _admit_session,  # pyright: ignore[reportPrivateUsage]  # one Live feature split across two modules
    _attachable_session,  # pyright: ignore[reportPrivateUsage]  # one Live feature split across two modules
    _authorize_session_model,  # pyright: ignore[reportPrivateUsage]  # one Live feature split across two modules
    _created_session,  # pyright: ignore[reportPrivateUsage]  # one Live feature split across two modules
    _http_url,  # pyright: ignore[reportPrivateUsage]  # one Live feature split across two modules
    _LiveSession,  # pyright: ignore[reportPrivateUsage]  # one Live feature split across two modules
    _post_session,  # pyright: ignore[reportPrivateUsage]  # one Live feature split across two modules
    _provider_answer,  # pyright: ignore[reportPrivateUsage]  # one Live feature split across two modules
    _provider_content_type,  # pyright: ignore[reportPrivateUsage]  # one Live feature split across two modules
    _refusal_unless_metered,  # pyright: ignore[reportPrivateUsage]  # one Live feature split across two modules
    _requested_session,  # pyright: ignore[reportPrivateUsage]  # one Live feature split across two modules
    _resolved_deployment,  # pyright: ignore[reportPrivateUsage]  # one Live feature split across two modules
    _ResolvedDeployment,  # pyright: ignore[reportPrivateUsage]  # one Live feature split across two modules
    _routing_data,  # pyright: ignore[reportPrivateUsage]  # one Live feature split across two modules
    _selected_deployment,  # pyright: ignore[reportPrivateUsage]  # one Live feature split across two modules
    _session_action_path,  # pyright: ignore[reportPrivateUsage]  # one Live feature split across two modules
    _session_token,  # pyright: ignore[reportPrivateUsage]  # one Live feature split across two modules
    _tokenized_answer,  # pyright: ignore[reportPrivateUsage]  # one Live feature split across two modules
    _unguarded_routing_data,  # pyright: ignore[reportPrivateUsage]  # one Live feature split across two modules
    _usable_session,  # pyright: ignore[reportPrivateUsage]  # one Live feature split across two modules
)
from litellm.types.llms.custom_http import httpxSpecialProvider
from litellm.types.realtime import (
    LiveSessionAcceptRequest,
    LiveSessionForkRequest,
    LiveSessionForkResponse,
    LiveSessionReferRequest,
    LiveSessionRejectRequest,
)

router: Final = APIRouter()
attach_router: Final = APIRouter()

_RAW_SESSION_ID: Final = re.compile(r"live_[A-Za-z0-9_-]+")
_INBOUND_CONTROL_GRANTS: Final = frozenset({"/live/sessions/*", "/v1/live/sessions/*"})
_UNKNOWN_SESSION_DETAIL: Final = "Session not found"
_UNKNOWN_SESSION_REASON: Final = "Unknown session"
_RAW_ID_NEEDS_MODEL: Final = "A raw session id needs the model query parameter"
_INVALID_FORK_START: Final = (
    "First message must be a session.start JSON object whose session.model, if set, is the source session's model"
)
_CALL_CONTROL_ANSWER: Final = {
    "description": "The provider applied the action. OpenAI answers with an empty body.",
    "content": {"application/json": {"schema": {"type": "object"}}},
}
_ACCEPT_BODY: Final = {
    "requestBody": inline_request_body(LiveSessionAcceptRequest, {"session": {"type": "live", "model": "gpt-live-1.1"}})
}
_REJECT_BODY: Final = {"requestBody": inline_request_body(LiveSessionRejectRequest, {"status_code": 486})}
_REFER_BODY: Final = {"requestBody": inline_request_body(LiveSessionReferRequest, {"target_uri": "tel:+14155550123"})}
_HANGUP_BODY: Final = {
    "requestBody": {
        "required": False,
        "content": {"application/json": {"schema": {"type": "object", "maxProperties": 0}}},
    }
}
_FORK_BODY: Final = {
    "requestBody": inline_request_body(
        LiveSessionForkRequest, {"transport": {"type": "webrtc", "sdp": "v=0..."}, "session": {}}
    )
}


@dataclass(frozen=True, slots=True)
class _ControlTarget:
    session: _LiveSession
    deployment: _ResolvedDeployment
    stored: bool


def _unknown_session() -> HTTPException:
    return HTTPException(status_code=http_status.HTTP_404_NOT_FOUND, detail=_UNKNOWN_SESSION_DETAIL)


def _names_inbound_grant(allowed_routes: object) -> bool:
    return any(
        isinstance(route, str) and route in _INBOUND_CONTROL_GRANTS for route in as_object_list(allowed_routes) or ()
    )


def _holds_inbound_grant(caller: UserAPIKeyAuth) -> bool:
    if caller.user_role == LitellmUserRoles.PROXY_ADMIN:
        return True
    return _names_inbound_grant(
        caller.allowed_routes  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]  # bare list field
    )


def _pinned_router_deployment(session: _LiveSession, caller: UserAPIKeyAuth) -> Mapping[str, object] | None:
    from litellm.proxy.proxy_server import llm_router

    if llm_router is None or not session.deployment_id:
        return None
    deployments: Final = llm_router.registry().deployments(session.model, team_id=caller.team_id) or ()
    candidates: Final = (as_str_mapping(deployment) for deployment in deployments)
    return next(
        (
            deployment
            for deployment in candidates
            if deployment is not None
            and (as_str_mapping(deployment.get("model_info")) or {}).get("id") == session.deployment_id
        ),
        None,
    )


async def _owned_target(
    session_id: str, caller: UserAPIKeyAuth, *, retained: bool, needs_model_access: bool = True
) -> _ControlTarget | None:
    owned: Final = _usable_session(session_id, caller, retained=retained)
    if owned is None:
        return None
    if needs_model_access:
        await _authorize_session_model(owned.session.model, caller)
    deployment: Final = await _resolved_deployment(_pinned_router_deployment(owned.session, caller))
    if deployment is None:
        raise _unknown_session()
    return _ControlTarget(session=owned.session, deployment=deployment, stored=owned.retained_until is not None)


async def _inbound_target(request: Request, session_id: str, caller: UserAPIKeyAuth) -> _ControlTarget:
    if _RAW_SESSION_ID.fullmatch(session_id) is None or not _holds_inbound_grant(caller):
        raise _unknown_session()
    model: Final = request.query_params.get("model")
    if not model:
        raise HTTPException(status_code=http_status.HTTP_400_BAD_REQUEST, detail=_RAW_ID_NEEDS_MODEL)
    await _authorize_session_model(model, caller)
    deployment: Final = await _selected_deployment(model, await _routing_data(request, caller, model))
    return _ControlTarget(
        session=_LiveSession(session_id=session_id, model=model, deployment_id=deployment.deployment_id),
        deployment=deployment,
        stored=False,
    )


async def _call_target(request: Request, session_id: str, caller: UserAPIKeyAuth, action: str) -> _ControlTarget:
    owned: Final = await _owned_target(session_id, caller, retained=False, needs_model_access=action != "hangup")
    return owned if owned is not None else await _inbound_target(request, session_id, caller)


async def _stored_target(session_id: str, caller: UserAPIKeyAuth) -> _ControlTarget:
    owned: Final = await _owned_target(session_id, caller, retained=True)
    if owned is None:
        raise _unknown_session()
    return owned


async def _optional_json_body(request: Request) -> Mapping[str, object] | None:
    raw_body: Final = await request.body()
    if not raw_body.strip():
        return None
    body: Final = loads_json_object(raw_body)
    if body is None:
        raise HTTPException(
            status_code=http_status.HTTP_400_BAD_REQUEST, detail="The request body must be a JSON object"
        )
    return body


async def _forward_call_control(request: Request, session_id: str, caller: UserAPIKeyAuth, action: str) -> Response:
    target: Final = await _call_target(request, session_id, caller, action)
    body: Final = await _optional_json_body(request)
    answer: Final = await _post_session(
        target.deployment.upstream, body, _session_action_path(target.session.session_id, action)
    )
    return _provider_answer(answer, body)


def _fork_overrides(body: Mapping[str, object], source_model: str) -> Mapping[str, object]:
    if "session" not in body:
        return {}
    overrides: Final = as_str_mapping(body.get("session"))
    if overrides is None:
        raise HTTPException(status_code=http_status.HTTP_400_BAD_REQUEST, detail="session must be an object")
    if overrides.get("model", source_model) != source_model:
        raise HTTPException(
            status_code=http_status.HTTP_400_BAD_REQUEST,
            detail="A fork cannot change session.model to another model group",
        )
    return {name: value for name, value in overrides.items() if name != "model"}


class _UpstreamRecording(StreamingResponse):
    def __init__(self, upstream: httpx.Response) -> None:
        super().__init__(
            content=upstream.aiter_bytes(),
            status_code=upstream.status_code,
            media_type=_provider_content_type(upstream.headers, "application/octet-stream"),
        )
        self._upstream: Final = upstream

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            await self._upstream.aclose()


def _started_session_id(frame: str) -> str | None:
    event: Final = loads_json_object(frame)
    if event is None or event.get("type") != "session.started":
        return None
    session: Final = as_str_mapping(event.get("session"))
    session_id: Final = None if session is None else session.get("id")
    return session_id if isinstance(session_id, str) and session_id else None


class _StoredSessionSocket:
    def __init__(self, client: WebSocket, mint_token: Callable[[str], str]) -> None:
        self._client: Final = client
        self._mint_token: Final = mint_token
        self._id_and_token: tuple[str, str] | None = None

    @property
    def scope(self) -> Scope:
        return self._client.scope

    @property
    def headers(self) -> Headers:
        return self._client.headers

    @property
    def url(self) -> URL:
        return self._client.url

    async def receive_text(self) -> str:
        return await self._client.receive_text()

    async def send_text(self, data: str) -> None:
        await self._client.send_text(self._with_token(data))

    async def close(self, code: int = 1000, reason: str | None = None) -> None:
        await self._client.close(code=code, reason=reason)

    def _with_token(self, frame: str) -> str:
        if self._id_and_token is None:
            session_id: Final = _started_session_id(frame)
            if session_id is None:
                return frame
            self._id_and_token = (session_id, self._mint_token(session_id))
        provider_id, token = self._id_and_token
        return frame.replace(json.dumps(provider_id), json.dumps(token))


@dataclass(frozen=True, slots=True)
class StoredSessionRelay:
    client: _StoredSessionSocket
    deployment_id: str


def _stored_session_socket(
    websocket: WebSocket, model: str, deployment_id: str, caller: UserAPIKeyAuth
) -> _StoredSessionSocket:
    return _StoredSessionSocket(
        websocket,
        lambda session_id: _session_token(
            _LiveSession(session_id=session_id, model=model, deployment_id=deployment_id), caller, stored=True
        ),
    )


async def stored_session_relay(
    websocket: WebSocket, session_start: Mapping[str, object], model: str, caller: UserAPIKeyAuth
) -> StoredSessionRelay | None:
    from litellm.proxy.proxy_server import llm_router
    from litellm.proxy.realtime_endpoints.endpoints import (
        _serving_deployment_id,  # pyright: ignore[reportPrivateUsage]  # the same up-front deployment pick a WebRTC call gets
    )

    session: Final = as_str_mapping(session_start.get("session"))
    if session is None or session.get("store") is not True:
        return None
    deployment_id: Final = await _serving_deployment_id(
        llm_router, model, {"litellm_metadata": {"user_api_key_team_id": caller.team_id}}
    )
    if deployment_id is None:
        return None
    return StoredSessionRelay(
        client=_stored_session_socket(websocket, model, deployment_id, caller), deployment_id=deployment_id
    )


async def _accept_or_refuse(websocket: WebSocket, caller: UserAPIKeyAuth, refusal: str | None) -> bool:
    from litellm.proxy.proxy_server import (
        _release_realtime_budget_reservation,  # pyright: ignore[reportPrivateUsage]  # the upgrade auth reserved budget for this socket
        _release_realtime_max_parallel_slot,  # pyright: ignore[reportPrivateUsage]  # the upgrade auth took a parallel slot for this socket
    )

    try:
        await websocket.accept()
        if refusal is not None:
            await websocket.close(code=1008, reason=refusal)
    except BaseException:
        await _release_realtime_budget_reservation(caller)
        await _release_realtime_max_parallel_slot(caller)
        raise
    if refusal is None:
        return True
    await _release_realtime_budget_reservation(caller)
    await _release_realtime_max_parallel_slot(caller)
    return False


def _attach_target(websocket: WebSocket, session_id: str, caller: UserAPIKeyAuth) -> _LiveSession | str:
    owned: Final = _attachable_session(session_id, caller)
    if owned is not None:
        return owned
    if _RAW_SESSION_ID.fullmatch(session_id) is None or not _holds_inbound_grant(caller):
        return _UNKNOWN_SESSION_REASON
    model: Final = websocket.query_params.get("model")
    if not model:
        return _RAW_ID_NEEDS_MODEL
    return _LiveSession(session_id=session_id, model=model, deployment_id="")


def _fork_session_start(first_message: str, source_model: str) -> Mapping[str, object] | None:
    frame: Final = loads_json_object(first_message)
    if frame is None or frame.get("type") != "session.start":
        return None
    overrides: Final = as_str_mapping(frame.get("session", {}))
    if overrides is None or overrides.get("model", source_model) != source_model:
        return None
    return {**frame, "session": {**overrides, "model": source_model}}


@router.post(
    "/v1/live/sessions/{session_id}/accept",
    tags=["realtime"],
    responses={200: _CALL_CONTROL_ANSWER},
    openapi_extra=_ACCEPT_BODY,
)
@router.post("/live/sessions/{session_id}/accept", tags=["realtime"], include_in_schema=False)
async def accept_live_session(
    session_id: str,
    request: Request,
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
) -> Response:
    """Accept an inbound call by its raw `live_...` id. Needs a proxy admin or a key granted inbound control."""
    if _RAW_SESSION_ID.fullmatch(session_id) is None or not _holds_inbound_grant(user_api_key_dict):
        raise _unknown_session()
    requested: Final = await _requested_session(request)
    if requested is None:
        raise HTTPException(status_code=http_status.HTTP_400_BAD_REQUEST, detail="session.model is required")
    body, session, model = requested
    admitted: Final = await _admit_session(request, user_api_key_dict, session, model)
    deployment: Final = admitted.deployment
    answer: Final = await _post_session(
        deployment.upstream,
        {**body, "session": {**admitted.provider_session, "model": deployment.provider_model}},
        _session_action_path(session_id, "accept"),
    )
    if not answer.is_success:
        return _provider_answer(answer, body)
    refusal: Final = await _refusal_unless_metered(
        _LiveSession(session_id=session_id, model=model, deployment_id=deployment.deployment_id),
        deployment.upstream,
        user_api_key_dict,
        call_is_live=True,
        stored=session.get("store") is True,
    )
    return refusal if refusal is not None else _provider_answer(answer, body)


@router.post(
    "/v1/live/sessions/{session_id}/reject",
    tags=["realtime"],
    responses={200: _CALL_CONTROL_ANSWER},
    openapi_extra=_REJECT_BODY,
)
@router.post("/live/sessions/{session_id}/reject", tags=["realtime"], include_in_schema=False)
async def reject_live_session(
    session_id: str,
    request: Request,
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
) -> Response:
    """Reject an inbound call. A raw `live_...` id needs inbound control and a `?model=` query parameter."""
    return await _forward_call_control(request, session_id, user_api_key_dict, "reject")


@router.post(
    "/v1/live/sessions/{session_id}/refer",
    tags=["realtime"],
    responses={200: _CALL_CONTROL_ANSWER},
    openapi_extra=_REFER_BODY,
)
@router.post("/live/sessions/{session_id}/refer", tags=["realtime"], include_in_schema=False)
async def refer_live_session(
    session_id: str,
    request: Request,
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
) -> Response:
    """Transfer a SIP call. Takes the proxy session token, or a raw id with inbound control and `?model=`."""
    return await _forward_call_control(request, session_id, user_api_key_dict, "refer")


@router.post(
    "/v1/live/sessions/{session_id}/hangup",
    tags=["realtime"],
    responses={200: _CALL_CONTROL_ANSWER},
    openapi_extra=_HANGUP_BODY,
)
@router.post("/live/sessions/{session_id}/hangup", tags=["realtime"], include_in_schema=False)
async def hangup_live_session(
    session_id: str,
    request: Request,
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
) -> Response:
    """End a call. Takes the proxy session token, or a raw id with inbound control and `?model=`."""
    return await _forward_call_control(request, session_id, user_api_key_dict, "hangup")


@router.post(
    "/v1/live/sessions/{session_id}/fork",
    tags=["realtime"],
    status_code=http_status.HTTP_201_CREATED,
    responses={201: {"model": LiveSessionForkResponse}},
    openapi_extra=_FORK_BODY,
)
@router.post("/live/sessions/{session_id}/fork", tags=["realtime"], include_in_schema=False)
async def fork_live_session(
    session_id: str,
    request: Request,
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
) -> Response:
    """Start a new session from a stored one over WebRTC. The answer carries a new session token and is metered."""
    target: Final = await _stored_target(session_id, user_api_key_dict)
    body: Final = await _optional_json_body(request) or {}
    transport: Final = as_str_mapping(body.get("transport"))
    if transport is None:
        raise HTTPException(status_code=http_status.HTTP_400_BAD_REQUEST, detail="transport must be an object")
    overrides: Final = _fork_overrides(body, target.session.model)
    await authorize_live_delegation(overrides, user_api_key_dict)
    await _unguarded_routing_data(request, user_api_key_dict, target.session.model)
    provider_overrides: Final = deployment_delegation(overrides, target.deployment.litellm_params, user_api_key_dict)
    forwarded: Final = {**body, "session": provider_overrides} if "session" in body else body
    answer: Final = await _post_session(
        target.deployment.upstream, forwarded, _session_action_path(target.session.session_id, "fork")
    )
    created_parts: Final = _created_session(answer) if answer.is_success else None
    if created_parts is None:
        return _provider_answer(answer, body)
    created, created_session, forked_id = created_parts
    forked: Final = _LiveSession(
        session_id=forked_id, model=target.session.model, deployment_id=target.session.deployment_id
    )
    store_override: Final = overrides.get("store")
    stored: Final = store_override if isinstance(store_override, bool) else target.stored
    refusal: Final = await _refusal_unless_metered(
        forked,
        target.deployment.upstream,
        user_api_key_dict,
        call_is_live=transport.get("type") == "sip",
        stored=stored,
    )
    if refusal is not None:
        return refusal
    return _tokenized_answer(answer, created, created_session, forked, user_api_key_dict, stored=stored)


@router.get("/v1/live/sessions/{session_id}/content", tags=["realtime"])
@router.get("/live/sessions/{session_id}/content", tags=["realtime"], include_in_schema=False)
async def download_live_session_content(
    session_id: str,
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
) -> Response:
    """Stream a stored session's recording from the provider."""
    target: Final = await _stored_target(session_id, user_api_key_dict)
    upstream: Final = target.deployment.upstream
    http_client: Final = get_async_httpx_client(llm_provider=httpxSpecialProvider.PassThroughEndpoint)
    try:
        recording: Final = await http_client.get(  # pyright: ignore[reportUnknownMemberType]  # untyped dict params
            _http_url(upstream, _session_action_path(target.session.session_id, "content")),
            headers=dict(upstream.headers),
            stream=True,
        )
    except httpx.HTTPStatusError as error:
        return _provider_answer(error.response, None)
    return _UpstreamRecording(recording)


@router.websocket("/v1/live/sessions/{session_id}/fork")
@router.websocket("/live/sessions/{session_id}/fork")
async def fork_live_session_socket(
    websocket: WebSocket,
    session_id: str,
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth_live_websocket)],
) -> None:
    """Start a new session from a stored one over a WebSocket. Its relay bills it like a primary session."""
    from litellm.proxy.proxy_server import (
        _authorize_live_session_model,  # pyright: ignore[reportPrivateUsage]  # the same model check a Live WebSocket session gets
        _receive_live_session_start,  # pyright: ignore[reportPrivateUsage]  # the first-frame wait a Live WebSocket session gets
        _reject_realtime_session,  # pyright: ignore[reportPrivateUsage]  # closes the socket and releases its holds
        _release_realtime_budget_reservation,  # pyright: ignore[reportPrivateUsage]  # the upgrade auth reserved budget for this socket
        _release_realtime_max_parallel_slot,  # pyright: ignore[reportPrivateUsage]  # the upgrade auth took a parallel slot for this socket
        _route_realtime_websocket_session,  # pyright: ignore[reportPrivateUsage]  # the WebSocket session route relays live sessions
        live_session_model_authorizer,
    )

    owned: Final = _usable_session(session_id, user_api_key_dict, retained=True)
    source: Final = (
        None if owned is None or _pinned_router_deployment(owned.session, user_api_key_dict) is None else owned
    )
    if not await _accept_or_refuse(
        websocket, user_api_key_dict, None if source is not None else _UNKNOWN_SESSION_REASON
    ):
        return
    if source is None:
        return
    first_message: Final = await _receive_live_session_start(websocket, user_api_key_dict)
    if first_message is None:
        return
    model: Final = source.session.model
    session_start: Final = _fork_session_start(first_message, model)
    if session_start is None:
        await _reject_realtime_session(
            websocket,
            user_api_key_dict,
            code=1008,
            reason="Invalid session.start frame",
            error_message=_INVALID_FORK_START,
            error_type="invalid_request_error",
            error_code="invalid_session_start",
        )
        return
    upgrade_auth: Final = user_api_key_dict.model_copy()
    await _release_realtime_budget_reservation(upgrade_auth)
    try:
        session_auth: Final = await _authorize_live_session_model(
            live_session_model_authorizer(), websocket, model, session_start.get("session")
        )
    except (ProxyException, HTTPException):
        await _reject_realtime_session(websocket, upgrade_auth, code=1008, reason="Session not allowed")
        return
    except BaseException:
        await _release_realtime_max_parallel_slot(upgrade_auth)
        raise
    overrides: Final = as_str_mapping(session_start.get("session")) or {}
    store_override: Final = overrides.get("store")
    stored: Final = store_override if isinstance(store_override, bool) else source.retained_until is not None
    await _route_realtime_websocket_session(
        _stored_session_socket(websocket, model, source.session.deployment_id, session_auth)  # pyright: ignore[reportArgumentType]  # only the socket surface the session route reads
        if stored
        else websocket,
        session_auth,
        route_model=model,
        model=None,
        intent=None,
        guardrails=None,
        request_path=websocket.url.path,
        live_session_start=session_start,
        live_fork_of=source.session.session_id,
        live_client_frame_gate=lambda client_frame, provider: live_client_frame(client_frame, provider, session_auth),
        pinned_deployment_id=source.session.deployment_id,
    )


@attach_router.websocket("/v1/live/sessions/{session_id}/attach")
@attach_router.websocket("/live/sessions/{session_id}/attach")
@attach_router.websocket("/openai/v1/live/sessions/{session_id}/attach")
async def attach_live_session(
    websocket: WebSocket,
    session_id: str,
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth_live_websocket)],
) -> None:
    """Join a running session as a sideband. Usage is billed once, by the proxy's own sideband."""
    from litellm.proxy.proxy_server import (
        _authorize_live_session_model,  # pyright: ignore[reportPrivateUsage]  # the same model check a Live WebSocket session gets
        _reject_realtime_session,  # pyright: ignore[reportPrivateUsage]  # closes the socket and releases its holds
        _release_realtime_budget_reservation,  # pyright: ignore[reportPrivateUsage]  # the upgrade auth reserved budget for this socket
        _release_realtime_max_parallel_slot,  # pyright: ignore[reportPrivateUsage]  # the upgrade auth took a parallel slot for this socket
        _route_realtime_websocket_session,  # pyright: ignore[reportPrivateUsage]  # the WebSocket session route relays live sessions
        live_session_model_authorizer,
    )

    target: Final = _attach_target(websocket, session_id, user_api_key_dict)
    if not await _accept_or_refuse(websocket, user_api_key_dict, target if isinstance(target, str) else None):
        return
    if isinstance(target, str):
        return
    upgrade_auth: Final = user_api_key_dict.model_copy()
    await _release_realtime_budget_reservation(upgrade_auth)
    try:
        session_auth: Final = await _authorize_live_session_model(
            live_session_model_authorizer(), websocket, target.model, None
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
        route_model=target.model,
        model=None,
        intent=None,
        guardrails=None,
        request_path=websocket.url.path,
        live_session_start={"type": "session.start", "session": {"model": target.model}},
        live_attach=LiveAttach(session_id=target.session_id, metered=False),
        live_client_frame_gate=lambda client_frame, provider: live_client_frame(client_frame, provider, session_auth),
        pinned_deployment_id=target.deployment_id or None,
    )
