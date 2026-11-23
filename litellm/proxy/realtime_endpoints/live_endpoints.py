"""GPT-Live sessions over WebRTC and SIP: the HTTP route that creates one and the sideband that joins it."""

import json
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Annotated, Final
from urllib.parse import quote

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi import status as http_status
from fastapi.responses import JSONResponse

from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.core_helpers import as_str_mapping
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,  # pyright: ignore[reportUnknownVariableType]  # untyped dict params
)
from litellm.llms.openai.live.handler import LiveAttach, LiveUpstream
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.auth_checks import can_key_call_resolved_model
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.common_utils.encrypt_decrypt_utils import decrypt_value_helper, encrypt_value_helper
from litellm.proxy.realtime_endpoints.live_delegation import authorize_live_delegation, deployment_delegation
from litellm.proxy.realtime_endpoints.sideband import join_sideband, webrtc_calls_are_metered
from litellm.realtime_api.main import live_guardrails_configured, live_session_upstream
from litellm.types.llms.custom_http import httpxSpecialProvider

router: Final = APIRouter()

_LIVE_SESSION_TOKEN_VERSION: Final = "live_v1"
_LIVE_SESSION_TOKEN_TTL_SECONDS: Final = 24 * 60 * 60
_LIVE_SESSION_RETENTION_SECONDS: Final = 30 * 24 * 60 * 60
_LIVE_SESSIONS_PATH: Final = "/v1/live/sessions"
_HTTP_SCHEMES: Final[Mapping[str, str]] = {"wss": "https", "ws": "http"}
_REDACTED_SECRET: Final = b"[REDACTED]"


@dataclass(frozen=True, slots=True)
class _LiveSession:
    """A session that runs upstream: its provider id, the model group asked for, and the deployment that serves it."""

    session_id: str
    model: str
    deployment_id: str


@dataclass(frozen=True, slots=True)
class _OwnedSession:
    session: _LiveSession
    expires_at: int
    retained_until: int | None


@dataclass(frozen=True, slots=True)
class _ResolvedDeployment:
    provider_model: str
    upstream: LiveUpstream
    litellm_params: Mapping[str, object]
    deployment_id: str


@dataclass(frozen=True, slots=True)
class _AdmittedSession:
    deployment: _ResolvedDeployment
    provider_session: Mapping[str, object]


def _session_token(session: _LiveSession, user_api_key_dict: UserAPIKeyAuth, *, stored: bool = False) -> str:
    now: Final = int(time.time())
    return encrypt_value_helper(
        json.dumps(
            {
                "v": _LIVE_SESSION_TOKEN_VERSION,
                "session_id": session.session_id,
                "model": session.model,
                "deployment_id": session.deployment_id,
                "key": user_api_key_dict.token or "",
                "expires_at": now + _LIVE_SESSION_TOKEN_TTL_SECONDS,
                **({"retained_until": now + _LIVE_SESSION_RETENTION_SECONDS} if stored else {}),
            },
            separators=(",", ":"),
        )
    )


def _owned_session(session_token: str, user_api_key_dict: UserAPIKeyAuth) -> _OwnedSession | None:
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
    retained_until: Final = payload.get("retained_until")
    if not isinstance(session_id, str) or not isinstance(model, str) or not isinstance(expires_at, int):
        return None
    if not isinstance(deployment_id, str):
        return None
    if retained_until is not None and not isinstance(retained_until, int):
        return None
    if not payload.get("key") or payload.get("key") != user_api_key_dict.token:
        return None
    return _OwnedSession(
        session=_LiveSession(session_id=session_id, model=model, deployment_id=deployment_id),
        expires_at=expires_at,
        retained_until=retained_until,
    )


def _usable_session(session_token: str, user_api_key_dict: UserAPIKeyAuth, *, retained: bool) -> _OwnedSession | None:
    owned: Final = _owned_session(session_token, user_api_key_dict)
    if owned is None:
        return None
    usable_until: Final = max(owned.expires_at, owned.retained_until or 0) if retained else owned.expires_at
    return owned if usable_until >= time.time() else None


def _attachable_session(  # pyright: ignore[reportUnusedFunction]  # the attach route in live_session_controls calls it
    session_token: str, user_api_key_dict: UserAPIKeyAuth
) -> _LiveSession | None:
    """The session behind a session token, when this caller may join it."""
    owned: Final = _usable_session(session_token, user_api_key_dict, retained=False)
    return None if owned is None else owned.session


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


def _http_url(upstream: LiveUpstream, path: str = "") -> str:
    scheme, _, rest = upstream.url.partition("://")
    return f"{_HTTP_SCHEMES.get(scheme, scheme)}://{rest}{path}"


def _session_action_path(session_id: str, action: str) -> str:
    return f"/{quote(session_id, safe='')}/{action}"


async def _post_session(upstream: LiveUpstream, body: Mapping[str, object] | None, path: str = "") -> httpx.Response:
    """POST to the provider's session API. A provider error comes back as its response, for the caller to read."""
    http_client: Final = get_async_httpx_client(llm_provider=httpxSpecialProvider.PassThroughEndpoint)
    try:
        return await http_client.post(  # pyright: ignore[reportUnknownMemberType]  # untyped dict params
            _http_url(upstream, path),
            headers={**upstream.headers, **({} if body is None else {"Content-Type": "application/json"})},
            json=None if body is None else dict(body),
        )
    except httpx.HTTPStatusError as error:
        return error.response


def _trunk_password(body: Mapping[str, object] | None) -> str | None:
    transport: Final = None if body is None else as_str_mapping(body.get("transport"))
    trunk: Final = None if transport is None else as_str_mapping(transport.get("trunk"))
    trunk_auth: Final = None if trunk is None else as_str_mapping(trunk.get("auth"))
    password: Final = None if trunk_auth is None else trunk_auth.get("password")
    return password if isinstance(password, str) and password else None


def _without_secret(content: bytes, secret: str | None) -> bytes:
    if secret is None:
        return content
    return content.replace(secret.encode(), _REDACTED_SECRET).replace(
        json.dumps(secret)[1:-1].encode(), _REDACTED_SECRET
    )


def _provider_content_type(headers: Mapping[str, str], default: str) -> str:
    return headers.get("content-type", default)


def _provider_answer(response: httpx.Response, sent_body: Mapping[str, object] | None) -> Response:
    return Response(
        content=_without_secret(response.content, _trunk_password(sent_body)),
        status_code=response.status_code,
        media_type=_provider_content_type(response.headers, "application/json"),
    )


async def _routing_data(request: Request, user_api_key_dict: UserAPIKeyAuth, model: str) -> Mapping[str, object]:
    from litellm.proxy.litellm_pre_call_utils import (
        add_litellm_data_to_request,  # pyright: ignore[reportUnknownVariableType]  # untyped dict params
    )
    from litellm.proxy.proxy_server import (
        general_settings,  # pyright: ignore[reportUnknownVariableType]  # untyped proxy settings dict
        proxy_config,
        version,
    )

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
    return {} if request_data is None else request_data


async def _unguarded_routing_data(
    request: Request, user_api_key_dict: UserAPIKeyAuth, model: str
) -> Mapping[str, object]:
    request_data: Final = await _routing_data(request, user_api_key_dict, model)
    if not request_data or live_guardrails_configured(request_data):
        raise HTTPException(
            status_code=http_status.HTTP_400_BAD_REQUEST, detail="Guardrails are not supported on OpenAI Live sessions"
        )
    return request_data


async def _resolved_deployment(deployment: Mapping[str, object] | None) -> _ResolvedDeployment | None:
    litellm_params: Final = None if deployment is None else as_str_mapping(deployment.get("litellm_params"))
    resolved: Final = None if litellm_params is None else await live_session_upstream(litellm_params)
    if deployment is None or litellm_params is None or resolved is None:
        return None
    model_info: Final = as_str_mapping(deployment.get("model_info"))
    deployment_id: Final = None if model_info is None else model_info.get("id")
    provider_model, upstream = resolved
    return _ResolvedDeployment(
        provider_model=provider_model,
        upstream=upstream,
        litellm_params=litellm_params,
        deployment_id=deployment_id if isinstance(deployment_id, str) else "",
    )


async def _selected_deployment(model: str, request_data: Mapping[str, object]) -> _ResolvedDeployment:
    from litellm.proxy.proxy_server import llm_router

    if llm_router is None:
        raise HTTPException(status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR, detail="No router is configured")
    resolved: Final = await _resolved_deployment(as_str_mapping(await llm_router.select(model, dict(request_data))))
    if resolved is None:
        raise HTTPException(
            status_code=http_status.HTTP_400_BAD_REQUEST,
            detail="Live sessions require an OpenAI or Azure deployment",
        )
    return resolved


async def _authorize_session_model(model: str, user_api_key_dict: UserAPIKeyAuth) -> None:
    from litellm.proxy.proxy_server import (
        llm_model_list,  # pyright: ignore[reportUnknownVariableType]  # untyped proxy model list
        llm_router,
    )

    if llm_router is None:
        raise HTTPException(status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR, detail="No router is configured")
    await can_key_call_resolved_model(
        model=model,
        valid_token=user_api_key_dict,
        llm_model_list=llm_model_list,  # pyright: ignore[reportUnknownArgumentType]  # untyped proxy model list
        llm_router=llm_router,
    )


async def _admit_session(
    request: Request, user_api_key_dict: UserAPIKeyAuth, session: Mapping[str, object], model: str
) -> _AdmittedSession:
    await _authorize_session_model(model, user_api_key_dict)
    await authorize_live_delegation(session, user_api_key_dict)
    request_data: Final = await _unguarded_routing_data(request, user_api_key_dict, model)
    deployment: Final = await _selected_deployment(model, request_data)
    return _AdmittedSession(
        deployment=deployment,
        provider_session=deployment_delegation(session, deployment.litellm_params, user_api_key_dict),
    )


async def _ended_call(upstream: LiveUpstream, session_id: str) -> bool:
    try:
        response: Final = await _post_session(upstream, None, _session_action_path(session_id, "hangup"))
    except Exception:  # noqa: BLE001  # any failure here must still reach the caller as a 503 that carries the token
        return False
    return response.is_success


async def _refusal_unless_metered(
    session: _LiveSession,
    upstream: LiveUpstream,
    user_api_key_dict: UserAPIKeyAuth,
    *,
    call_is_live: bool,
    stored: bool,
) -> Response | None:
    from litellm.proxy.proxy_server import (
        general_settings,  # pyright: ignore[reportUnknownVariableType]  # untyped proxy settings dict
    )

    if not webrtc_calls_are_metered(general_settings):  # pyright: ignore[reportUnknownArgumentType]  # untyped proxy settings dict
        return None
    if await _start_live_billing_sideband(session, user_api_key_dict):
        return None
    if not call_is_live:
        raise HTTPException(
            status_code=http_status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The proxy could not join this session to meter it, so the session was not started",
        )
    if await _ended_call(upstream, session.session_id):
        raise HTTPException(
            status_code=http_status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The proxy could not join this call to meter it, so the call was ended",
        )
    verbose_proxy_logger.error(
        "Live call on deployment %s has no billing and could not be ended", session.deployment_id
    )
    return JSONResponse(
        status_code=http_status.HTTP_503_SERVICE_UNAVAILABLE,
        content={
            "detail": "The proxy could not join this call to meter it or end it. End it with POST "
            "/v1/live/sessions/{session.id}/hangup",
            "session": {"id": _session_token(session, user_api_key_dict, stored=stored)},
        },
    )


def _tokenized_answer(
    upstream_response: httpx.Response,
    created: Mapping[str, object],
    created_session: Mapping[str, object],
    session: _LiveSession,
    user_api_key_dict: UserAPIKeyAuth,
    *,
    stored: bool,
) -> Response:
    return JSONResponse(
        status_code=upstream_response.status_code,
        content={
            **created,
            "session": {
                **created_session,
                "id": _session_token(session, user_api_key_dict, stored=stored),
                "model": session.model,
            },
        },
    )


@router.post("/v1/live/sessions", tags=["realtime"])
@router.post("/live/sessions", tags=["realtime"], include_in_schema=False)
@router.post("/openai/v1/live/sessions", tags=["realtime"], include_in_schema=False)
async def create_live_session(
    request: Request,
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
) -> Response:
    """
    Create a GPT-Live session over WebRTC or SIP. The body carries the session and the transport. A WebRTC
    answer carries the SDP, and a SIP answer returns before the callee picks up. The returned `session.id`
    attaches a sideband on `/v1/live/sessions/{session_id}/attach` and controls the session.
    """
    requested: Final = await _requested_session(request)
    if requested is None:
        raise HTTPException(status_code=http_status.HTTP_400_BAD_REQUEST, detail="session.model is required")
    body, session, model = requested
    transport: Final = as_str_mapping(body.get("transport"))
    if transport is None:
        raise HTTPException(status_code=http_status.HTTP_400_BAD_REQUEST, detail="transport must be an object")
    admitted: Final = await _admit_session(request, user_api_key_dict, session, model)
    deployment: Final = admitted.deployment
    upstream_response: Final = await _post_session(
        deployment.upstream, {**body, "session": {**admitted.provider_session, "model": deployment.provider_model}}
    )
    created_parts: Final = _created_session(upstream_response) if upstream_response.is_success else None
    if created_parts is None:
        return _provider_answer(upstream_response, body)
    created, created_session, upstream_session_id = created_parts
    live_session: Final = _LiveSession(
        session_id=upstream_session_id, model=model, deployment_id=deployment.deployment_id
    )
    stored: Final = session.get("store") is True
    refusal: Final = await _refusal_unless_metered(
        live_session,
        deployment.upstream,
        user_api_key_dict,
        call_is_live=transport.get("type") == "sip",
        stored=stored,
    )
    if refusal is not None:
        return refusal
    return _tokenized_answer(
        upstream_response, created, created_session, live_session, user_api_key_dict, stored=stored
    )
