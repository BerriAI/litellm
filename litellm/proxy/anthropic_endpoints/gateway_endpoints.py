"""
Claude Code gateway protocol.

Implements the wire contract the Claude Code CLI uses to talk to a gateway:
OAuth 2.0 device-authorization sign-in (RFC 8414 / RFC 8628), inference via the
Anthropic Messages API, managed settings, and OTLP telemetry ingestion. See
https://code.claude.com/docs/en/claude-apps-gateway.

Everything lives under the ``/claude_code_gateway`` base so operators point
Claude Code at ``https://<proxy-host>/claude_code_gateway`` via ``/login``. The
device flow reuses the proxy's existing SSO login machinery: the browser leg is
served by ``/sso/key/generate`` and the shared ``cli_sso_session_cache`` flow,
so the bearer token minted here is the same session JWT the LiteLLM CLI uses and
is accepted by every bearer-authenticated proxy route.
"""

import hashlib
import json
import secrets
from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, TypeAdapter

from litellm.constants import (
    CLI_JWT_EXPIRATION_HOURS,
    CLI_SSO_SESSION_TTL_SECONDS,
    LITELLM_CLI_SOURCE_IDENTIFIER,
)
from litellm.proxy.anthropic_endpoints.endpoints import anthropic_response, count_tokens
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

GATEWAY_PREFIX: Final = "/claude_code_gateway"
_DEVICE_CODE_GRANT: Final = "urn:ietf:params:oauth:grant-type:device_code"
_REFRESH_TOKEN_GRANT: Final = "refresh_token"
_DEVICE_POLL_INTERVAL_SECONDS: Final = 5
_SECONDS_PER_HOUR: Final = 3600
_MANAGED_SETTINGS_ADAPTER: Final = TypeAdapter(dict[str, object])
_NO_SETTINGS: Final = MappingProxyType({})
_POST_ONLY: Final = ["POST"]  # mutable-ok: FastAPI's add_api_route only accepts a list of methods


class _GatewaySessionData(BaseModel):
    user_id: str
    user_role: str | None = None
    models: list[str] = Field(default_factory=list)
    teams: tuple[str, ...] = ()


class _OAuthErrorBody(BaseModel):
    error: str
    error_description: str | None = None


class _AuthorizationServerMetadata(BaseModel):
    issuer: str
    device_authorization_endpoint: str
    token_endpoint: str
    grant_types_supported: tuple[str, ...]


class _DeviceAuthorizationBody(BaseModel):
    device_code: str
    user_code: str
    verification_uri: str
    verification_uri_complete: str
    expires_in: int
    interval: int


class _AccessTokenBody(BaseModel):
    access_token: str
    expires_in: int
    token_type: str = "Bearer"


def _general_settings() -> Mapping[str, object]:
    from litellm.proxy.proxy_server import general_settings

    return general_settings or _NO_SETTINGS


def _is_gateway_enabled() -> bool:
    return bool(_general_settings().get("enable_claude_code_gateway", False))


def ensure_gateway_enabled() -> None:
    from fastapi import HTTPException

    if not _is_gateway_enabled():
        raise HTTPException(status_code=404, detail="Claude Code gateway is not enabled")


def _managed_settings() -> dict[str, object] | None:
    settings: Final[object] = _general_settings().get("claude_code_gateway_managed_settings")
    if not isinstance(settings, dict):
        return None
    return _MANAGED_SETTINGS_ADAPTER.validate_python(settings)


def _oauth_error(*, status_code: int, error: str, description: str | None = None) -> "_OAuthError":
    return _OAuthError(status_code=status_code, error=error, description=description)


class _OAuthError(Exception):
    def __init__(self, *, status_code: int, error: str, description: str | None) -> None:
        self.status_code = status_code
        self.error = error
        self.description = description


def _oauth_error_response(err: _OAuthError) -> JSONResponse:
    body: Final = _OAuthErrorBody(error=err.error, error_description=err.description)
    return JSONResponse(status_code=err.status_code, content=body.model_dump(exclude_none=True))


router: Final = APIRouter(
    prefix=GATEWAY_PREFIX,
    tags=["Claude Code gateway"],  # mutable-ok: FastAPI's APIRouter only accepts a list of tags
)
_GATEWAY_ENABLED: Final = (Depends(ensure_gateway_enabled),)
_AUTHENTICATED: Final = (Depends(user_api_key_auth),)

router.add_api_route(
    "/v1/messages",
    anthropic_response,
    methods=_POST_ONLY,
    dependencies=_GATEWAY_ENABLED,
    include_in_schema=False,
)
router.add_api_route(
    "/v1/messages/count_tokens",
    count_tokens,
    methods=_POST_ONLY,
    dependencies=_GATEWAY_ENABLED,
    include_in_schema=False,
)


@router.get("/.well-known/oauth-authorization-server", include_in_schema=False)
async def oauth_authorization_server(request: Request) -> JSONResponse:
    if not _is_gateway_enabled():
        return _oauth_error_response(_oauth_error(status_code=404, error="not_found"))

    from litellm.proxy.utils import get_custom_url

    request_base_url: Final = str(request.base_url)
    metadata: Final = _AuthorizationServerMetadata(
        issuer=get_custom_url(request_base_url=request_base_url, route="claude_code_gateway"),
        device_authorization_endpoint=get_custom_url(
            request_base_url=request_base_url, route="claude_code_gateway/oauth/device_authorization"
        ),
        token_endpoint=get_custom_url(request_base_url=request_base_url, route="claude_code_gateway/oauth/token"),
        grant_types_supported=(_DEVICE_CODE_GRANT, _REFRESH_TOKEN_GRANT),
    )
    return JSONResponse(content=metadata.model_dump())


@router.post("/oauth/device_authorization", include_in_schema=False)
async def device_authorization(request: Request) -> JSONResponse:
    from urllib.parse import urlencode

    from litellm.proxy.management_endpoints.ui_sso import (
        _check_cli_sso_start_rate_limit,  # pyright: ignore[reportPrivateUsage]  # shared device-flow helper
        _generate_cli_sso_user_code,  # pyright: ignore[reportPrivateUsage]  # shared device-flow helper
        _hash_cli_sso_secret,  # pyright: ignore[reportPrivateUsage]  # shared device-flow helper
        _normalize_cli_sso_user_code,  # pyright: ignore[reportPrivateUsage]  # shared device-flow helper
        _set_cli_sso_flow,  # pyright: ignore[reportPrivateUsage]  # shared device-flow helper
    )
    from litellm.proxy.proxy_server import cli_sso_session_cache
    from litellm.proxy.utils import get_custom_url

    if not _is_gateway_enabled():
        return _oauth_error_response(_oauth_error(status_code=404, error="not_found"))

    _check_cli_sso_start_rate_limit(
        request=request,
        cache=cli_sso_session_cache,
        use_x_forwarded_for=bool(_general_settings().get("use_x_forwarded_for", False)),
    )

    device_code: Final = f"cli-{secrets.token_urlsafe(24)}"
    user_code: Final = _generate_cli_sso_user_code()
    flow: Final = {  # mutable-ok: the shared CLI SSO cache entry is a dict the browser leg mutates
        "poll_secret_hash": _hash_cli_sso_secret(device_code),
        "user_code_hash": _hash_cli_sso_secret(_normalize_cli_sso_user_code(user_code)),
        "sso_complete": False,
        "user_code_verified": False,
        "session_data": None,
    }
    _set_cli_sso_flow(login_id=device_code, cache=cli_sso_session_cache, flow=flow)

    request_base_url: Final = str(request.base_url)
    verification_uri: Final = get_custom_url(request_base_url=request_base_url, route="sso/key/generate")
    query: Final = MappingProxyType({"source": LITELLM_CLI_SOURCE_IDENTIFIER, "key": device_code})
    body: Final = _DeviceAuthorizationBody(
        device_code=device_code,
        user_code=user_code,
        verification_uri=f"{verification_uri}?{urlencode(query)}",
        verification_uri_complete=(
            f"{verification_uri}?{urlencode(MappingProxyType({**query, 'user_code': user_code}))}"
        ),
        expires_in=CLI_SSO_SESSION_TTL_SECONDS,
        interval=_DEVICE_POLL_INTERVAL_SECONDS,
    )
    return JSONResponse(content=body.model_dump())


def _mint_access_token_from_flow(flow: Mapping[str, object]) -> str:
    from litellm.proxy._types import LiteLLM_UserTable
    from litellm.proxy.auth.auth_checks import ExperimentalUIJWTToken

    raw_session_data: Final = flow.get("session_data")
    if not isinstance(raw_session_data, dict):
        raise _oauth_error(status_code=400, error="authorization_pending")

    session_data: Final = _GatewaySessionData.model_validate(raw_session_data)
    team_id: Final = session_data.teams[0] if session_data.teams else None
    user_info: Final = LiteLLM_UserTable(
        user_id=session_data.user_id,
        user_role=session_data.user_role,
        models=session_data.models,
    )
    return ExperimentalUIJWTToken.get_cli_jwt_auth_token(user_info=user_info, team_id=team_id)


async def _handle_device_code_grant(device_code: str | None) -> JSONResponse:
    from fastapi import HTTPException

    from litellm.proxy.management_endpoints.ui_sso import (
        _get_cli_sso_flow_cache_key,  # pyright: ignore[reportPrivateUsage]  # shared device-flow helper
        _get_cli_sso_flow_or_raise,  # pyright: ignore[reportPrivateUsage]  # shared device-flow helper
    )
    from litellm.proxy.proxy_server import cli_sso_session_cache

    if not device_code:
        return _oauth_error_response(
            _oauth_error(status_code=400, error="invalid_request", description="device_code is required")
        )

    try:
        flow: Final = _get_cli_sso_flow_or_raise(login_id=device_code, cache=cli_sso_session_cache)
    except HTTPException:
        return _oauth_error_response(_oauth_error(status_code=400, error="expired_token"))

    if not flow.get("sso_complete") or not flow.get("user_code_verified"):
        return _oauth_error_response(_oauth_error(status_code=400, error="authorization_pending"))

    try:
        access_token: Final = _mint_access_token_from_flow(flow)
    except _OAuthError as err:
        return _oauth_error_response(err)

    cli_sso_session_cache.delete_cache(key=_get_cli_sso_flow_cache_key(device_code))
    body: Final = _AccessTokenBody(access_token=access_token, expires_in=CLI_JWT_EXPIRATION_HOURS * _SECONDS_PER_HOUR)
    return JSONResponse(content=body.model_dump())


@router.post("/oauth/token", include_in_schema=False)
async def oauth_token(request: Request) -> JSONResponse:
    if not _is_gateway_enabled():
        return _oauth_error_response(_oauth_error(status_code=404, error="not_found"))

    form: Final = await request.form()
    grant_type: Final = form.get("grant_type")

    if grant_type == _DEVICE_CODE_GRANT:
        device_code: Final = form.get("device_code")
        return await _handle_device_code_grant(device_code if isinstance(device_code, str) else None)

    if grant_type == _REFRESH_TOKEN_GRANT:
        return _oauth_error_response(
            _oauth_error(
                status_code=401,
                error="invalid_grant",
                description="This gateway does not issue refresh tokens; sign in again",
            )
        )

    return _oauth_error_response(
        _oauth_error(
            status_code=400, error="unsupported_grant_type", description=f"Unsupported grant_type: {grant_type}"
        )
    )


@router.get("/managed/settings", include_in_schema=False, dependencies=_AUTHENTICATED)
async def managed_settings(request: Request) -> Response:
    ensure_gateway_enabled()

    settings: Final = _managed_settings()
    if settings is None:
        return Response(status_code=404)

    body: Final = json.dumps(settings, sort_keys=True, separators=(",", ":"))
    etag: Final = '"' + hashlib.sha256(body.encode("utf-8")).hexdigest() + '"'
    headers: Final = MappingProxyType({"ETag": etag})
    if request.headers.get("If-None-Match") == etag:
        return Response(status_code=304, headers=headers)
    return Response(content=body, media_type="application/json", headers=headers)


def _accept_otlp() -> Response:
    ensure_gateway_enabled()
    return Response(status_code=200)


@router.post("/v1/metrics", include_in_schema=False, dependencies=_AUTHENTICATED)
async def otlp_metrics() -> Response:
    return _accept_otlp()


@router.post("/v1/logs", include_in_schema=False, dependencies=_AUTHENTICATED)
async def otlp_logs() -> Response:
    return _accept_otlp()


@router.post("/v1/traces", include_in_schema=False, dependencies=_AUTHENTICATED)
async def otlp_traces() -> Response:
    return _accept_otlp()
