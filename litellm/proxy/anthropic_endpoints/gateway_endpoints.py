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
from typing import Any

from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import JSONResponse

from litellm.constants import (
    CLI_JWT_EXPIRATION_HOURS,
    CLI_SSO_SESSION_TTL_SECONDS,
    LITELLM_CLI_SOURCE_IDENTIFIER,
)
from litellm.proxy.anthropic_endpoints.endpoints import anthropic_response, count_tokens
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

GATEWAY_PREFIX = "/claude_code_gateway"
_DEVICE_CODE_GRANT = "urn:ietf:params:oauth:grant-type:device_code"
_REFRESH_TOKEN_GRANT = "refresh_token"
_DEVICE_POLL_INTERVAL_SECONDS = 5


def _is_gateway_enabled() -> bool:
    from litellm.proxy.proxy_server import general_settings

    return bool((general_settings or {}).get("enable_claude_code_gateway", False))


def ensure_gateway_enabled() -> None:
    from fastapi import HTTPException

    if not _is_gateway_enabled():
        raise HTTPException(status_code=404, detail="Claude Code gateway is not enabled")


def _managed_settings() -> dict[str, Any] | None:
    from litellm.proxy.proxy_server import general_settings

    settings = (general_settings or {}).get("claude_code_gateway_managed_settings")
    return settings if isinstance(settings, dict) else None


def _oauth_error(*, status_code: int, error: str, description: str | None = None) -> "_OAuthError":
    return _OAuthError(status_code=status_code, error=error, description=description)


class _OAuthError(Exception):
    def __init__(self, *, status_code: int, error: str, description: str | None) -> None:
        self.status_code = status_code
        self.error = error
        self.description = description


def _oauth_error_response(err: _OAuthError) -> JSONResponse:
    body: dict[str, str] = {"error": err.error}
    if err.description is not None:
        body["error_description"] = err.description
    return JSONResponse(status_code=err.status_code, content=body)


router = APIRouter(prefix=GATEWAY_PREFIX, tags=["Claude Code gateway"])

router.add_api_route(
    "/v1/messages",
    anthropic_response,
    methods=["POST"],
    dependencies=[Depends(ensure_gateway_enabled)],
    include_in_schema=False,
)
router.add_api_route(
    "/v1/messages/count_tokens",
    count_tokens,
    methods=["POST"],
    dependencies=[Depends(ensure_gateway_enabled)],
    include_in_schema=False,
)


@router.get("/.well-known/oauth-authorization-server", include_in_schema=False)
async def oauth_authorization_server(request: Request) -> JSONResponse:
    if not _is_gateway_enabled():
        return _oauth_error_response(_oauth_error(status_code=404, error="not_found"))

    from litellm.proxy.utils import get_custom_url

    request_base_url = str(request.base_url)
    issuer = get_custom_url(request_base_url=request_base_url, route="claude_code_gateway")
    return JSONResponse(
        content={
            "issuer": issuer,
            "device_authorization_endpoint": get_custom_url(
                request_base_url=request_base_url, route="claude_code_gateway/oauth/device_authorization"
            ),
            "token_endpoint": get_custom_url(
                request_base_url=request_base_url, route="claude_code_gateway/oauth/token"
            ),
            "grant_types_supported": [_DEVICE_CODE_GRANT, _REFRESH_TOKEN_GRANT],
        }
    )


@router.post("/oauth/device_authorization", include_in_schema=False)
async def device_authorization(request: Request) -> JSONResponse:
    from urllib.parse import urlencode

    from litellm.proxy.management_endpoints.ui_sso import (
        _check_cli_sso_start_rate_limit,
        _generate_cli_sso_user_code,
        _hash_cli_sso_secret,
        _normalize_cli_sso_user_code,
        _set_cli_sso_flow,
    )
    from litellm.proxy.proxy_server import cli_sso_session_cache, general_settings
    from litellm.proxy.utils import get_custom_url

    if not _is_gateway_enabled():
        return _oauth_error_response(_oauth_error(status_code=404, error="not_found"))

    _check_cli_sso_start_rate_limit(
        request=request,
        cache=cli_sso_session_cache,
        use_x_forwarded_for=bool((general_settings or {}).get("use_x_forwarded_for", False)),
    )

    device_code = f"cli-{secrets.token_urlsafe(24)}"
    user_code = _generate_cli_sso_user_code()
    flow = {
        "poll_secret_hash": _hash_cli_sso_secret(device_code),
        "user_code_hash": _hash_cli_sso_secret(_normalize_cli_sso_user_code(user_code)),
        "sso_complete": False,
        "user_code_verified": False,
        "session_data": None,
    }
    _set_cli_sso_flow(login_id=device_code, cache=cli_sso_session_cache, flow=flow)

    request_base_url = str(request.base_url)
    verification_uri = get_custom_url(request_base_url=request_base_url, route="sso/key/generate")
    verification_uri_complete = (
        verification_uri
        + "?"
        + urlencode({"source": LITELLM_CLI_SOURCE_IDENTIFIER, "key": device_code, "user_code": user_code})
    )
    verification_uri_no_code = (
        verification_uri + "?" + urlencode({"source": LITELLM_CLI_SOURCE_IDENTIFIER, "key": device_code})
    )
    return JSONResponse(
        content={
            "device_code": device_code,
            "user_code": user_code,
            "verification_uri": verification_uri_no_code,
            "verification_uri_complete": verification_uri_complete,
            "expires_in": CLI_SSO_SESSION_TTL_SECONDS,
            "interval": _DEVICE_POLL_INTERVAL_SECONDS,
        }
    )


def _mint_access_token_from_flow(flow: dict[str, Any]) -> str:
    from litellm.proxy._types import LiteLLM_UserTable
    from litellm.proxy.auth.auth_checks import ExperimentalUIJWTToken

    session_data = flow.get("session_data")
    if not isinstance(session_data, dict):
        raise _oauth_error(status_code=400, error="authorization_pending")

    teams = session_data.get("teams") or []
    team_id = teams[0] if isinstance(teams, list) and teams else None
    user_info = LiteLLM_UserTable(
        user_id=session_data["user_id"],
        user_role=session_data["user_role"],
        models=session_data.get("models", []),
    )
    return ExperimentalUIJWTToken.get_cli_jwt_auth_token(user_info=user_info, team_id=team_id)


async def _handle_device_code_grant(device_code: str | None) -> JSONResponse:
    from fastapi import HTTPException

    from litellm.proxy.management_endpoints.ui_sso import (
        _get_cli_sso_flow_cache_key,
        _get_cli_sso_flow_or_raise,
    )
    from litellm.proxy.proxy_server import cli_sso_session_cache

    if not device_code:
        return _oauth_error_response(
            _oauth_error(status_code=400, error="invalid_request", description="device_code is required")
        )

    try:
        flow = _get_cli_sso_flow_or_raise(login_id=device_code, cache=cli_sso_session_cache)
    except HTTPException:
        return _oauth_error_response(_oauth_error(status_code=400, error="expired_token"))

    if not flow.get("sso_complete") or not flow.get("user_code_verified"):
        return _oauth_error_response(_oauth_error(status_code=400, error="authorization_pending"))

    try:
        access_token = _mint_access_token_from_flow(flow)
    except _OAuthError as err:
        return _oauth_error_response(err)

    cli_sso_session_cache.delete_cache(key=_get_cli_sso_flow_cache_key(device_code))
    return JSONResponse(
        content={
            "access_token": access_token,
            "token_type": "Bearer",
            "expires_in": CLI_JWT_EXPIRATION_HOURS * 3600,
        }
    )


@router.post("/oauth/token", include_in_schema=False)
async def oauth_token(request: Request) -> JSONResponse:
    if not _is_gateway_enabled():
        return _oauth_error_response(_oauth_error(status_code=404, error="not_found"))

    form = await request.form()
    grant_type = form.get("grant_type")

    if grant_type == _DEVICE_CODE_GRANT:
        device_code = form.get("device_code")
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


@router.get("/managed/settings", include_in_schema=False, dependencies=[Depends(user_api_key_auth)])
async def managed_settings(request: Request) -> Response:
    ensure_gateway_enabled()

    settings = _managed_settings()
    if settings is None:
        return Response(status_code=404)

    body = json.dumps(settings, sort_keys=True, separators=(",", ":"))
    etag = '"' + hashlib.sha256(body.encode("utf-8")).hexdigest() + '"'
    if_none_match = request.headers.get("If-None-Match")
    if if_none_match is not None and if_none_match == etag:
        return Response(status_code=304, headers={"ETag": etag})
    return Response(content=body, media_type="application/json", headers={"ETag": etag})


def _accept_otlp() -> Response:
    ensure_gateway_enabled()
    return Response(status_code=200)


@router.post("/v1/metrics", include_in_schema=False, dependencies=[Depends(user_api_key_auth)])
async def otlp_metrics() -> Response:
    return _accept_otlp()


@router.post("/v1/logs", include_in_schema=False, dependencies=[Depends(user_api_key_auth)])
async def otlp_logs() -> Response:
    return _accept_otlp()


@router.post("/v1/traces", include_in_schema=False, dependencies=[Depends(user_api_key_auth)])
async def otlp_traces() -> Response:
    return _accept_otlp()
