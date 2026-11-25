import os
import time
from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial
from typing import Annotated, Final
from urllib.parse import urlencode, urlsplit

import httpx
import jwt
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import AnyHttpUrl, BaseModel, ConfigDict
from starlette.responses import RedirectResponse

from litellm.constants import LITELLM_PROXY_ADMIN_NAME
from litellm.integrations.lens.config import LensController
from litellm.integrations.lens.proxy.buffering import (
    LensRoute,
)
from litellm.integrations.lens.remote import LensConnection, bounded_response
from litellm.litellm_core_utils.url_utils import is_local_host
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.authorization import OwnedRows, resolve_trace_read_scope
from litellm.proxy.auth.authorization_dependencies import LogTeamLookupDependency
from litellm.proxy.auth.route_checks import RouteChecks
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth, user_api_key_auth_builder

router: Final = APIRouter(prefix="/lens", tags=["Lens"], route_class=LensRoute)
PUBLIC_CONTRACT: Final = 1
_REQUEST_AUDIENCE: Final = "litellm-lens"
_LOGIN_AUDIENCE: Final = "litellm-lens-login"
_LOGIN_TOKEN_SECONDS: Final = 60
_PASSWORD_CHANGE_ROUTE: Final = "/user/password/change"
_LOGIN_UNAVAILABLE: Final = (
    "Configure LITELLM_LENS_URL, LENS_GATEWAY_SECRET and LITELLM_LENS_PUBLIC_URL to sign in to Lens"
)


class Identity(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    user_role: LitellmUserRoles
    user_id: str | None
    team_id: str | None
    org_id: str | None
    token: str | None
    models: tuple[str, ...]
    log_team_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True, repr=False)
class Connection:
    remote: LensConnection
    secret: str

    @classmethod
    def from_env(cls, environ: Mapping[str, str] = os.environ) -> "Connection | None":
        secret: Final = environ.get("LENS_GATEWAY_SECRET", "")
        if not 32 <= len(secret.encode()) <= 512:
            return None
        try:
            return cls(LensConnection.from_env(environ), secret)
        except ValueError:
            return None

    def identity_token(self, identity: Identity, now: int) -> str | None:
        return self._signed(identity, now, _REQUEST_AUDIENCE, 30)

    def login_token(self, identity: Identity, user_name: str | None, now: int) -> str | None:
        return self._signed(identity, now, _LOGIN_AUDIENCE, _LOGIN_TOKEN_SECONDS, user_name)

    def _signed(
        self, identity: Identity, now: int, audience: str, lifetime: int, user_name: str | None = None
    ) -> str | None:
        subject: Final = identity.user_id or identity.token
        if not subject:
            return None
        return jwt.encode(
            {
                "iss": "litellm",
                "aud": audience,
                "sub": subject,
                "iat": now,
                "exp": now + lifetime,
                "identity": identity.model_dump(mode="json") | ({"user_name": user_name} if user_name else {}),
            },
            self.secret,
            algorithm="HS256",
        )


async def delegated_identity(
    auth: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
    log_team_lookup: LogTeamLookupDependency,
) -> Identity:
    scope: Final = await resolve_trace_read_scope(auth, partial(log_team_lookup, auth))
    return Identity.model_validate(
        {
            **auth.model_dump(include={"user_id", "team_id", "org_id", "token", "models"}),
            "user_role": auth.user_role or LitellmUserRoles.INTERNAL_USER,
            "log_team_ids": scope.team_ids if isinstance(scope, OwnedRows) else (),
        }
    )


class ServiceStatus(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    storage_ready: bool = False
    credentials_ready: bool = False
    release: str = ""
    protocol_version: int = 0
    public_contract: int = 0


class ServiceConnection(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    url: str
    connected: bool
    status: ServiceStatus
    configured: bool = False
    release: str = ""
    browser_url: str = ""


def lens_browser_url(lens_url: str, gateway_url: str) -> str:
    try:
        target: Final = AnyHttpUrl(lens_url)
        gateway: Final = AnyHttpUrl(gateway_url) if gateway_url else None
        if target.username is not None or target.password is not None:
            return ""
        if target.query or target.fragment:
            return ""
        if is_local_host(target.host or "") and not is_local_host((gateway.host or "") if gateway else ""):
            return ""
        return lens_url.rstrip("/")
    except ValueError:
        return ""


class _GatewaySettings(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    general_settings: Mapping[str, object]


async def service_status(connection: Connection, client: httpx.AsyncClient) -> ServiceStatus:
    try:
        async with client.stream(
            "GET", connection.remote.endpoint("/internal/status"), headers=connection.remote.headers, timeout=2
        ) as response:
            if response.status_code == 200:
                return ServiceStatus.model_validate_json(await bounded_response(response, 16 * 1024))
    except (ValueError, RuntimeError, httpx.HTTPError):
        pass
    return ServiceStatus()


async def service_connection(identity: Identity, gateway_url: str = "") -> ServiceConnection:
    from litellm.proxy import proxy_server

    settings: Final = _GatewaySettings.model_validate(vars(proxy_server)).general_settings
    connection: Final = Connection.from_env()
    status: Final = (
        await service_status(connection, connection.remote.control_client()) if connection else ServiceStatus()
    )
    return ServiceConnection(
        url=os.environ.get("LITELLM_LENS_PUBLIC_URL", "").rstrip("/"),
        connected=bool(connection) and status.public_contract == PUBLIC_CONTRACT,
        status=status,
        configured=LensController.is_lens_setup(settings.get("tracing")),
        release=status.release,
        browser_url=lens_browser_url(os.environ.get("LITELLM_LENS_PUBLIC_URL", ""), gateway_url),
    )


async def browser_session(request: Request) -> UserAPIKeyAuth | None:
    from litellm.proxy.mcp_server.byok_oauth_endpoints import (  # noqa: PLC0415  # circular import at module load
        session_identity_from_cookie,
    )

    user_id, session_key = session_identity_from_cookie(request)
    if not user_id or not session_key:
        return None
    try:
        auth: Final = await user_api_key_auth_builder(
            request=request,
            api_key=f"Bearer {session_key}",
            azure_api_key_header="",
            anthropic_api_key_header=None,
            google_ai_studio_api_key_header=None,
            azure_apim_header=None,
            request_data={},
        )
    except Exception:  # noqa: BLE001  # any rejected dashboard session means the browser signs in again
        return None
    return auth if auth.user_id == user_id else None


def login_user_name(auth: UserAPIKeyAuth, admin_user_name: str) -> str | None:
    if auth.user_email:
        return auth.user_email
    return admin_user_name if auth.user_id == LITELLM_PROXY_ADMIN_NAME else None


def sign_in_redirect(sign_in_url: str) -> RedirectResponse:
    # The dashboard login page sends any browser holding an unexpired `token` cookie straight back
    # here, so a session this route refused has to be dropped or the two pages redirect forever
    response: Final = RedirectResponse(sign_in_url, status_code=303)
    response.delete_cookie("token", path="/")
    response.delete_cookie("token", path=urlsplit(sign_in_url).path.removesuffix("/login"))
    return response


def login_redirect(
    connection: Connection | None,
    lens_url: str,
    identity: Identity | None,
    user_name: str | None,
    sign_in_url: str,
    now: int,
) -> Response:
    if connection is None or not lens_url:
        raise HTTPException(503, _LOGIN_UNAVAILABLE)
    if identity is None:
        return sign_in_redirect(sign_in_url)
    token: Final = connection.login_token(identity, user_name, now)
    if token is None:
        raise HTTPException(403, "Lens requires an authenticated user or key")
    return RedirectResponse(
        f"{lens_url}/ui/#{urlencode({'token': token})}",
        status_code=303,
        headers={"cache-control": "no-store", "referrer-policy": "no-referrer"},
    )


def password_change_redirect(auth: UserAPIKeyAuth | None, route: str, base_url: str) -> Response | None:
    if auth is None:
        return None
    if auth.user_role != LitellmUserRoles.PROXY_ADMIN:
        RouteChecks.custom_admin_only_route_check(route=route)
    if RouteChecks.key_allowed_routes_grant(route=route, valid_token=auth):
        return None
    if RouteChecks.key_allowed_routes_grant(route=_PASSWORD_CHANGE_ROUTE, valid_token=auth):
        return RedirectResponse(f"{base_url}/ui/change-password", status_code=303)
    raise HTTPException(403, "This session is not allowed to sign in to Lens")


@router.get("/login", include_in_schema=False)
@router.get("/login/", include_in_schema=False)
async def lens_login(request: Request, log_team_lookup: LogTeamLookupDependency) -> Response:
    from litellm.proxy.mcp_server.oauth_utils import (  # noqa: PLC0415  # circular import at module load
        get_request_base_url,
    )

    auth: Final = await browser_session(request)
    route: Final = request.url.path.rstrip("/")
    base_url: Final = get_request_base_url(request)
    blocked: Final = password_change_redirect(auth, route, base_url)
    if blocked is not None:
        return blocked
    return login_redirect(
        Connection.from_env(),
        os.environ.get("LITELLM_LENS_PUBLIC_URL", "").rstrip("/"),
        await delegated_identity(auth, log_team_lookup) if auth else None,
        login_user_name(auth, os.environ.get("UI_USERNAME", "admin")) if auth else None,
        f"{base_url}/ui/login?{urlencode({'redirect_to': route})}",
        int(time.time()),
    )
