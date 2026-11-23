"""What a browser loads from Lens through the gateway: the Lens UI and the APIs its pages call.

Everything here is passed through to the Lens this gateway is connected to, so the UI and its page APIs follow that
Lens's version and nothing about them is pinned in LiteLLM. Sending traces and spend to Lens is a different concern
and lives in `litellm.integrations.lens.exporter` and `tracing_endpoints`; none of it is reachable from these routes.
"""

import time
from collections.abc import Awaitable, Callable
from types import MappingProxyType
from typing import Final
from urllib.parse import quote

import httpx
from fastapi import APIRouter, HTTPException, Request, Response
from starlette.responses import JSONResponse, RedirectResponse

from litellm.integrations.lens.proxy.adapter import (
    PUBLIC_CONTRACT,
    Connection,
    Identity,
    ServiceConnection,
    delegated_identity,
    service_connection,
)
from litellm.integrations.lens.proxy.buffering import (
    FORWARD_BUFFER_BUDGET,
    AdmittedBody,
    BodyReservation,
    BufferBudget,
    BufferedResponse,
    LensRoute,
    request_body,
)
from litellm.integrations.lens.remote import MAX_RESPONSE_BYTES, LensConnection, bounded_response
from litellm.proxy._types import LitellmUserRoles, SpecialHeaders, UserAPIKeyAuth
from litellm.proxy.auth.authorization_dependencies import LogTeamLookupDependency
from litellm.proxy.auth.ip_address_utils import IPAddressUtils
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

router: Final = APIRouter(prefix="/lens", tags=["Lens"], route_class=LensRoute)

_REQUEST_HEADERS: Final = frozenset({"content-type", "x-lens-contract", "idempotency-key"})
_RESPONSE_HEADERS: Final = frozenset({"content-type", "content-disposition", "retry-after", "x-lens-contract"})
_PRIVATE: Final = MappingProxyType({"cache-control": "private, no-store"})
_UI_MOUNT_HEADER: Final = "x-lens-ui-mount"
_UI_RESPONSE_HEADERS: Final = frozenset({"content-type", "cache-control", "x-content-type-options"})
_UI_FRAME: Final = "frame-ancestors 'self'"


def _framed_policy(upstream: str | None) -> str:
    """Lens's own content policy with its framing rule replaced by one that lets only the dashboard frame it."""
    kept: Final = tuple(
        directive.strip()
        for directive in (upstream or "").split(";")
        if directive.strip() and not directive.strip().lower().startswith("frame-ancestors")
    )
    return "; ".join((*kept, _UI_FRAME))


_UI_UNAVAILABLE: Final = "The connected Lens does not serve a UI this gateway can show. Update Lens to use it here"
_KEY_HEADERS: Final = (
    SpecialHeaders.openai_authorization.value,
    SpecialHeaders.custom_litellm_api_key.value,
)
_READ_METHODS: Final = frozenset({"GET", "HEAD", "OPTIONS"})
_ADMIN_PREFIX: Final = "admin"
_ADMIN_ROLES: Final = frozenset({LitellmUserRoles.PROXY_ADMIN, LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY})


def ui_mount(root_path: str) -> str:
    return root_path.rstrip("/") + "/lens/ui"


def _leaves_ui(path: str) -> bool:
    return "\\" in path or any(part in (".", "..") for part in path.split("/"))


async def lens_ui_asset(
    path: str,
    remote: LensConnection | None,
    client: httpx.AsyncClient | None,
    mount: str,
    *,
    budget: BufferBudget = FORWARD_BUFFER_BUDGET,
) -> Response:
    """Lens serves its one UI build at `/ui/` and rewrites its addresses to the mount it is asked for.

    A Lens that predates that ignores the header and would hand back a page whose assets point at the gateway's own
    `/ui`, so only a response that names the mount back is shown.
    """
    if remote is None or client is None or _leaves_ui(path):
        raise HTTPException(404, _UI_UNAVAILABLE)
    reservation: Final = BodyReservation(budget)
    try:
        async with client.stream(
            "GET", remote.endpoint("/ui/" + quote(path, safe="/")), headers={_UI_MOUNT_HEADER: mount}
        ) as response:
            if response.status_code == 404 or (
                response.status_code == 200 and response.headers.get(_UI_MOUNT_HEADER) != mount
            ):
                raise HTTPException(404, _UI_UNAVAILABLE)
            if response.status_code != 200:
                raise HTTPException(503, "Lens service is unavailable")
            return BufferedResponse(
                await bounded_response(response, MAX_RESPONSE_BYTES, reserve=reservation.reserve),
                status_code=200,
                headers={
                    **{name: value for name, value in response.headers.items() if name in _UI_RESPONSE_HEADERS},
                    "x-frame-options": "SAMEORIGIN",
                    "content-security-policy": _framed_policy(response.headers.get("content-security-policy")),
                },
                reservation=reservation,
            )
    except (httpx.HTTPError, RuntimeError) as error:
        reservation.release()
        raise HTTPException(503, "Lens service is unavailable") from error
    except BaseException:
        reservation.release()
        raise


def _remote() -> LensConnection | None:
    try:
        return LensConnection.from_env()
    except ValueError:
        return None


@router.get("/ui", include_in_schema=False)
async def lens_ui_entry(request: Request) -> Response:
    return RedirectResponse(str(request.url.replace(path=request.url.path + "/")), status_code=307)


@router.get("/ui/{path:path}", include_in_schema=False)
async def lens_ui(path: str = "") -> Response:
    from litellm.proxy.utils import get_server_root_path  # noqa: PLC0415  # circular import at module load

    remote: Final = _remote()
    client: Final = remote.control_client() if remote else None
    return await lens_ui_asset(path, remote, client, ui_mount(get_server_root_path()))


def same_origin(request: Request) -> bool:
    """Whether a browser says this request came from a page on the gateway's own address."""
    site: Final = request.headers.get("sec-fetch-site")
    if site is not None:
        return site == "same-origin"
    origin: Final = request.headers.get("origin")
    scheme: Final = "https" if IPAddressUtils.is_request_https(request) else "http"
    return origin is not None and origin == f"{scheme}://{request.headers.get('host')}"


async def _key_auth(request: Request, key: str) -> UserAPIKeyAuth:
    return await user_api_key_auth(
        request=request,
        api_key=key,
        azure_api_key_header="",
        anthropic_api_key_header=None,
        google_ai_studio_api_key_header=None,
        azure_apim_header=None,
        custom_litellm_key_header=request.headers.get(SpecialHeaders.custom_litellm_api_key.value),
    )


def _cookie_identity(request: Request) -> tuple[str | None, str | None]:
    from litellm.proxy.mcp_server.byok_oauth_endpoints import (  # noqa: PLC0415  # circular import at module load
        session_identity_from_cookie,
    )

    return session_identity_from_cookie(request)


async def dashboard_session(
    request: Request,
    *,
    cookie: Callable[[Request], tuple[str | None, str | None]] = _cookie_identity,
    key: Callable[[Request, str], Awaitable[UserAPIKeyAuth]] = _key_auth,
) -> UserAPIKeyAuth | None:
    """The dashboard login in the browser's cookie, held to the same route rules as that login sent as a key."""
    user_id, session_key = cookie(request)
    if not user_id or not session_key:
        return None
    try:
        auth: Final = await key(request, f"Bearer {session_key}")
    except Exception:  # noqa: BLE001  # any rejected dashboard session means the browser signs in again
        return None
    return auth if auth.user_id == user_id else None


async def page_auth(
    request: Request,
    *,
    session: Callable[[Request], Awaitable[UserAPIKeyAuth | None]] = dashboard_session,
    key: Callable[[Request, str], Awaitable[UserAPIKeyAuth]] = _key_auth,
) -> UserAPIKeyAuth:
    """A key when the caller sends one (the CLI, MCP clients), otherwise the dashboard login the browser holds."""
    if any(name in request.headers for name in _KEY_HEADERS):
        return await key(request, request.headers.get(SpecialHeaders.openai_authorization.value, ""))
    auth: Final = await session(request)
    if auth is None:
        raise HTTPException(401, "Sign in to LiteLLM to use Lens")
    if request.method not in _READ_METHODS and not same_origin(request):
        raise HTTPException(403, "Lens changes must come from the LiteLLM dashboard")
    return auth


async def forward(
    request: Request,
    identity: Identity,
    path: str,
    connection: Connection | None,
    client: httpx.AsyncClient,
    now: int,
    *,
    budget: BufferBudget = FORWARD_BUFFER_BUDGET,
) -> Response:
    if connection is None:
        raise HTTPException(503, "Configure LITELLM_LENS_URL and LENS_GATEWAY_SECRET for the Lens service")
    if path.endswith("/"):
        return Response(
            status_code=307, headers={"location": str(request.url.replace(path=request.url.path.rstrip("/")))}
        )
    if any(part in (".", "..") for part in path.split("/")) or "\\" in path:
        raise HTTPException(400, "Invalid Lens path")
    if len(request.headers.getlist("x-lens-contract")) > 1:
        return JSONResponse({"detail": "contract_version", "code": "contract_version"}, status_code=409)
    token: Final = connection.identity_token(identity, now)
    if token is None:
        raise HTTPException(403, "Lens requires an authenticated user or key")
    headers: Final = {
        "x-lens-contract": str(PUBLIC_CONTRACT),
        **{name: value for name, value in request.headers.items() if name in _REQUEST_HEADERS},
        "authorization": f"Bearer {token}",
        "accept": "application/json",
    }
    endpoint: Final = connection.remote.endpoint("/lens" + ("/" + quote(path, safe="/") if path else ""))
    reservation: Final = BodyReservation(budget)
    try:
        async with client.stream(
            request.method,
            endpoint,
            params=tuple(request.query_params.multi_items()),
            content=request.receive.body
            if isinstance(request.receive, AdmittedBody)
            else await request_body(request, reservation=reservation),
            headers=headers,
        ) as response:
            body: Final = await bounded_response(response, MAX_RESPONSE_BYTES, reserve=reservation.reserve)
            return BufferedResponse(
                body,
                status_code=response.status_code,
                headers={
                    **{name: value for name, value in response.headers.items() if name in _RESPONSE_HEADERS},
                    **_PRIVATE,
                },
                reservation=reservation,
            )
    except (httpx.HTTPError, RuntimeError) as error:
        reservation.release()
        raise HTTPException(503, "Lens service is unavailable") from error
    except BaseException:
        reservation.release()
        raise


@router.get("/service", response_model=ServiceConnection)
async def lens_service(request: Request, log_team_lookup: LogTeamLookupDependency) -> ServiceConnection:
    return await service_connection(
        await delegated_identity(await page_auth(request), log_team_lookup), str(request.base_url)
    )


def admin_page(path: str) -> bool:
    """Whether a path under `/lens/` is one Lens keeps for admins: everything under its `admin/` prefix.

    Lens decides which of its page APIs go under that prefix, so a Lens release can add or move one with no change here.
    """
    return path == _ADMIN_PREFIX or path.startswith(f"{_ADMIN_PREFIX}/")


def allow_page(path: str, auth: UserAPIKeyAuth) -> None:
    """Stop anyone but a LiteLLM admin or admin viewer before the request reaches Lens's admin pages.

    Lens still tells the two apart: several of its admin reads are POSTs, so it refuses a viewer's writes itself.
    """
    if admin_page(path) and auth.user_role not in _ADMIN_ROLES:
        raise HTTPException(403, "Only LiteLLM admins can use this Lens page")


@router.api_route("", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
@router.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
async def lens_request(request: Request, log_team_lookup: LogTeamLookupDependency, path: str = "") -> Response:
    connection: Final = Connection.from_env()
    if connection is None:
        raise HTTPException(503, "Configure LITELLM_LENS_URL and LENS_GATEWAY_SECRET for the Lens service")
    auth: Final = await page_auth(request)
    allow_page(path, auth)
    identity: Final = await delegated_identity(auth, log_team_lookup)
    return await forward(request, identity, path, connection, connection.remote.control_client(), int(time.time()))
