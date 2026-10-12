import os
import time
from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial
from typing import Annotated, Final
from urllib.parse import quote

import httpx
import jwt
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, ConfigDict
from starlette.responses import JSONResponse

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.authorization import OwnedRows, resolve_trace_read_scope
from litellm.proxy.auth.authorization_dependencies import LogTeamLookupDependency
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.lens.buffering import (
    FORWARD_BUFFER_BUDGET,
    AdmittedBody,
    BodyReservation,
    BufferBudget,
    BufferedResponse,
    LensRoute,
    request_body,
)
from litellm.tracing.remote import MAX_RESPONSE_BYTES, LensConnection, bounded_response

router: Final = APIRouter(prefix="/lens", tags=["Lens"], route_class=LensRoute)
PUBLIC_CONTRACT: Final = 1
_REQUEST_HEADERS: Final = frozenset({"content-type", "x-lens-contract", "idempotency-key"})
_RESPONSE_HEADERS: Final = frozenset({"content-type", "content-disposition", "retry-after", "x-lens-contract"})


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
        subject: Final = identity.user_id or identity.token
        if not subject:
            return None
        return jwt.encode(
            {
                "iss": "litellm",
                "aud": "litellm-lens",
                "sub": subject,
                "iat": now,
                "exp": now + 30,
                "identity": identity.model_dump(mode="json"),
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


@router.get("/service", response_model=ServiceConnection)
async def service_connection(
    identity: Annotated[Identity, Depends(delegated_identity)],
) -> ServiceConnection:
    connection: Final = Connection.from_env()
    status: Final = (
        await service_status(connection, connection.remote.control_client()) if connection else ServiceStatus()
    )
    return ServiceConnection(
        url=os.environ.get("LITELLM_LENS_PUBLIC_URL", "").rstrip("/"),
        connected=bool(connection) and status.public_contract == PUBLIC_CONTRACT,
        status=status,
        configured=bool(os.environ.get("LITELLM_LENS_URL")),
        release=status.release,
    )


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
                headers={name: value for name, value in response.headers.items() if name in _RESPONSE_HEADERS},
                reservation=reservation,
            )
    except (httpx.HTTPError, RuntimeError) as error:
        reservation.release()
        raise HTTPException(503, "Lens service is unavailable") from error
    except BaseException:
        reservation.release()
        raise


@router.api_route("", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
@router.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
async def lens_request(
    request: Request,
    identity: Annotated[Identity, Depends(delegated_identity)],
    path: str = "",
) -> Response:
    connection: Final = Connection.from_env()
    if connection is None:
        raise HTTPException(503, "Configure LITELLM_LENS_URL and LENS_GATEWAY_SECRET for the Lens service")
    return await forward(request, identity, path, connection, connection.remote.control_client(), int(time.time()))
