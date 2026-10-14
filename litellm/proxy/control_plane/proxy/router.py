"""Authenticated management passthrough: one allowlisted dataplane route, as the caller, audited."""

from collections.abc import Mapping
from typing import Annotated, Final

from fastapi import APIRouter, Depends, Request, Security
from fastapi.responses import JSONResponse
from pydantic import TypeAdapter, ValidationError

from litellm.proxy.control_plane.audit.dependencies import get_audit_log
from litellm.proxy.control_plane.audit.event import redact_path
from litellm.proxy.control_plane.audit.log import AuditLog, Reservation
from litellm.proxy.control_plane.auth.dependencies import require_scopes
from litellm.proxy.control_plane.auth.rbac import Principal, Scope
from litellm.proxy.control_plane.common.exceptions import ControlPlaneException, bad_gateway, forbidden, problem

AUDIT_UNAVAILABLE: Final = "audit-unavailable"


def audit_unavailable() -> ControlPlaneException:
    return ControlPlaneException(
        problem(AUDIT_UNAVAILABLE, "Audit Unavailable", 503, "No audit sink can record this write right now.")
    )


from litellm.proxy.control_plane.dataplanes.client import (
    DataplaneClient,
    DataplaneReply,
    DataplaneUnreachable,
    Method,
)
from litellm.proxy.control_plane.dataplanes.credentials import Caller, CredentialUnavailable
from litellm.proxy.control_plane.dataplanes.dependencies import dataplane_access, get_caller, get_dataplane_client
from litellm.proxy.control_plane.dataplanes.registry import Dataplane
from litellm.proxy.control_plane.proxy.allowlist import Denied, decide

DATAPLANE_HEADER: Final = "X-LiteLLM-Dataplane"
READ_METHODS: Final = ("GET",)
WRITE_METHODS: Final = ("POST", "PUT", "PATCH", "DELETE")

router: Final = APIRouter(tags=["proxy"])

_BODY: Final = TypeAdapter(Mapping[str, object])
_METHOD: Final[TypeAdapter[Method]] = TypeAdapter(Method)


def bad_request(detail: str) -> ControlPlaneException:
    return ControlPlaneException(problem("bad-request", "Bad Request", 400, detail))


async def _json_body(request: Request) -> Mapping[str, object] | None:
    raw: Final = await request.body()
    if not raw:
        return None
    try:
        return _BODY.validate_json(raw)
    except ValidationError:
        raise bad_request("Request body must be a JSON object.") from None


async def _forward(
    request: Request,
    path: str,
    principal: Principal,
    dataplane: Dataplane,
    caller: Caller,
    client: DataplaneClient,
    audit: AuditLog,
) -> JSONResponse:
    method: Final = _METHOD.validate_python(request.method)
    reservation: Final = audit.reserve() if method in WRITE_METHODS else None
    if method in WRITE_METHODS and reservation is None:
        raise audit_unavailable()
    try:
        return await _forward_reserved(request, path, method, principal, dataplane, caller, client, audit, reservation)
    finally:
        if reservation is not None:
            reservation.release()


async def _forward_reserved(
    request: Request,
    path: str,
    method: Method,
    principal: Principal,
    dataplane: Dataplane,
    caller: Caller,
    client: DataplaneClient,
    audit: AuditLog,
    reservation: Reservation | None,
) -> JSONResponse:
    decision: Final = decide(path)
    if isinstance(decision, Denied):
        audit.record(
            principal.subject,
            principal.email,
            dataplane.id,
            method,
            decision.path,
            "denied",
            detail=decision.reason,
            reservation=reservation,
        )
        raise ControlPlaneException(forbidden(f"{redact_path(decision.path)}: {decision.reason}."))
    result: Final = await client.request(
        method, dataplane, decision.path, caller, params=dict(request.query_params), json=await _json_body(request)
    )
    match result:
        case DataplaneReply(status=status, credential=credential) if 300 <= status < 400:
            audit.record(
                principal.subject,
                principal.email,
                dataplane.id,
                method,
                decision.path,
                "denied",
                status,
                credential,
                detail="dataplane answered with a redirect, which the control plane never follows",
                reservation=reservation,
            )
            raise ControlPlaneException(bad_gateway("Dataplane answered with a redirect, which is not followed."))
        case DataplaneReply(status=status, body=body, credential=credential):
            audit.record(
                principal.subject,
                principal.email,
                dataplane.id,
                method,
                decision.path,
                "allowed",
                status,
                credential,
                reservation=reservation,
            )
            return JSONResponse(status_code=status, content=body, headers={DATAPLANE_HEADER: dataplane.id})
        case DataplaneUnreachable(detail=detail, attempts=attempts):
            audit.record(
                principal.subject,
                principal.email,
                dataplane.id,
                method,
                decision.path,
                "unreachable",
                detail=detail,
                reservation=reservation,
            )
            raise ControlPlaneException(bad_gateway(f"Dataplane unreachable after {attempts} attempt(s): {detail}"))
        case CredentialUnavailable(reason=reason):
            audit.record(
                principal.subject,
                principal.email,
                dataplane.id,
                method,
                decision.path,
                "unreachable",
                detail=reason,
                reservation=reservation,
            )
            raise ControlPlaneException(bad_gateway(f"No usable credential for this dataplane: {reason}"))


@router.api_route("/dataplanes/{dataplane_id}/proxy/{path:path}", methods=list(READ_METHODS))
async def proxy_read(
    request: Request,
    path: str,
    principal: Annotated[Principal, Security(require_scopes, scopes=[Scope.PROXY_READ.value])],
    dataplane: Annotated[Dataplane, Depends(dataplane_access)],
    caller: Annotated[Caller, Depends(get_caller)],
    client: Annotated[DataplaneClient, Depends(get_dataplane_client)],
    audit: Annotated[AuditLog, Depends(get_audit_log)],
) -> JSONResponse:
    """Read one allowlisted management route on a dataplane as the caller."""
    return await _forward(request, path, principal, dataplane, caller, client, audit)


@router.api_route("/dataplanes/{dataplane_id}/proxy/{path:path}", methods=list(WRITE_METHODS))
async def proxy_write(
    request: Request,
    path: str,
    principal: Annotated[Principal, Security(require_scopes, scopes=[Scope.PROXY_WRITE.value])],
    dataplane: Annotated[Dataplane, Depends(dataplane_access)],
    caller: Annotated[Caller, Depends(get_caller)],
    client: Annotated[DataplaneClient, Depends(get_dataplane_client)],
    audit: Annotated[AuditLog, Depends(get_audit_log)],
) -> JSONResponse:
    """Write through one allowlisted management route on a dataplane as the caller, sent exactly once."""
    return await _forward(request, path, principal, dataplane, caller, client, audit)
