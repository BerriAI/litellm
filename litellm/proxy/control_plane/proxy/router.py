"""Authenticated management passthrough: one allowlisted dataplane route, as the caller, audited."""

from collections.abc import Mapping
from typing import Annotated, Final

from fastapi import APIRouter, Depends, Request, Security
from fastapi.responses import JSONResponse, Response
from fastapi.security import SecurityScopes
from pydantic import TypeAdapter, ValidationError

from litellm.proxy.control_plane.audit.dependencies import get_audit_log
from litellm.proxy.control_plane.audit.log import AuditLog, redact_path
from litellm.proxy.control_plane.auth.dependencies import get_principal, require_scopes
from litellm.proxy.control_plane.auth.rbac import Principal, Scope
from litellm.proxy.control_plane.common.exceptions import ControlPlaneException, bad_gateway, error_body, forbidden
from litellm.proxy.control_plane.dataplanes.client import (
    DataplaneClient,
    DataplaneReply,
    DataplaneResponseTooLarge,
    DataplaneUnreachable,
    Method,
)
from litellm.proxy.control_plane.dataplanes.credentials import Caller, CredentialUnavailable
from litellm.proxy.control_plane.dataplanes.dependencies import (
    dataplane_access,
    get_caller,
    get_dataplane_client,
    get_registry,
)
from litellm.proxy.control_plane.dataplanes.registry import Dataplane, DataplaneRegistry
from litellm.proxy.control_plane.proxy.allowlist import Denied, decide

DATAPLANE_HEADER: Final = "X-LiteLLM-Dataplane"
READ_METHODS: Final = ("GET", "HEAD")
WRITE_METHODS: Final = ("POST", "PUT", "PATCH", "DELETE")

router: Final = APIRouter(tags=["proxy"])

_BODY: Final = TypeAdapter(Mapping[str, object])
_METHOD: Final[TypeAdapter[Method]] = TypeAdapter(Method)


def bad_request(detail: str) -> ControlPlaneException:
    return ControlPlaneException(error_body("bad-request", "Bad Request", 400, detail))


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
) -> Response:
    method: Final = _METHOD.validate_python(request.method)
    decision: Final = decide(path)
    if isinstance(decision, Denied):
        audit.record(
            principal.subject, principal.email, dataplane.id, method, decision.path, "denied", detail=decision.reason
        )
        raise ControlPlaneException(forbidden(f"{redact_path(decision.path)}: {decision.reason}."))
    result: Final = await client.request(
        method,
        dataplane,
        decision.path,
        caller,
        params=tuple(request.query_params.multi_items()),
        json=await _body_or_audit(request, principal, dataplane, method, decision.path, audit),
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
            )
            raise ControlPlaneException(bad_gateway("Dataplane answered with a redirect, which is not followed."))
        case DataplaneReply(
            status=status,
            body=body,
            credential=credential,
            raw_content=raw_content,
            content_type=content_type,
            content_disposition=content_disposition,
            cache_control=cache_control,
        ):
            audit.record(
                principal.subject, principal.email, dataplane.id, method, decision.path, "allowed", status, credential
            )
            if raw_content is not None:
                headers: Final = {
                    **({"Content-Type": content_type} if content_type is not None else {}),
                    **({"Content-Disposition": content_disposition} if content_disposition is not None else {}),
                    **({"Cache-Control": cache_control} if cache_control is not None else {}),
                    DATAPLANE_HEADER: dataplane.id,
                }
                return Response(content=raw_content, status_code=status, headers=headers)
            return JSONResponse(
                status_code=status,
                content=body,
                headers={
                    **({"Cache-Control": cache_control} if cache_control is not None else {}),
                    **({"Content-Disposition": content_disposition} if content_disposition is not None else {}),
                    DATAPLANE_HEADER: dataplane.id,
                },
            )
        case DataplaneResponseTooLarge(limit_bytes=limit_bytes):
            audit.record(
                principal.subject,
                principal.email,
                dataplane.id,
                method,
                decision.path,
                "unreachable",
                502,
                detail="dataplane response exceeded the configured byte limit",
            )
            raise ControlPlaneException(
                error_body(
                    "dataplane-response-too-large",
                    "Bad Gateway",
                    502,
                    f"Dataplane response exceeded the {limit_bytes}-byte limit.",
                )
            )
        case DataplaneUnreachable(detail=detail, attempts=attempts):
            audit.record(
                principal.subject, principal.email, dataplane.id, method, decision.path, "unreachable", detail=detail
            )
            raise ControlPlaneException(bad_gateway(f"Dataplane unreachable after {attempts} attempt(s): {detail}"))
        case CredentialUnavailable(reason=reason):
            audit.record(
                principal.subject, principal.email, dataplane.id, method, decision.path, "unreachable", detail=reason
            )
            raise ControlPlaneException(bad_gateway(f"No usable credential for this dataplane: {reason}"))


async def _body_or_audit(
    request: Request,
    principal: Principal,
    dataplane: Dataplane,
    method: Method,
    path: str,
    audit: AuditLog,
) -> Mapping[str, object] | None:
    try:
        return await _json_body(request)
    except ControlPlaneException as error:
        audit.record(
            principal.subject,
            principal.email,
            dataplane.id,
            method,
            path,
            "denied",
            error.body.status,
            detail=error.body.detail,
        )
        raise


async def _passthrough_dataplane_access(
    request: Request,
    dataplane_id: str,
    path: str,
    principal: Annotated[Principal, Depends(get_principal)],
    registry: Annotated[DataplaneRegistry, Depends(get_registry)],
    audit: Annotated[AuditLog, Depends(get_audit_log)],
) -> Dataplane:
    try:
        return await dataplane_access(dataplane_id, principal, registry)
    except ControlPlaneException as error:
        if error.body.status == 403:
            audit.record(
                principal.subject,
                principal.email,
                dataplane_id,
                request.method,
                "/" + path.lstrip("/"),
                "denied",
                error.body.status,
                detail=error.body.detail,
            )
        raise


async def _passthrough_scope(
    security_scopes: SecurityScopes,
    request: Request,
    dataplane_id: str,
    path: str,
    principal: Annotated[Principal, Depends(get_principal)],
    audit: Annotated[AuditLog, Depends(get_audit_log)],
) -> Principal:
    try:
        return await require_scopes(security_scopes, principal)
    except ControlPlaneException as error:
        audit.record(
            principal.subject,
            principal.email,
            dataplane_id,
            request.method,
            "/" + path.lstrip("/"),
            "denied",
            error.body.status,
            detail=error.body.detail,
        )
        raise


@router.api_route("/dataplanes/{dataplane_id}/proxy/{path:path}", methods=list(READ_METHODS))
async def proxy_read(
    request: Request,
    path: str,
    principal: Annotated[Principal, Security(_passthrough_scope, scopes=[Scope.PROXY_READ.value])],
    dataplane: Annotated[Dataplane, Depends(_passthrough_dataplane_access)],
    caller: Annotated[Caller, Depends(get_caller)],
    client: Annotated[DataplaneClient, Depends(get_dataplane_client)],
    audit: Annotated[AuditLog, Depends(get_audit_log)],
) -> Response:
    """Read one allowlisted management route on a dataplane as the caller."""
    return await _forward(request, path, principal, dataplane, caller, client, audit)


@router.api_route("/dataplanes/{dataplane_id}/proxy/{path:path}", methods=list(WRITE_METHODS))
async def proxy_write(
    request: Request,
    path: str,
    principal: Annotated[Principal, Security(_passthrough_scope, scopes=[Scope.PROXY_WRITE.value])],
    dataplane: Annotated[Dataplane, Depends(_passthrough_dataplane_access)],
    caller: Annotated[Caller, Depends(get_caller)],
    client: Annotated[DataplaneClient, Depends(get_dataplane_client)],
    audit: Annotated[AuditLog, Depends(get_audit_log)],
) -> Response:
    """Write through one allowlisted management route on a dataplane as the caller, sent exactly once."""
    return await _forward(request, path, principal, dataplane, caller, client, audit)
