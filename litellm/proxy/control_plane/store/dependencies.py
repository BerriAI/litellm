import asyncio
from collections.abc import AsyncIterator
from typing import Annotated, Final

from fastapi import Depends, Request
from fastapi.exceptions import RequestValidationError
from fastapi.security import SecurityScopes

from litellm.proxy.control_plane.audit.dependencies import get_audit_log
from litellm.proxy.control_plane.audit.event import CONTROL_PLANE_ID
from litellm.proxy.control_plane.audit.log import AuditLog
from litellm.proxy.control_plane.auth.dependencies import get_principal
from litellm.proxy.control_plane.auth.rbac import Principal, Scope
from litellm.proxy.control_plane.common.exceptions import ControlPlaneException, forbidden, service_unavailable
from litellm.proxy.control_plane.store.runtime import RuntimeConfig


def get_runtime_config() -> RuntimeConfig:
    raise RuntimeError("get_runtime_config is not bound, build the app with create_app")


async def audited_write(
    request: Request,
    security_scopes: SecurityScopes,
    principal: Annotated[Principal, Depends(get_principal)],
    audit: Annotated[AuditLog, Depends(get_audit_log)],
) -> AsyncIterator[Principal]:
    reservation: Final = audit.reserve()
    try:
        if reservation is None:
            raise ControlPlaneException(service_unavailable("Audit sink backpressure, try again shortly."))
        if principal.missing_scopes(frozenset(Scope(scope) for scope in security_scopes.scopes)):
            raise ControlPlaneException(forbidden("Missing runtime write scope."))
        if not principal.dataplanes.all:
            raise ControlPlaneException(forbidden("Runtime writes require a fleet-wide dataplane binding."))
        yield principal
    except ControlPlaneException as error:
        audit.record(
            principal.subject,
            principal.email,
            CONTROL_PLANE_ID,
            request.method,
            request.url.path,
            "denied",
            error.body.status,
            detail=error.body.detail,
            reservation=reservation,
        )
        raise
    except RequestValidationError:
        audit.record(
            principal.subject,
            principal.email,
            CONTROL_PLANE_ID,
            request.method,
            request.url.path,
            "denied",
            422,
            detail="Invalid runtime write body.",
            reservation=reservation,
        )
        raise
    except (Exception, asyncio.CancelledError) as error:
        audit.record(
            principal.subject,
            principal.email,
            CONTROL_PLANE_ID,
            request.method,
            request.url.path,
            "unreachable",
            503 if isinstance(error, asyncio.CancelledError) else 500,
            detail="Runtime write failed.",
            reservation=reservation,
        )
        raise
    else:
        audit.record(
            principal.subject,
            principal.email,
            CONTROL_PLANE_ID,
            request.method,
            request.url.path,
            "allowed",
            204 if request.method == "DELETE" else 200,
            reservation=reservation,
        )
    finally:
        if reservation is not None:
            reservation.release()
