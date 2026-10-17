from typing import Annotated, Final

from fastapi import APIRouter, Depends, Query, Security

from litellm.proxy.control_plane.audit.dependencies import get_audit_log
from litellm.proxy.control_plane.audit.log import AuditLog
from litellm.proxy.control_plane.auth.dependencies import require_scopes
from litellm.proxy.control_plane.auth.rbac import DataplaneBinding, Principal, Scope
from litellm.proxy.control_plane.common.exceptions import ControlPlaneException, forbidden
from litellm.types.proxy.control_plane.audit import AuditEventResponse, AuditListResponse

router: Final = APIRouter(prefix="/audit", tags=["audit"])


def visible_dataplanes(binding: DataplaneBinding, requested: str | None) -> frozenset[str] | None:
    """What this caller may read: everything when fleet-wide, else only bound ids, and a requested id must be bound."""
    if requested is not None and not (binding.all or requested in binding.ids):
        raise ControlPlaneException(forbidden(f"Dataplane {requested!r} is not bound to this caller."))
    if requested is not None:
        return frozenset({requested})
    return None if binding.all else binding.ids


@router.get("", response_model=AuditListResponse)
async def list_audit(
    principal: Annotated[Principal, Security(require_scopes, scopes=[Scope.AUDIT_READ.value])],
    audit: Annotated[AuditLog, Depends(get_audit_log)],
    limit: Annotated[int, Query(ge=1, le=1000)] = 100,
    dataplane: Annotated[str | None, Query(description="Only events against this dataplane id.")] = None,
    actor: Annotated[str | None, Query(description="Only events by this control plane subject.")] = None,
) -> AuditListResponse:
    """Newest first, what every control plane caller did on the dataplanes this caller is bound to."""
    visible: Final = visible_dataplanes(principal.dataplanes, dataplane)
    return AuditListResponse(
        items=tuple(
            AuditEventResponse(
                at=event.at,
                actor=event.actor,
                actor_email=event.actor_email,
                dataplane_id=event.dataplane_id,
                method=event.method,
                path=event.path,
                decision=event.decision,
                status=event.status,
                credential=event.credential,
                detail=event.detail,
            )
            for event in audit.recent(limit, visible, actor)
        )
    )
