from typing import Annotated, Final, TypeAlias

from fastapi import APIRouter, Depends, HTTPException, Path, Query

from litellm.models.alert import (
    Alert,
    AlertFilter,
    AlertID,
    AlertStatus,
    AlertStore,
    AlertUpdate,
    AlertView,
    AlertViewPage,
)
from litellm.models.performance_alert import AlertMonitoring, AlertMonitoringUpdate
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.deps.repositories import AlertMonitoringRepositoryDep, AlertRepositoryDep
from litellm.types.proxy.auth.user_roles import LitellmUserRoles

router: Final = APIRouter(tags=["alerts"])
Auth: TypeAlias = Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)]
AlertPath: TypeAlias = Annotated[AlertID, Path(description="Alert ID returned by GET /alerts.")]


def _authorize(user: UserAPIKeyAuth, *, write: bool) -> None:
    if user.user_role != LitellmUserRoles.PROXY_ADMIN and (
        write or user.user_role != LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY
    ):
        raise HTTPException(status_code=403, detail="Only gateway admins can manage alerts")


def _authorized_store(user: UserAPIKeyAuth, repository: AlertStore | None, *, write: bool) -> AlertStore:
    _authorize(user, write=write)
    if repository is None:
        raise HTTPException(status_code=503, detail="Alerts require a database connection")
    return repository


def _found(alert: Alert | None) -> Alert:
    if alert is None:
        raise HTTPException(status_code=404, detail="Alert not found")
    return alert


@router.get("/alerts", response_model=AlertViewPage, summary="List saved alerts")
async def list_alerts(
    user: Auth,
    repository: AlertRepositoryDep,
    status: Annotated[AlertStatus, Query(description="Status to list.")] = "open",
    limit: Annotated[int, Query(ge=1, le=100, description="Maximum alerts per page.")] = 25,
    cursor: Annotated[AlertID | None, Query(description="next_cursor from the preceding page.")] = None,
    alert_type: Annotated[
        AlertFilter | None, Query(alias="type", description="Alert category, or all performance alerts.")
    ] = None,
) -> AlertViewPage:
    """Read saved alerts with current resource names and eight days of Usage history. Requires a gateway admin."""
    store: Final = _authorized_store(user, repository, write=False)
    page: Final = await store.list(status, limit, int(cursor) if cursor is not None else None, alert_type)
    return AlertViewPage(data=await store.enrich(page.data), next_cursor=page.next_cursor)


@router.get("/alerts/monitoring", response_model=AlertMonitoring, summary="Read performance monitoring settings")
async def get_monitoring(user: Auth, repository: AlertMonitoringRepositoryDep) -> AlertMonitoring:
    """Read the gateway's opt-out setting and free allowance. Requires a gateway admin or view-only admin."""
    _authorize(user, write=False)
    if repository is None:
        raise HTTPException(status_code=503, detail="Alerts require a database connection")
    return await repository.monitoring()


@router.patch("/alerts/monitoring", response_model=AlertMonitoring, summary="Enable or disable performance monitoring")
async def update_monitoring(
    data: AlertMonitoringUpdate, user: Auth, repository: AlertMonitoringRepositoryDep
) -> AlertMonitoring:
    """Change monitoring for every worker using this database. Only gateway admins can write. Keeps saved alerts."""
    _authorize(user, write=True)
    if repository is None:
        raise HTTPException(status_code=503, detail="Alerts require a database connection")
    if data.enabled and (await repository.monitoring()).disabled_by_config:
        raise HTTPException(status_code=409, detail="Performance monitoring is disabled in proxy settings")
    return await repository.set_enabled(data.enabled)


@router.get("/alerts/{alert_id}", response_model=AlertView, summary="Read an alert")
async def get_alert(alert_id: AlertPath, user: Auth, repository: AlertRepositoryDep) -> AlertView:
    """Return the saved evidence and status. Requires a gateway admin or view-only admin."""
    store: Final = _authorized_store(user, repository, write=False)
    alert: Final = _found(await store.get(int(alert_id)))
    return (await store.enrich((alert,)))[0]


@router.patch("/alerts/{alert_id}", response_model=Alert, summary="Change an alert's status")
async def update_alert(alert_id: AlertPath, data: AlertUpdate, user: Auth, repository: AlertRepositoryDep) -> Alert:
    """Resolve, dismiss, or reopen an alert. Only gateway admins can write. Returns the updated alert."""
    store: Final = _authorized_store(user, repository, write=True)
    return _found(await store.set_status(int(alert_id), data.status, user.user_id or "proxy-admin"))
