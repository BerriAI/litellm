"""Alerts and their supporting Activity backed by existing daily aggregates."""

from collections.abc import Generator
from contextlib import contextmanager
from datetime import date, datetime, timezone
from typing import Final

from fastapi import APIRouter, HTTPException

from litellm.models.issues import IssueActivityStore, IssueStore
from litellm.proxy.alerting_endpoints.usage_alerts import (
    Limit,
    Offset,
    ScopeId,
    User,
    require_repository,
    require_scope,
    require_write_access,
)
from litellm.proxy.deps.repositories import IssueActivityRepositoryDep, IssuesRepositoryDep, UsageAlertsRepositoryDep
from litellm.proxy.management.teams.authz import is_team_admin
from litellm.proxy.management_endpoints.common_utils import team_member_has_permission
from litellm.repositories.analytics_budget import AnalyticsCapacityError
from litellm.types.proxy.auth.user_roles import LitellmUserRoles
from litellm.types.proxy.issues import (
    Alert,
    AlertActivityResponse,
    AlertCohort,
    AlertFamily,
    AlertFamilyList,
    AlertFamilySetting,
    AlertFamilyUpdate,
    AlertImpact,
    AlertKind,
    AlertList,
    AlertScope,
    AlertSignalList,
    AlertUpdate,
    PerformanceMonitoring,
    PerformanceMonitoringUpdate,
)
from litellm.types.proxy.management_endpoints.common_daily_activity import (
    DailyActivityKeyPageResponse,
    DailyActivityUserPageResponse,
)
from litellm.types.proxy.usage_alert_defaults import PerformanceIssueFamily
from litellm.types.proxy.usage_alerts import UsageAlertStatus

router: Final = APIRouter(prefix="/alerts", tags=["alerts"])


def require_issues(repository: IssueStore | None) -> IssueStore:
    if repository is None:
        raise HTTPException(status_code=503, detail="Alerts require a database connection")
    return repository


def require_activity(repository: IssueActivityStore | None) -> IssueActivityStore:
    if repository is None:
        raise HTTPException(status_code=503, detail="Alert Activity requires a database connection")
    return repository


@contextmanager
def activity_budget() -> Generator[None]:
    try:
        yield
    except AnalyticsCapacityError as exc:
        raise HTTPException(status_code=503, detail=str(exc), headers={"Retry-After": "5"}) from exc


async def authorized_issue(
    alert_id: str,
    user: User,
    repository: IssuesRepositoryDep,
    alerts: UsageAlertsRepositoryDep,
    *,
    write: bool = False,
) -> Alert:
    issue: Final = await require_issues(repository).get_issue(alert_id)
    if issue is None:
        raise HTTPException(status_code=404, detail="Alert not found")
    await require_scope(
        user,
        require_repository(alerts),
        "model" if issue.cohort.scope_type == "provider" else issue.cohort.scope_type,
        issue.cohort.scope_id,
        write=write,
    )
    return issue


async def can_view_activity(issue: Alert, user: User, repository: IssueStore) -> bool:
    if issue.cohort.scope_type != "team" or user.user_role in (
        LitellmUserRoles.PROXY_ADMIN,
        LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY,
    ):
        return True
    if user.user_id is None or issue.cohort.scope_id is None:
        return False
    team: Final = await repository.team_activity_access(issue.cohort.scope_id, user.user_id)
    return team is not None and (
        is_team_admin(user, team) or team_member_has_permission(user, team, "/team/daily/activity")
    )


async def require_activity_access(issue: Alert, user: User, repository: IssuesRepositoryDep) -> None:
    if not await can_view_activity(issue, user, require_issues(repository)):
        raise HTTPException(status_code=403, detail="Full team Activity permission is required for this alert")


async def require_status_scopes(
    issue: Alert, user: User, repository: IssueStore, alerts: UsageAlertsRepositoryDep
) -> tuple[AlertCohort, ...] | None:
    store: Final = require_repository(alerts)
    await require_scope(
        user,
        store,
        "model" if issue.cohort.scope_type == "provider" else issue.cohort.scope_type,
        issue.cohort.scope_id,
        write=True,
    )
    if user.user_role == LitellmUserRoles.PROXY_ADMIN:
        return None
    scopes: Final = await repository.status_scopes(issue.id)
    if len(scopes) > 10:
        raise HTTPException(status_code=403, detail="An administrator must update this alert's status")
    for scope in scopes:
        await require_scope(
            user, store, "model" if scope.scope_type == "provider" else scope.scope_type, scope.scope_id, write=True
        )
    return scopes


def activity_range(issue: Alert, start_date: date | None, end_date: date | None) -> tuple[date, date]:
    start: Final = start_date or issue.range_start
    end: Final = end_date or issue.range_end
    if start > end or (end - start).days > 365:
        raise HTTPException(status_code=422, detail="Choose an ordered date range of at most 366 days")
    return start, end


@router.get("", response_model=AlertList)
async def list_alerts(
    user: User,
    repository: IssuesRepositoryDep,
    alerts: UsageAlertsRepositoryDep,
    kind: AlertKind | None = None,
    family: AlertFamily | None = None,
    status: UsageAlertStatus | None = None,
    scope_type: AlertScope | None = None,
    scope_id: ScopeId = None,
    limit: Limit = 50,
    offset: Offset = 0,
) -> AlertList:
    await require_scope(
        user, require_repository(alerts), "model" if scope_type == "provider" else scope_type, scope_id, write=False
    )
    return await require_issues(repository).list_issues(kind, family, status, scope_type, scope_id, limit, offset)


@router.get("/monitoring", response_model=PerformanceMonitoring)
async def get_performance_monitoring(
    user: User, repository: IssuesRepositoryDep, alerts: UsageAlertsRepositoryDep
) -> PerformanceMonitoring:
    from litellm.proxy.proxy_server import premium_user

    await require_scope(user, require_repository(alerts), "deployment", None, write=False)
    return await require_issues(repository).performance_monitoring(premium_user is True)


@router.patch("/monitoring", response_model=PerformanceMonitoring)
async def set_performance_monitoring(
    data: PerformanceMonitoringUpdate, user: User, repository: IssuesRepositoryDep, alerts: UsageAlertsRepositoryDep
) -> PerformanceMonitoring:
    from litellm.proxy.proxy_server import (
        premium_user,
        scheduler,  # pyright: ignore[reportUnknownVariableType]  # legacy scheduler initialized during startup
    )
    from litellm.proxy.spend_tracking.usage_alert_job import USAGE_ALERT_JOB_ID

    require_write_access(user, "/alerts/monitoring")
    await require_scope(user, require_repository(alerts), "deployment", None, write=True)
    saved: Final = await require_issues(repository).set_performance_monitoring(data.enabled, premium_user is True)
    if saved.can_detect and scheduler is not None and scheduler.get_job(USAGE_ALERT_JOB_ID) is not None:  # pyright: ignore[reportUnknownMemberType]  # APScheduler has no parameter stubs
        scheduler.modify_job(USAGE_ALERT_JOB_ID, next_run_time=datetime.now(timezone.utc))  # pyright: ignore[reportUnknownMemberType]  # APScheduler has no parameter stubs
    return saved


@router.get("/defaults", response_model=AlertFamilyList)
async def list_alert_defaults(
    user: User,
    repository: IssuesRepositoryDep,
    alerts: UsageAlertsRepositoryDep,
) -> AlertFamilyList:
    await require_scope(user, require_repository(alerts), "deployment", None, write=False)
    return await require_issues(repository).family_settings()


@router.patch("/defaults/{family}", response_model=AlertFamilySetting)
async def update_alert_default(
    family: PerformanceIssueFamily,
    data: AlertFamilyUpdate,
    user: User,
    repository: IssuesRepositoryDep,
    alerts: UsageAlertsRepositoryDep,
) -> AlertFamilySetting:
    require_write_access(user, f"/alerts/defaults/{family}")
    await require_scope(user, require_repository(alerts), "deployment", None, write=True)
    try:
        result: Final = await require_issues(repository).update_family(family, data)
    except ValueError as exc:
        raise HTTPException(
            status_code=422, detail="Invalid family policy; check direction and history bounds"
        ) from exc
    if result is None:
        raise HTTPException(status_code=409, detail="Built-in rules changed or are unavailable; reload and retry")
    return result


@router.get("/{alert_id}", response_model=Alert)
async def get_alert(
    alert_id: str,
    user: User,
    repository: IssuesRepositoryDep,
    alerts: UsageAlertsRepositoryDep,
) -> Alert:
    issue: Final = await authorized_issue(alert_id, user, repository, alerts)
    activity_allowed: Final = await can_view_activity(issue, user, require_issues(repository))
    try:
        require_write_access(user, f"/alerts/{alert_id}")
        await require_status_scopes(issue, user, require_issues(repository), alerts)
    except HTTPException as exc:
        if exc.status_code != 403:
            raise
        return issue.model_copy(update={"can_update_status": False, "can_view_activity": activity_allowed})
    return issue.model_copy(update={"can_update_status": True, "can_view_activity": activity_allowed})


@router.patch("/{alert_id}", response_model=Alert)
async def update_alert(
    alert_id: str,
    data: AlertUpdate,
    user: User,
    repository: IssuesRepositoryDep,
    alerts: UsageAlertsRepositoryDep,
) -> Alert:
    require_write_access(user, f"/alerts/{alert_id}")
    issue: Final = await authorized_issue(alert_id, user, repository, alerts)
    scopes: Final = await require_status_scopes(issue, user, require_issues(repository), alerts)
    if issue.kind == "model_deprecation" and not issue.active and data.status == "open":
        raise HTTPException(status_code=409, detail="This retirement alert reopens when the model receives new traffic")
    changed: Final = await require_issues(repository).set_status(
        alert_id, data.status, data.expected_status, user.user_id, authorized_scopes=scopes
    )
    if changed is None:
        raise HTTPException(status_code=409, detail="Alert status changed; reload and retry")
    return changed


@router.get("/{alert_id}/signals", response_model=AlertSignalList)
async def list_alert_signals(
    alert_id: str,
    user: User,
    repository: IssuesRepositoryDep,
    alerts: UsageAlertsRepositoryDep,
    limit: Limit = 50,
    offset: Offset = 0,
) -> AlertSignalList:
    issue: Final = await authorized_issue(alert_id, user, repository, alerts)
    visible_scope: Final = (
        None
        if user.user_role
        in (
            LitellmUserRoles.PROXY_ADMIN,
            LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY,
        )
        else issue.cohort
    )
    return await require_issues(repository).list_signals(alert_id, limit, offset, visible_scope)


@router.get("/{alert_id}/activity", response_model=AlertActivityResponse)
async def get_alert_activity(
    alert_id: str,
    user: User,
    repository: IssuesRepositoryDep,
    alerts: UsageAlertsRepositoryDep,
    activity: IssueActivityRepositoryDep,
    start_date: date | None = None,
    end_date: date | None = None,
    model_limit: Limit = 20,
) -> AlertActivityResponse:
    issue: Final = await authorized_issue(alert_id, user, repository, alerts)
    await require_activity_access(issue, user, repository)
    start, end = activity_range(issue, start_date, end_date)
    with activity_budget():
        return await require_activity(activity).daily_activity(issue.cohort, start, end, model_limit)


@router.get("/{alert_id}/keys", response_model=DailyActivityKeyPageResponse)
async def get_alert_keys(
    alert_id: str,
    user: User,
    repository: IssuesRepositoryDep,
    alerts: UsageAlertsRepositoryDep,
    activity: IssueActivityRepositoryDep,
    start_date: date | None = None,
    end_date: date | None = None,
    limit: Limit = 20,
    offset: Offset = 0,
) -> DailyActivityKeyPageResponse:
    issue: Final = await authorized_issue(alert_id, user, repository, alerts)
    await require_activity_access(issue, user, repository)
    start, end = activity_range(issue, start_date, end_date)
    with activity_budget():
        return await require_activity(activity).keys(issue.cohort, start, end, limit, offset)


@router.get("/{alert_id}/users", response_model=DailyActivityUserPageResponse)
async def get_alert_users(
    alert_id: str,
    user: User,
    repository: IssuesRepositoryDep,
    alerts: UsageAlertsRepositoryDep,
    activity: IssueActivityRepositoryDep,
    start_date: date | None = None,
    end_date: date | None = None,
    limit: Limit = 20,
    offset: Offset = 0,
) -> DailyActivityUserPageResponse:
    issue: Final = await authorized_issue(alert_id, user, repository, alerts)
    await require_activity_access(issue, user, repository)
    if issue.cohort.scope_type == "team":
        raise HTTPException(status_code=422, detail="Historical team aggregates do not retain user membership")
    start, end = activity_range(issue, start_date, end_date)
    with activity_budget():
        return await require_activity(activity).users(issue.cohort, start, end, limit, offset)


@router.get("/{alert_id}/impact", response_model=AlertImpact)
async def get_alert_impact(
    alert_id: str,
    user: User,
    repository: IssuesRepositoryDep,
    alerts: UsageAlertsRepositoryDep,
    activity: IssueActivityRepositoryDep,
    start_date: date | None = None,
    end_date: date | None = None,
) -> AlertImpact:
    issue: Final = await authorized_issue(alert_id, user, repository, alerts)
    await require_activity_access(issue, user, repository)
    start, end = activity_range(issue, start_date, end_date)
    with activity_budget():
        return await require_activity(activity).impact(issue.cohort, start, end)
