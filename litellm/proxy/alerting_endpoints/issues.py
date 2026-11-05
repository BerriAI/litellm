"""Unified issue inbox and investigations backed by existing daily aggregates."""

from datetime import date
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
)
from litellm.proxy.deps.repositories import IssueActivityRepositoryDep, IssuesRepositoryDep, UsageAlertsRepositoryDep
from litellm.types.proxy.auth.user_roles import LitellmUserRoles
from litellm.types.proxy.issues import (
    Issue,
    IssueFamily,
    IssueFamilyList,
    IssueFamilySetting,
    IssueFamilyUpdate,
    IssueImpact,
    IssueKind,
    IssueList,
    IssueScope,
    IssueSignalList,
    IssueUpdate,
)
from litellm.types.proxy.management_endpoints.common_daily_activity import (
    DailyActivityKeyPageResponse,
    DailyActivityUserPageResponse,
    SpendAnalyticsPaginatedResponse,
)
from litellm.types.proxy.usage_alert_defaults import PerformanceIssueFamily
from litellm.types.proxy.usage_alerts import UsageAlertStatus

router: Final = APIRouter(prefix="/issues", tags=["issues"])


def require_issues(repository: IssueStore | None) -> IssueStore:
    if repository is None:
        raise HTTPException(status_code=503, detail="Issues require a database connection")
    return repository


def require_activity(repository: IssueActivityStore | None) -> IssueActivityStore:
    if repository is None:
        raise HTTPException(status_code=503, detail="Issue Activity requires a database connection")
    return repository


async def authorized_issue(
    issue_id: str,
    user: User,
    repository: IssuesRepositoryDep,
    alerts: UsageAlertsRepositoryDep,
    *,
    write: bool = False,
) -> Issue:
    issue: Final = await require_issues(repository).get_issue(issue_id)
    if issue is None:
        raise HTTPException(status_code=404, detail="Issue not found")
    await require_scope(
        user,
        require_repository(alerts),
        "model" if issue.cohort.scope_type == "provider" else issue.cohort.scope_type,
        issue.cohort.scope_id,
        write=write,
    )
    return issue


def activity_range(issue: Issue, start_date: date | None, end_date: date | None) -> tuple[date, date]:
    start: Final = start_date or issue.range_start
    end: Final = end_date or issue.range_end
    if start > end or (end - start).days > 365:
        raise HTTPException(status_code=422, detail="Choose an ordered date range of at most 366 days")
    return start, end


@router.get("", response_model=IssueList)
async def list_issues(
    user: User,
    repository: IssuesRepositoryDep,
    alerts: UsageAlertsRepositoryDep,
    kind: IssueKind | None = None,
    family: IssueFamily | None = None,
    status: UsageAlertStatus | None = None,
    scope_type: IssueScope | None = None,
    scope_id: ScopeId = None,
    limit: Limit = 50,
    offset: Offset = 0,
) -> IssueList:
    await require_scope(
        user, require_repository(alerts), "model" if scope_type == "provider" else scope_type, scope_id, write=False
    )
    return await require_issues(repository).list_issues(kind, family, status, scope_type, scope_id, limit, offset)


@router.get("/families", response_model=IssueFamilyList)
async def list_issue_families(
    user: User,
    repository: IssuesRepositoryDep,
    alerts: UsageAlertsRepositoryDep,
) -> IssueFamilyList:
    await require_scope(user, require_repository(alerts), "deployment", None, write=False)
    return await require_issues(repository).family_settings()


@router.patch("/families/{family}", response_model=IssueFamilySetting)
async def update_issue_family(
    family: PerformanceIssueFamily,
    data: IssueFamilyUpdate,
    user: User,
    repository: IssuesRepositoryDep,
    alerts: UsageAlertsRepositoryDep,
) -> IssueFamilySetting:
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


@router.get("/{issue_id}", response_model=Issue)
async def get_issue(
    issue_id: str,
    user: User,
    repository: IssuesRepositoryDep,
    alerts: UsageAlertsRepositoryDep,
) -> Issue:
    return await authorized_issue(issue_id, user, repository, alerts)


@router.patch("/{issue_id}", response_model=Issue)
async def update_issue(
    issue_id: str,
    data: IssueUpdate,
    user: User,
    repository: IssuesRepositoryDep,
    alerts: UsageAlertsRepositoryDep,
) -> Issue:
    issue: Final = await authorized_issue(issue_id, user, repository, alerts, write=True)
    if issue.kind == "model_deprecation" and not issue.active and data.status == "open":
        raise HTTPException(status_code=409, detail="This retirement issue reopens when the model receives new traffic")
    changed: Final = await require_issues(repository).set_status(
        issue_id, data.status, data.expected_status, user.user_id
    )
    if changed is None:
        raise HTTPException(status_code=409, detail="Issue status changed; reload and retry")
    return changed


@router.get("/{issue_id}/signals", response_model=IssueSignalList)
async def list_issue_signals(
    issue_id: str,
    user: User,
    repository: IssuesRepositoryDep,
    alerts: UsageAlertsRepositoryDep,
    limit: Limit = 50,
    offset: Offset = 0,
) -> IssueSignalList:
    issue: Final = await authorized_issue(issue_id, user, repository, alerts)
    visible_scope: Final = (
        None
        if user.user_role
        in (
            LitellmUserRoles.PROXY_ADMIN,
            LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY,
        )
        else issue.cohort
    )
    return await require_issues(repository).list_signals(issue_id, limit, offset, visible_scope)


@router.get("/{issue_id}/activity", response_model=SpendAnalyticsPaginatedResponse)
async def get_issue_activity(
    issue_id: str,
    user: User,
    repository: IssuesRepositoryDep,
    alerts: UsageAlertsRepositoryDep,
    activity: IssueActivityRepositoryDep,
    start_date: date | None = None,
    end_date: date | None = None,
) -> SpendAnalyticsPaginatedResponse:
    issue: Final = await authorized_issue(issue_id, user, repository, alerts)
    start, end = activity_range(issue, start_date, end_date)
    return await require_activity(activity).daily_activity(issue.cohort, start, end)


@router.get("/{issue_id}/keys", response_model=DailyActivityKeyPageResponse)
async def get_issue_keys(
    issue_id: str,
    user: User,
    repository: IssuesRepositoryDep,
    alerts: UsageAlertsRepositoryDep,
    activity: IssueActivityRepositoryDep,
    start_date: date | None = None,
    end_date: date | None = None,
    limit: Limit = 20,
    offset: Offset = 0,
) -> DailyActivityKeyPageResponse:
    issue: Final = await authorized_issue(issue_id, user, repository, alerts)
    start, end = activity_range(issue, start_date, end_date)
    return await require_activity(activity).keys(issue.cohort, start, end, limit, offset)


@router.get("/{issue_id}/users", response_model=DailyActivityUserPageResponse)
async def get_issue_users(
    issue_id: str,
    user: User,
    repository: IssuesRepositoryDep,
    alerts: UsageAlertsRepositoryDep,
    activity: IssueActivityRepositoryDep,
    start_date: date | None = None,
    end_date: date | None = None,
    limit: Limit = 20,
    offset: Offset = 0,
) -> DailyActivityUserPageResponse:
    issue: Final = await authorized_issue(issue_id, user, repository, alerts)
    if issue.cohort.scope_type == "team":
        raise HTTPException(status_code=422, detail="Historical team aggregates do not retain user membership")
    start, end = activity_range(issue, start_date, end_date)
    return await require_activity(activity).users(issue.cohort, start, end, limit, offset)


@router.get("/{issue_id}/impact", response_model=IssueImpact)
async def get_issue_impact(
    issue_id: str,
    user: User,
    repository: IssuesRepositoryDep,
    alerts: UsageAlertsRepositoryDep,
    activity: IssueActivityRepositoryDep,
    start_date: date | None = None,
    end_date: date | None = None,
) -> IssueImpact:
    issue: Final = await authorized_issue(issue_id, user, repository, alerts)
    start, end = activity_range(issue, start_date, end_date)
    return await require_activity(activity).impact(issue.cohort, start, end)
