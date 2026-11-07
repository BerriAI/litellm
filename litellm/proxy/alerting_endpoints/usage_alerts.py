"""Public management API for daily usage alert rules and incidents."""

from datetime import date
from typing import Annotated, Final, TypeAlias

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from pydantic import BaseModel, TypeAdapter

from litellm.models.usage_alerts import UsageAlertStore
from litellm.proxy._types import LiteLLMRoutes, UserAPIKeyAuth
from litellm.proxy.auth.route_checks import RouteChecks
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.deps.repositories import UsageAlertsRepositoryDep
from litellm.repositories.daily_activity_buckets import api_key_from_bucket
from litellm.types.proxy.auth.user_roles import LitellmUserRoles
from litellm.types.proxy.usage_alerts import (
    UsageAlert,
    UsageAlertGroupList,
    UsageAlertGroupUpdate,
    UsageAlertGroupUpdateResult,
    UsageAlertList,
    UsageAlertMetric,
    UsageAlertModelList,
    UsageAlertRule,
    UsageAlertRuleInput,
    UsageAlertRuleKind,
    UsageAlertRuleList,
    UsageAlertScope,
    UsageAlertStatus,
    UsageAlertUpdate,
)

router: Final = APIRouter(prefix="/usage", tags=["usage alerts"])


def require_repository(repository: UsageAlertStore | None) -> UsageAlertStore:
    if repository is None:
        raise HTTPException(status_code=503, detail="Usage alerts require a database connection")
    return repository


User: TypeAlias = Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)]
Repository: TypeAlias = UsageAlertsRepositoryDep
Limit: TypeAlias = Annotated[int, Query(ge=1, le=100)]
Offset: TypeAlias = Annotated[int, Query(ge=0, le=100000)]
ScopeId: TypeAlias = Annotated[str | None, Query(min_length=1, max_length=1024)]


def _allows_write(grant: str, route: str) -> bool:
    if grant == "info_routes":
        return False
    paths: Final = (
        TypeAdapter(tuple[str, ...]).validate_python(LiteLLMRoutes[grant].value)
        if grant in LiteLLMRoutes.__members__
        else (grant,)
    )
    return RouteChecks.check_route_access(route, paths)


class _WriteGrants(BaseModel, frozen=True):
    allowed_routes: tuple[str, ...] | None = None


def require_write_access(user: UserAPIKeyAuth, route: str) -> None:
    grants: Final = _WriteGrants.model_validate(user, from_attributes=True).allowed_routes
    if not grants:
        return
    if not any(_allows_write(grant, route) for grant in grants):
        raise HTTPException(status_code=403, detail="This credential does not grant alert write access")


def require_notification_access(
    user: UserAPIKeyAuth, data: UsageAlertRuleInput, existing: UsageAlertRule | None = None
) -> None:
    if user.user_role == LitellmUserRoles.PROXY_ADMIN:
        return
    if data.notification_channels or data.email_recipients:
        raise HTTPException(status_code=403, detail="Only proxy admins can configure notification delivery")
    if existing is not None and (existing.notification_channels or existing.overrides_defaults):
        raise HTTPException(status_code=403, detail="Only proxy admins can change operator-managed alert rules")


async def require_scope(
    user: UserAPIKeyAuth,
    repository: UsageAlertStore,
    scope_type: UsageAlertScope | None,
    scope_id: str | None,
    *,
    write: bool,
) -> None:
    if user.key_type == "llm_api" or (write and user.key_type == "read_only"):
        raise HTTPException(status_code=403, detail="A management-capable credential is required")
    if user.user_role == LitellmUserRoles.PROXY_ADMIN:
        return
    if user.user_role == LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY:
        if not write:
            return
        raise HTTPException(status_code=403, detail="Read-only users cannot manage usage alerts")
    if user.user_role not in (
        LitellmUserRoles.INTERNAL_USER,
        LitellmUserRoles.INTERNAL_USER_VIEW_ONLY,
        LitellmUserRoles.ORG_ADMIN,
        LitellmUserRoles.TEAM,
    ):
        raise HTTPException(status_code=403, detail="Usage alerts require a management role")
    if write and user.user_role == LitellmUserRoles.INTERNAL_USER_VIEW_ONLY:
        raise HTTPException(status_code=403, detail="Read-only users cannot manage usage alerts")
    if scope_id is None:
        raise HTTPException(status_code=403, detail="Specify a scope you can access")
    if scope_type == "user" and user.user_id is not None and scope_id == user.user_id:
        return
    if scope_type == "team":
        role: Final = await repository.team_role(scope_id, user.user_id) if user.user_id else None
        if role == "admin" or (
            not write and (role == "user" or (user.user_role == LitellmUserRoles.TEAM and user.team_id == scope_id))
        ):
            return
    if scope_type == "key":
        ownership: Final = await repository.key_ownership(api_key_from_bucket(scope_id))
        if ownership is not None:
            owner_id, team_id = ownership
            if user.user_id is not None and owner_id == user.user_id:
                return
            if user.user_id and team_id and await repository.team_role(team_id, user.user_id) == "admin":
                return
    raise HTTPException(status_code=403, detail="You cannot access this alert scope")


@router.get("/alert-rules", response_model=UsageAlertRuleList)
async def list_usage_alert_rules(
    user: User,
    repository: Repository,
    scope_type: UsageAlertScope | None = None,
    scope_id: ScopeId = None,
    limit: Limit = 50,
    offset: Offset = 0,
    kind: UsageAlertRuleKind = "all",
) -> UsageAlertRuleList:
    store: Final = require_repository(repository)
    await require_scope(user, store, scope_type, scope_id, write=False)
    return await store.list_rules(scope_type, scope_id, limit, offset, kind)


@router.get("/alert-groups", response_model=UsageAlertGroupList)
async def list_usage_alert_groups(
    user: User,
    repository: Repository,
    scope_type: UsageAlertScope | None = None,
    scope_id: ScopeId = None,
    status: UsageAlertStatus | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
    limit: Limit = 50,
    offset: Offset = 0,
) -> UsageAlertGroupList:
    store: Final = require_repository(repository)
    await require_scope(user, store, scope_type, scope_id, write=False)
    if start_date is not None and end_date is not None and start_date > end_date:
        raise HTTPException(status_code=422, detail="start_date must not be after end_date")
    return await store.list_groups(scope_type, scope_id, status, start_date, end_date, limit, offset)


@router.patch("/alert-groups", response_model=UsageAlertGroupUpdateResult)
async def update_usage_alert_group(
    data: UsageAlertGroupUpdate, user: User, repository: Repository
) -> UsageAlertGroupUpdateResult:
    store: Final = require_repository(repository)
    require_write_access(user, "/usage/alert-groups")
    await require_scope(user, store, data.scope_type, data.scope_id, write=True)
    return await store.set_group_status(data)


@router.get("/alert-models", response_model=UsageAlertModelList)
async def list_usage_alert_models(
    user: User,
    repository: Repository,
    search: ScopeId = None,
    cursor: ScopeId = None,
    limit: Limit = 20,
) -> UsageAlertModelList:
    store: Final = require_repository(repository)
    await require_scope(user, store, "model", None, write=False)
    return await store.list_reported_models(search, cursor, limit)


@router.post("/alert-rules", response_model=UsageAlertRule, status_code=201)
async def create_usage_alert_rule(data: UsageAlertRuleInput, user: User, repository: Repository) -> UsageAlertRule:
    """Create a daily usage alert rule for a scope the caller can manage."""
    store: Final = require_repository(repository)
    require_write_access(user, "/usage/alert-rules")
    await require_scope(user, store, data.scope_type, data.scope_id, write=True)
    require_notification_access(user, data)
    saved: Final = await store.save_rule(
        data, user.user_id, overrides_defaults=user.user_role == LitellmUserRoles.PROXY_ADMIN
    )
    if saved is None:
        raise HTTPException(status_code=409, detail="Alert rule could not be saved")
    return saved


@router.get("/alert-rules/{rule_id}", response_model=UsageAlertRule)
async def get_usage_alert_rule(rule_id: str, user: User, repository: Repository) -> UsageAlertRule:
    store: Final = require_repository(repository)
    rule: Final = await store.get_rule(rule_id)
    if rule is None:
        raise HTTPException(status_code=404, detail="Alert rule not found")
    await require_scope(user, store, rule.scope_type, rule.scope_id, write=False)
    return rule


@router.put("/alert-rules/{rule_id}", response_model=UsageAlertRule)
async def update_usage_alert_rule(
    rule_id: str, data: UsageAlertRuleInput, user: User, repository: Repository
) -> UsageAlertRule:
    store: Final = require_repository(repository)
    require_write_access(user, f"/usage/alert-rules/{rule_id}")
    rule: Final = await store.get_rule(rule_id)
    if rule is None:
        raise HTTPException(status_code=404, detail="Alert rule not found")
    await require_scope(user, store, rule.scope_type, rule.scope_id, write=True)
    if rule.is_default and (rule.scope_type, rule.scope_id, rule.metric, rule.direction) != (
        data.scope_type,
        data.scope_id,
        data.metric,
        data.direction,
    ):
        raise HTTPException(
            status_code=422, detail="Create a custom rule to change a built-in rule's scope, metric, or direction"
        )
    await require_scope(user, store, data.scope_type, data.scope_id, write=True)
    require_notification_access(user, data, rule)
    saved: Final = await store.save_rule(data, user.user_id, existing=rule, overrides_defaults=rule.overrides_defaults)
    if saved is None:
        raise HTTPException(status_code=409, detail="Alert rule changed while being updated. Refresh and try again")
    return saved


@router.delete("/alert-rules/{rule_id}", status_code=204)
async def delete_usage_alert_rule(rule_id: str, user: User, repository: Repository) -> Response:
    store: Final = require_repository(repository)
    require_write_access(user, f"/usage/alert-rules/{rule_id}")
    rule: Final = await store.get_rule(rule_id)
    if rule is None:
        raise HTTPException(status_code=404, detail="Alert rule not found")
    await require_scope(user, store, rule.scope_type, rule.scope_id, write=True)
    require_notification_access(user, rule, rule)
    if not await store.delete_rule(rule_id, existing=rule):
        raise HTTPException(status_code=409, detail="Alert rule changed while being deleted. Refresh and try again")
    return Response(status_code=204)


@router.get("/alerts", response_model=UsageAlertList)
async def list_usage_alerts(
    user: User,
    repository: Repository,
    scope_type: UsageAlertScope | None = None,
    scope_id: ScopeId = None,
    metric: UsageAlertMetric | None = None,
    status: UsageAlertStatus | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
    limit: Limit = 50,
    offset: Offset = 0,
) -> UsageAlertList:
    store: Final = require_repository(repository)
    await require_scope(user, store, scope_type, scope_id, write=False)
    if start_date is not None and end_date is not None and start_date > end_date:
        raise HTTPException(status_code=422, detail="start_date must not be after end_date")
    return await store.list_alerts(scope_type, scope_id, metric, status, start_date, end_date, limit, offset)


@router.get("/alerts/{alert_id}", response_model=UsageAlert)
async def get_usage_alert(alert_id: str, user: User, repository: Repository) -> UsageAlert:
    store: Final = require_repository(repository)
    alert: Final = await store.get_alert(alert_id)
    if alert is None:
        raise HTTPException(status_code=404, detail="Usage alert not found")
    await require_scope(user, store, alert.scope_type, alert.scope_id, write=False)
    return alert


@router.patch("/alerts/{alert_id}", response_model=UsageAlert)
async def update_usage_alert(alert_id: str, data: UsageAlertUpdate, user: User, repository: Repository) -> UsageAlert:
    store: Final = require_repository(repository)
    require_write_access(user, f"/usage/alerts/{alert_id}")
    alert: Final = await store.get_alert(alert_id)
    if alert is None:
        raise HTTPException(status_code=404, detail="Usage alert not found")
    await require_scope(user, store, alert.scope_type, alert.scope_id, write=True)
    updated: Final = await store.set_status(alert_id, data.status)
    if updated is None:
        raise HTTPException(status_code=404, detail="Usage alert not found")
    return updated
