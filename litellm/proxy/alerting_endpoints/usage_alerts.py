"""Performance rules and read-only detection evidence."""

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
    AlertModelList,
    AlertRule,
    AlertRuleInput,
    AlertRuleList,
    PerformanceSignal,
    PerformanceSignalList,
    UsageAlertMetric,
    UsageAlertRuleKind,
    UsageAlertScope,
    UsageAlertStatus,
)

router: Final = APIRouter(prefix="/alerts", tags=["alerts"])


def require_repository(repository: UsageAlertStore | None) -> UsageAlertStore:
    if repository is None:
        raise HTTPException(status_code=503, detail="Alert signals require a database connection")
    return repository


User: TypeAlias = Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)]
Repository: TypeAlias = UsageAlertsRepositoryDep
Limit: TypeAlias = Annotated[int, Query(ge=1, le=100)]
Offset: TypeAlias = Annotated[int, Query(ge=0, le=100000)]
ScopeId: TypeAlias = Annotated[str | None, Query(min_length=1, max_length=1024)]


def _allows_write(grant: str, route: str) -> bool:
    if grant == "info_routes":
        return False
    if route == grant or route.startswith(grant + "/"):
        return True
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


def require_notification_access(user: UserAPIKeyAuth, data: AlertRuleInput, existing: AlertRule | None = None) -> None:
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
        raise HTTPException(status_code=403, detail="Read-only users cannot manage alerts")
    if user.user_role not in (
        LitellmUserRoles.INTERNAL_USER,
        LitellmUserRoles.INTERNAL_USER_VIEW_ONLY,
        LitellmUserRoles.ORG_ADMIN,
        LitellmUserRoles.TEAM,
    ):
        raise HTTPException(status_code=403, detail="Alert signals require a management role")
    if write and user.user_role == LitellmUserRoles.INTERNAL_USER_VIEW_ONLY:
        raise HTTPException(status_code=403, detail="Read-only users cannot manage alerts")
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


@router.get("/rules", response_model=AlertRuleList)
async def list_alert_rules(
    user: User,
    repository: Repository,
    scope_type: UsageAlertScope | None = None,
    scope_id: ScopeId = None,
    limit: Limit = 50,
    offset: Offset = 0,
    kind: UsageAlertRuleKind = "all",
) -> AlertRuleList:
    store: Final = require_repository(repository)
    await require_scope(user, store, scope_type, scope_id, write=False)
    return await store.list_rules(scope_type, scope_id, limit, offset, kind)


@router.get("/models", response_model=AlertModelList)
async def list_alert_models(
    user: User,
    repository: Repository,
    search: ScopeId = None,
    cursor: ScopeId = None,
    limit: Limit = 20,
) -> AlertModelList:
    store: Final = require_repository(repository)
    await require_scope(user, store, "model", None, write=False)
    return await store.list_reported_models(search, cursor, limit)


@router.post("/rules", response_model=AlertRule, status_code=201)
async def create_alert_rule(data: AlertRuleInput, user: User, repository: Repository) -> AlertRule:
    """Create a daily performance rule for a scope the caller can manage."""
    store: Final = require_repository(repository)
    require_write_access(user, "/alerts/rules")
    await require_scope(user, store, data.scope_type, data.scope_id, write=True)
    require_notification_access(user, data)
    saved: Final = await store.save_rule(
        data, user.user_id, overrides_defaults=user.user_role == LitellmUserRoles.PROXY_ADMIN
    )
    if saved is None:
        raise HTTPException(status_code=409, detail="Alert rule could not be saved")
    return saved


@router.get("/rules/{rule_id}", response_model=AlertRule)
async def get_alert_rule(rule_id: str, user: User, repository: Repository) -> AlertRule:
    store: Final = require_repository(repository)
    rule: Final = await store.get_rule(rule_id)
    if rule is None:
        raise HTTPException(status_code=404, detail="Alert rule not found")
    await require_scope(user, store, rule.scope_type, rule.scope_id, write=False)
    return rule


@router.put("/rules/{rule_id}", response_model=AlertRule)
async def update_alert_rule(rule_id: str, data: AlertRuleInput, user: User, repository: Repository) -> AlertRule:
    store: Final = require_repository(repository)
    require_write_access(user, f"/alerts/rules/{rule_id}")
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


@router.delete("/rules/{rule_id}", status_code=204)
async def delete_alert_rule(rule_id: str, user: User, repository: Repository) -> Response:
    store: Final = require_repository(repository)
    require_write_access(user, f"/alerts/rules/{rule_id}")
    rule: Final = await store.get_rule(rule_id)
    if rule is None:
        raise HTTPException(status_code=404, detail="Alert rule not found")
    await require_scope(user, store, rule.scope_type, rule.scope_id, write=True)
    require_notification_access(user, rule, rule)
    if not await store.delete_rule(rule_id, existing=rule):
        raise HTTPException(status_code=409, detail="Alert rule changed while being deleted. Refresh and try again")
    return Response(status_code=204)


@router.get("/signals", response_model=PerformanceSignalList)
async def list_performance_signals(
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
) -> PerformanceSignalList:
    store: Final = require_repository(repository)
    await require_scope(user, store, scope_type, scope_id, write=False)
    if start_date is not None and end_date is not None and start_date > end_date:
        raise HTTPException(status_code=422, detail="start_date must not be after end_date")
    return await store.list_alerts(scope_type, scope_id, metric, status, start_date, end_date, limit, offset)


@router.get("/signals/{signal_id}", response_model=PerformanceSignal)
async def get_performance_signal(signal_id: str, user: User, repository: Repository) -> PerformanceSignal:
    store: Final = require_repository(repository)
    alert: Final = await store.get_alert(signal_id)
    if alert is None:
        raise HTTPException(status_code=404, detail="Alert signal not found")
    await require_scope(user, store, alert.scope_type, alert.scope_id, write=False)
    return alert
