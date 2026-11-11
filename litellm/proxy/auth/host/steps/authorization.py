"""The `common_checks` steps the Python auth host still runs for gateway-auth-core's authorization stage. The
core runs the stage in order and decides the team blocked flag, the unpriced model block, the trace id gates,
the client-side tags rule, `enforce_user_param`, the proxy budget, and the organization and route checks. The steps here raise the proxy's own
error.

TODO: moves to gateway-auth-core one check at a time.

Blocked on:
- model access reads access groups and model aliases from the Python router
- guardrail, vector store and tool grants are read from Python-owned settings

The core decides the budget checks itself. This module answers their spend reads from the Python-owned spend
counters and sends their alerts through `ProxyLogging`, and those two stay here until the counters and alerting
move to Rust.

After the move:
- each check becomes a read of the rows and limits it needs and a comparison in the core
- this module, its `RustAuthStep`s and their `answer.py` cases are deleted once the last check leaves"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Final, Literal, assert_never

import litellm
from litellm.proxy._types import CallInfo, Litellm_EntityType
from litellm.proxy.auth import auth_checks
from litellm.proxy.auth.auth_utils import request_dispatched_to_pass_through_endpoint
from litellm.proxy.auth.route_checks import RouteChecks, RouteFacts, route_facts
from litellm.proxy.spend_tracking.budget_reservation import get_budget_window_start
from litellm.redis.stores.spend_counters import (
    end_user_counter,
    key_window_counter,
    model_access_group_counter,
    org_counter,
    tag_counter,
    team_counter,
    team_member_counter,
    team_window_counter,
    user_counter,
)
from litellm.utils import get_utc_datetime

if TYPE_CHECKING:
    from fastapi import Request

    from litellm.models.budget import LiteLLM_BudgetTable
    from litellm.models.end_user import LiteLLM_EndUserTable
    from litellm.models.team import BudgetLimitEntry, LiteLLM_TeamTable
    from litellm.models.team_membership import LiteLLM_TeamMembership
    from litellm.models.user import LiteLLM_UserTable
    from litellm.proxy._types import LiteLLM_ProjectTableCachedObj
    from litellm.proxy.utils import ProxyLogging
    from litellm.router import Router
    from litellm.types.proxy.auth.user_api_key_auth import UserAPIKeyAuth


SpendScope = Literal[
    "team", "team_window", "key_window", "organization", "tag", "model_access_group", "user", "team_member", "end_user"
]
AlertKind = Literal["team", "team_soft_budget", "organization", "user", "team_member"]
BudgetWindowFacts = tuple[str, float]


@dataclass(frozen=True, slots=True)
class KeyBudgetFacts:
    token: str
    org_id: str | None
    windows: tuple[BudgetWindowFacts, ...]


@dataclass(frozen=True, slots=True)
class TeamBudgetFacts:
    team_id: str
    max_budget: float | None
    spend: float | None
    soft_budget: float | None
    organization_id: str | None
    windows: tuple[BudgetWindowFacts, ...]


@dataclass(frozen=True, slots=True)
class UserBudgetFacts:
    user_id: str
    max_budget: float | None
    spend: float | None


@dataclass(frozen=True, slots=True)
class EndUserBudgetFacts:
    end_user_id: str
    max_budget: float | None
    spend: float | None


@dataclass(frozen=True, slots=True)
class BudgetFacts:
    """The budget fields `budgets::check` decides on. python-bridge reads it by attribute."""

    key: KeyBudgetFacts | None
    team: TeamBudgetFacts | None
    user: UserBudgetFacts | None
    end_user: EndUserBudgetFacts | None
    apply_user_budget_to_team_keys: bool
    is_info_route: bool
    fail_closed_budget_enforcement: bool
    now: float


@dataclass(frozen=True, slots=True)
class MemberBudget:
    """The team member's recorded spend, their own budget row as `(max_budget, temp_budget_increase,
    temp_budget_expiry)`, and the team's default member ceiling."""

    spend: float
    budget: tuple[float | None, float | None, float | None] | None
    team_default: float | None


def epoch(moment: datetime | None) -> float | None:
    if moment is None:
        return None
    return (moment if moment.tzinfo is not None else moment.replace(tzinfo=timezone.utc)).timestamp()


def member_budget(membership: LiteLLM_TeamMembership | None, team_default: LiteLLM_BudgetTable | None) -> MemberBudget:
    row: Final = membership.litellm_budget_table if membership is not None else None
    return MemberBudget(
        spend=(membership.spend if membership is not None else 0.0) or 0.0,
        budget=None if row is None else (row.max_budget, row.temp_budget_increase, epoch(row.temp_budget_expiry)),
        team_default=team_default.max_budget if team_default is not None else None,
    )


def _window(entry: BudgetLimitEntry | Mapping[str, object]) -> BudgetWindowFacts:
    window: Final = entry if isinstance(entry, Mapping) else entry.model_dump()
    return str(window["budget_duration"]), float(window["max_budget"])  # pyright: ignore[reportArgumentType]  # BudgetLimitEntry types max_budget as float


def _windows(limits: Sequence[BudgetLimitEntry | Mapping[str, object]] | None) -> tuple[BudgetWindowFacts, ...]:
    return tuple(_window(entry) for entry in limits or ())


def _window_entry(
    limits: Sequence[BudgetLimitEntry | Mapping[str, object]] | None, duration: str
) -> BudgetLimitEntry | Mapping[str, object] | None:
    return next((entry for entry in limits or () if _window(entry)[0] == duration), None)


@dataclass(frozen=True, slots=True)
class AuthorizeFacts:
    """What the core reads about the request before the stage runs. python-bridge reads it by attribute."""

    route: str
    method: str | None
    is_llm_api_route: bool
    dispatched_to_pass_through: bool
    has_model: bool
    skip_budget_checks: bool
    user_id: str | None
    agent_id: str | None
    team_id: str | None
    team_blocked: bool
    team_requires_trace_id: bool
    enforce_user_param: bool
    has_user_param: bool
    reject_clientside_metadata_tags: bool
    has_client_tags: bool
    block_unpriced_models: bool
    max_budget: float
    global_proxy_spend: float | None
    route_access: RouteFacts
    budgets: BudgetFacts


@dataclass(frozen=True, slots=True)
class Authorization:
    """The request `common_checks` authorizes and the rows its caller already loaded."""

    request: Request
    request_body: dict  # mutable-ok: the tag merge writes the key's tags into the body the tag budget reads
    route: str
    model: str | list[str] | None
    skip_budget_checks: bool
    valid_token: UserAPIKeyAuth | None
    team_object: LiteLLM_TeamTable | None
    user_object: LiteLLM_UserTable | None
    end_user_object: LiteLLM_EndUserTable | None
    project_object: LiteLLM_ProjectTableCachedObj | None
    general_settings: dict
    global_proxy_spend: float | None
    llm_router: Router | None
    proxy_logging_obj: ProxyLogging

    def facts(self) -> AuthorizeFacts:
        token: Final = self.valid_token
        team: Final = self.team_object
        return AuthorizeFacts(
            route=self.route,
            method=self.request.method,
            is_llm_api_route=RouteChecks.is_llm_api_route(route=self.route),
            dispatched_to_pass_through=request_dispatched_to_pass_through_endpoint(self.request),
            has_model=bool(self.model),
            skip_budget_checks=self.skip_budget_checks,
            user_id=None if token is None else token.user_id,
            agent_id=None if token is None else token.agent_id,
            team_id=None if team is None else team.team_id,
            team_blocked=team is not None and team.blocked is True,
            team_requires_trace_id=team is not None and auth_checks.team_requires_trace_id(team),
            enforce_user_param=bool(self.general_settings.get("enforce_user_param", False)),
            has_user_param="user" in self.request_body,
            reject_clientside_metadata_tags=bool(self.general_settings.get("reject_clientside_metadata_tags", False)),
            has_client_tags=auth_checks.has_client_metadata_tags(self.request, self.request_body),
            block_unpriced_models=litellm.block_requests_for_models_without_pricing,
            max_budget=litellm.max_budget,
            global_proxy_spend=self.global_proxy_spend,
            route_access=route_facts(self.route, self.request, self.request_body, token, self.user_object),
            budgets=self._budget_facts(),
        )

    def _budget_facts(self) -> BudgetFacts:
        from litellm.proxy.proxy_server import fail_closed_budget_enforcement_enabled  # noqa: PLC0415  # proxy global

        token: Final = self.valid_token
        team: Final = self.team_object
        user: Final = self.user_object
        end_user: Final = self.end_user_object
        end_user_budget: Final = end_user.litellm_budget_table if end_user is not None else None
        return BudgetFacts(
            key=None
            if token is None
            else KeyBudgetFacts(token=str(token.token), org_id=token.org_id, windows=_windows(token.budget_limits)),
            team=None
            if team is None or team.team_id is None
            else TeamBudgetFacts(
                team_id=team.team_id,
                max_budget=team.max_budget,
                spend=team.spend,
                soft_budget=team.soft_budget,
                organization_id=team.organization_id,
                windows=_windows(team.budget_limits),
            ),
            user=None
            if user is None
            else UserBudgetFacts(user_id=user.user_id, max_budget=user.max_budget, spend=user.spend),
            end_user=None
            if end_user is None or end_user_budget is None
            else EndUserBudgetFacts(
                end_user_id=end_user.user_id, max_budget=end_user_budget.max_budget, spend=end_user.spend
            ),
            apply_user_budget_to_team_keys=self.general_settings.get("apply_user_budget_to_team_keys") is True,
            is_info_route=RouteChecks.is_info_route(self.route),
            fail_closed_budget_enforcement=fail_closed_budget_enforcement_enabled(),
            now=get_utc_datetime().timestamp(),
        )

    def unpriced_models(self) -> Sequence[str]:
        return auth_checks.unpriced_models_in_request(self.model, self.llm_router)

    def has_trace_id(self, headers_only: bool) -> bool:
        return auth_checks.request_has_trace_id(self.request_body, self.request, headers_only=headers_only)

    def merge_budget_tags(self) -> None:
        auth_checks.merge_budget_tags(self.request, self.request_body, self.route, self.valid_token)

    async def check_team_model_access(self, membership_read: bool, membership: LiteLLM_TeamMembership | None) -> None:
        await auth_checks.check_team_model_access_for_request(
            self.model,
            self.team_object,
            self.valid_token,
            self.llm_router,
            self.proxy_logging_obj,
            membership,
            membership_read,
        )

    async def check_model_grants(self) -> None:
        await auth_checks.check_model_grants_for_request(
            self.model,
            self.request_body,
            self.valid_token,
            self.user_object,
            self.end_user_object,
            self.team_object,
            self.llm_router,
            self.proxy_logging_obj,
        )

    async def check_project(self) -> None:
        await auth_checks.check_project_for_request(
            self.model,
            self.project_object,
            self.valid_token,
            self.llm_router,
            self.skip_budget_checks,
            self.proxy_logging_obj,
        )

    async def stamp_access_groups(
        self, membership_read: bool, membership: LiteLLM_TeamMembership | None
    ) -> Sequence[str]:
        return await auth_checks.stamp_access_groups_for_request(
            self.model,
            self.valid_token,
            self.team_object,
            self.project_object,
            self.llm_router,
            self.proxy_logging_obj,
            membership,
            membership_read,
        )

    async def spend(
        self,
        scope: SpendScope,
        entity_id: str,
        team_id: str | None,
        window: str | None,
        recorded: float,
        max_budget: float,
    ) -> float:
        from litellm.proxy.proxy_server import get_current_spend  # noqa: PLC0415  # proxy global

        match scope:
            case "team_window" | "key_window":
                limits: Final = (
                    (self.team_object.budget_limits if self.team_object is not None else None)
                    if scope == "team_window"
                    else (self.valid_token.budget_limits if self.valid_token is not None else None)
                )
                duration: Final = window or ""
                return await get_current_spend(
                    counter_key=(team_window_counter if scope == "team_window" else key_window_counter)(
                        entity_id, duration
                    ),
                    fallback_spend=0.0,
                    max_budget=max_budget,
                    window_entity_type="Team" if scope == "team_window" else "Key",
                    window_entity_id=entity_id,
                    window_duration=duration,
                    window_start=get_budget_window_start(_window_entry(limits, duration)),
                )
            case "team_member":
                return await get_current_spend(
                    counter_key=team_member_counter(entity_id, team_id or ""),
                    fallback_spend=recorded,
                    max_budget=max_budget,
                )
            case "team" | "organization" | "user":
                counter: Final = {"team": team_counter, "organization": org_counter, "user": user_counter}[scope]
                return await get_current_spend(
                    counter_key=counter(entity_id), fallback_spend=recorded, max_budget=max_budget
                )
            case "tag" | "model_access_group" | "end_user":
                authoritative: Final = {
                    "tag": tag_counter,
                    "model_access_group": model_access_group_counter,
                    "end_user": end_user_counter,
                }[scope]
                return await get_current_spend(
                    counter_key=authoritative(entity_id),
                    fallback_spend=recorded,
                    max_budget=max_budget,
                    fallback_authoritative=True,
                )
            case unreachable:
                assert_never(unreachable)

    async def budget_alert(
        self, kind: AlertKind, org_id: str | None, spend: float | None, max_budget: float | None
    ) -> None:
        token: Final = self.valid_token
        team: Final = self.team_object
        user: Final = self.user_object
        match kind:
            case "team_soft_budget":
                await auth_checks._team_soft_budget_check(  # pyright: ignore[reportPrivateUsage]  # sends the soft budget alert to the team's configured emails
                    team_object=team, valid_token=token, proxy_logging_obj=self.proxy_logging_obj
                )
            case "team_member":
                if token is None or token.user_id is None or team is None:
                    return
                auth_checks.team_member_max_budget_alert_check(
                    team_id=team.team_id,
                    team_alias=team.team_alias,
                    team_metadata=team.metadata,
                    organization_id=team.organization_id,
                    user_id=token.user_id,
                    user_email=user.user_email if user is not None else None,
                    proxy_logging_obj=self.proxy_logging_obj,
                    spend=spend or 0.0,
                    max_budget=max_budget or 0.0,
                )
            case "team" | "organization" | "user":
                alert_type, call_info = self._budget_call_info(kind, org_id, spend or 0.0, max_budget or 0.0)
                asyncio.create_task(self.proxy_logging_obj.budget_alerts(type=alert_type, user_info=call_info))
            case unreachable:
                assert_never(unreachable)

    def _budget_call_info(
        self, kind: Literal["team", "organization", "user"], org_id: str | None, spend: float, max_budget: float
    ) -> tuple[Literal["team_budget", "organization_budget", "user_budget"], CallInfo]:
        token: Final = self.valid_token
        match kind:
            case "user":
                user: Final = self.user_object
                return "user_budget", CallInfo(
                    spend=spend,
                    max_budget=max_budget,
                    user_id=user.user_id if user is not None else None,
                    user_email=user.user_email if user is not None else None,
                    event_group=Litellm_EntityType.USER,
                )
            case "team" | "organization":
                return ("team_budget" if kind == "team" else "organization_budget"), CallInfo(
                    token=token.token if token is not None else None,
                    spend=spend,
                    max_budget=max_budget,
                    user_id=token.user_id if token is not None else None,
                    team_id=token.team_id if token is not None else None,
                    team_alias=token.team_alias if token is not None else None,
                    organization_id=org_id if kind == "organization" else (token.org_id if token else None),
                    event_group=Litellm_EntityType.TEAM if kind == "team" else Litellm_EntityType.ORGANIZATION,
                )
            case unreachable:
                assert_never(unreachable)

    def check_guardrail_modification(self) -> None:
        auth_checks.check_guardrail_modification_for_request(self.request_body, self.team_object)

    async def check_object_access(self) -> None:
        await auth_checks.check_object_access_for_request(
            self.request_body,
            self.general_settings,
            self.route,
            self.valid_token,
            self.team_object,
            self.user_object,
            self.llm_router,
        )
