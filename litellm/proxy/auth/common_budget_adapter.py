import asyncio
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Final, Literal, TypeAlias

from pydantic import TypeAdapter
from typing_extensions import ReadOnly, TypedDict

import litellm
from litellm._auth import common_budgets
from litellm._auth.common_budgets import AlertFacts, BudgetDecision, BudgetFacts, MemberLimitFacts
from litellm.constants import EMAIL_BUDGET_ALERT_MAX_SPEND_ALERT_PERCENTAGE
from litellm.models.budget import LiteLLM_BudgetTable
from litellm.proxy._types import (
    CallInfo,
    LiteLLM_EndUserTable,
    Litellm_EntityType,
    LiteLLM_ProjectTableCachedObj,
    LiteLLM_TeamTable,
    LiteLLM_UserTable,
    UserAPIKeyAuth,
)
from litellm.proxy.utils import ProxyLogging

_METADATA: Final = TypeAdapter(Mapping[str, object])
_ITEMS: Final = TypeAdapter(tuple[object, ...])
AlertKind: TypeAlias = Literal[
    "token_budget",
    "user_budget",
    "soft_budget",
    "max_budget_alert",
    "team_budget",
    "organization_budget",
    "proxy_budget",
    "projected_limit_exceeded",
    "project_budget",
]
_ALERT: Final = TypeAdapter[AlertKind](AlertKind)


def metadata(raw: object) -> Mapping[str, object]:
    return _METADATA.validate_python(raw or {})


@dataclass(frozen=True, slots=True)
class CounterRead:
    key: str
    fallback: float
    max_budget: float | None
    authoritative: bool = False
    window_entity_type: str | None = None
    window_entity_id: str | None = None
    window_duration: str | None = None
    window_start: datetime | None = None


class _CounterOptions(TypedDict, total=False):
    fallback_authoritative: ReadOnly[bool]
    window_entity_type: ReadOnly[str | None]
    window_entity_id: ReadOnly[str | None]
    window_duration: ReadOnly[str | None]
    window_start: ReadOnly[datetime | None]


async def read_counter(counter: CounterRead) -> float:
    from litellm.proxy.proxy_server import get_current_spend

    authoritative: Final[_CounterOptions] = {"fallback_authoritative": True} if counter.authoritative else {}
    window: Final[_CounterOptions] = (
        {
            "window_entity_type": counter.window_entity_type,
            "window_entity_id": counter.window_entity_id,
            "window_duration": counter.window_duration,
            "window_start": counter.window_start,
        }
        if counter.window_entity_type is not None
        else {}
    )
    options: Final[_CounterOptions] = {**authoritative, **window}
    return await get_current_spend(
        counter_key=counter.key,
        fallback_spend=counter.fallback,
        max_budget=counter.max_budget,
        **options,
    )


async def read_budget(
    facts: BudgetFacts,
    counter: CounterRead,
    reader: Callable[[CounterRead], Awaitable[float]] = read_counter,
) -> tuple[BudgetFacts, BudgetDecision]:
    initial: Final = common_budgets.check(facts)
    if not initial.read_spend:
        return facts, initial
    snapshot: Final = replace(facts, spend=await reader(counter))
    return snapshot, common_budgets.check(snapshot)


def emit_alert(logging: ProxyLogging, kind: str, info: CallInfo) -> None:
    asyncio.create_task(logging.budget_alerts(type=_ALERT.validate_python(kind), user_info=info))


def key_alert_info(
    key: UserAPIKeyAuth, spend: float | None, limit: float | None, entity: Litellm_EntityType
) -> CallInfo:
    return CallInfo(
        token=key.token,
        spend=spend if spend is not None else 0.0,
        max_budget=limit,
        user_id=key.user_id,
        team_id=key.team_id,
        team_alias=key.team_alias,
        organization_id=key.org_id,
        event_group=entity,
    )


def raise_budget(facts: BudgetFacts, entity: Litellm_EntityType, identity: str | None, message: str | None) -> None:
    assert facts.spend is not None and facts.limit is not None
    raise litellm.BudgetExceededError(
        current_cost=facts.spend,
        max_budget=facts.limit,
        message=message,
        entity_type=entity.value,
        entity_id=identity,
    )


async def team_max_budget_check(
    team_object: LiteLLM_TeamTable | None,
    valid_token: UserAPIKeyAuth | None,
    proxy_logging_obj: ProxyLogging,
) -> None:
    if team_object is None:
        return
    facts, result = await read_budget(
        BudgetFacts("Team", team_object.max_budget, token_present=valid_token is not None),
        CounterRead(f"spend:team:{team_object.team_id}", team_object.spend or 0.0, team_object.max_budget),
    )
    if result.alert is not None and valid_token is not None:
        emit_alert(
            proxy_logging_obj,
            result.alert,
            key_alert_info(valid_token, facts.spend, facts.limit, Litellm_EntityType.TEAM),
        )
    if result.denied:
        raise_budget(
            facts,
            Litellm_EntityType.TEAM,
            team_object.team_id,
            f"Budget has been exceeded! Team={team_object.team_id} Current cost: {facts.spend}, Max budget: {facts.limit}",
        )


async def virtual_key_max_budget_check(
    valid_token: UserAPIKeyAuth,
    proxy_logging_obj: ProxyLogging,
    user_obj: LiteLLM_UserTable | None = None,
) -> None:
    from litellm.proxy.auth.auth_checks import key_budget_exceeded_error, send_token_budget_alert

    raw_percentage: Final[object] = litellm.budget_exceeded_throttle_percentage
    percentage: Final = (
        float(raw_percentage)
        if isinstance(raw_percentage, (int, float)) and not isinstance(raw_percentage, bool)
        else None
    )
    facts, result = await read_budget(
        BudgetFacts(
            "Key",
            valid_token.max_budget,
            throttle_percentage=percentage,
            throttle_opt_in=metadata(valid_token.metadata).get("throttle_on_budget_exceeded") is True,  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]  # legacy metadata is validated by metadata()
            tpm_limit=valid_token.tpm_limit,
            rpm_limit=valid_token.rpm_limit,
        ),
        CounterRead(f"spend:key:{valid_token.token}", valid_token.spend or 0.0, valid_token.max_budget),
    )
    if result.alert is not None and facts.spend is not None:
        send_token_budget_alert(valid_token, facts.spend, proxy_logging_obj, user_obj)
    if result.throttle_percentage is not None:
        valid_token.budget_throttle_pct = result.throttle_percentage  # rebind-ok: native request-local throttle
    if result.denied and facts.spend is not None and facts.limit is not None:
        raise key_budget_exceeded_error(valid_token, facts.spend, facts.limit)


async def project_max_budget_check(
    project_object: LiteLLM_ProjectTableCachedObj | None,
    valid_token: UserAPIKeyAuth | None,
    proxy_logging_obj: ProxyLogging,
) -> None:
    from litellm.proxy.auth.auth_checks import project_spend_counter_key

    if project_object is None:
        return
    limit: Final = (
        project_object.litellm_budget_table.max_budget if project_object.litellm_budget_table is not None else None
    )
    facts, result = await read_budget(
        BudgetFacts("Project", limit, token_present=valid_token is not None),
        CounterRead(project_spend_counter_key(project_object.project_id), project_object.spend or 0.0, limit),
    )
    if result.alert is not None and valid_token is not None:
        emit_alert(
            proxy_logging_obj,
            result.alert,
            key_alert_info(valid_token, facts.spend, limit, Litellm_EntityType.PROJECT),
        )
    if result.denied:
        raise_budget(
            facts,
            Litellm_EntityType.PROJECT,
            project_object.project_id,
            f"Budget has been exceeded! Project={project_object.project_id} Current cost: {facts.spend}, Max budget: {limit}",
        )


async def user_max_budget_check(
    user_object: LiteLLM_UserTable | None,
    team_object: LiteLLM_TeamTable | None,
    general_settings: Mapping[str, object],
    proxy_logging_obj: ProxyLogging,
) -> None:
    if user_object is None:
        return
    facts, result = await read_budget(
        BudgetFacts(
            "User",
            user_object.max_budget,
            team_key=team_object is not None and team_object.team_id is not None,  # pyright: ignore[reportUnnecessaryComparison]  # retain the legacy None guard for constructed cache rows
            apply_user_budget_to_team_keys=general_settings.get("apply_user_budget_to_team_keys") is True,
        ),
        CounterRead(f"spend:user:{user_object.user_id}", user_object.spend or 0.0, user_object.max_budget),
    )
    if result.alert is not None:
        emit_alert(
            proxy_logging_obj,
            result.alert,
            CallInfo(
                spend=facts.spend if facts.spend is not None else 0.0,
                max_budget=facts.limit,
                user_id=user_object.user_id,
                user_email=user_object.user_email,
                event_group=Litellm_EntityType.USER,
            ),
        )
    if result.denied:
        raise_budget(
            facts,
            Litellm_EntityType.USER,
            user_object.user_id,
            f"ExceededBudget: User={user_object.user_id} over budget. Spend={facts.spend}, Budget={facts.limit}",
        )


async def check_end_user_budget(end_user_obj: LiteLLM_EndUserTable, route: str) -> None:
    from litellm.proxy.auth.route_checks import RouteChecks

    limit: Final = (
        end_user_obj.litellm_budget_table.max_budget if end_user_obj.litellm_budget_table is not None else None
    )
    facts, result = await read_budget(
        BudgetFacts("EndUser", limit, info_route=RouteChecks.is_info_route(route)),
        CounterRead(f"spend:end_user:{end_user_obj.user_id}", end_user_obj.spend or 0.0, limit, authoritative=True),
    )
    if result.denied:
        raise_budget(
            facts,
            Litellm_EntityType.END_USER,
            end_user_obj.user_id,
            f"ExceededBudget: End User={end_user_obj.user_id} over budget. Spend={facts.spend}, Budget={facts.limit}",
        )


def global_proxy_budget_check(global_proxy_spend: float | None, skip_budget_checks: bool, route: str) -> None:
    from litellm.proxy._types import MODEL_LIST_ROUTES
    from litellm.proxy.auth.route_checks import RouteChecks

    facts: Final = BudgetFacts(
        "Proxy",
        litellm.max_budget,
        global_proxy_spend,
        skip=skip_budget_checks,
        llm_route=RouteChecks.is_llm_api_route(route=route),
        model_list_route=route in MODEL_LIST_ROUTES,
    )
    if common_budgets.check(facts).denied:
        raise_budget(facts, Litellm_EntityType.PROXY, None, None)


async def virtual_key_soft_budget_check(
    valid_token: UserAPIKeyAuth,
    proxy_logging_obj: ProxyLogging,
    user_obj: LiteLLM_UserTable | None = None,
) -> None:
    from litellm.proxy.auth.auth_checks import send_soft_budget_alert

    if common_budgets.check(BudgetFacts("KeySoft", valid_token.soft_budget, valid_token.spend)).alert is not None:
        send_soft_budget_alert(valid_token, proxy_logging_obj, user_obj)


def _soft_emails(raw: object) -> tuple[str, ...]:
    if isinstance(raw, str):
        return tuple(email.strip() for email in raw.split(",") if email.strip())
    if isinstance(raw, list):
        return tuple(email for email in _ITEMS.validate_python(raw) if isinstance(email, str) and email.strip())
    return ()


async def team_soft_budget_check(
    team_object: LiteLLM_TeamTable | None,
    valid_token: UserAPIKeyAuth | None,
    proxy_logging_obj: ProxyLogging,
) -> None:
    if team_object is None:
        return
    emails: Final = _soft_emails(metadata(team_object.metadata).get("soft_budget_alerting_emails"))  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]  # validated legacy metadata boundary
    result: Final = common_budgets.check(
        BudgetFacts(
            "TeamSoft",
            team_object.soft_budget,
            team_object.spend,
            token_present=valid_token is not None,
            alert_recipients=emails,
        )
    )
    if result.alert is not None and valid_token is not None:
        emit_alert(
            proxy_logging_obj,
            result.alert,
            CallInfo(
                token=valid_token.token,
                spend=team_object.spend if team_object.spend is not None else 0.0,
                max_budget=team_object.max_budget,
                soft_budget=team_object.soft_budget,
                user_id=valid_token.user_id,
                team_id=valid_token.team_id,
                team_alias=valid_token.team_alias,
                organization_id=valid_token.org_id,
                user_email=None,
                key_alias=valid_token.key_alias,
                event_group=Litellm_EntityType.TEAM,
                alert_emails=list(emails),
            ),
        )


async def project_soft_budget_check(
    project_object: LiteLLM_ProjectTableCachedObj | None,
    valid_token: UserAPIKeyAuth | None,
    proxy_logging_obj: ProxyLogging,
) -> None:
    if project_object is None:
        return
    limit: Final = (
        project_object.litellm_budget_table.soft_budget if project_object.litellm_budget_table is not None else None
    )
    result: Final = common_budgets.check(
        BudgetFacts("ProjectSoft", limit, project_object.spend, token_present=valid_token is not None)
    )
    if result.alert is not None and valid_token is not None:
        emit_alert(
            proxy_logging_obj,
            result.alert,
            CallInfo(
                token=valid_token.token,
                spend=project_object.spend,
                max_budget=None,
                soft_budget=limit,
                user_id=valid_token.user_id,
                team_id=valid_token.team_id,
                team_alias=valid_token.team_alias,
                organization_id=valid_token.org_id,
                event_group=Litellm_EntityType.PROJECT,
            ),
        )


def resolve_effective_team_member_budget(
    member_budget: LiteLLM_BudgetTable | None,
    team_default_budget: LiteLLM_BudgetTable | None,
    now: datetime,
) -> float | None:
    return common_budgets.member_limit(
        read_member_limit_facts(
            member_budget, team_default_budget.max_budget if team_default_budget is not None else None, now
        )
    )


def read_member_limit_facts(
    member_budget: LiteLLM_BudgetTable | None, default_limit: float | None, now: datetime
) -> MemberLimitFacts:
    expiry: Final = member_budget.temp_budget_expiry if member_budget is not None else None
    aware_expiry: Final = (
        expiry.replace(tzinfo=timezone.utc) if expiry is not None and expiry.tzinfo is None else expiry
    )
    expiry_number: Final = (
        aware_expiry.replace(tzinfo=timezone.utc).timestamp()
        if aware_expiry is not None and aware_expiry.tzinfo is now.tzinfo
        else aware_expiry.timestamp()
        if aware_expiry is not None
        else None
    )
    now_number: Final = (
        now.replace(tzinfo=timezone.utc).timestamp()
        if aware_expiry is not None and aware_expiry.tzinfo is now.tzinfo
        else now.timestamp()
    )
    return MemberLimitFacts(
        member_limit=member_budget.max_budget if member_budget is not None else None,
        default_limit=default_limit,
        temporary_increase=member_budget.temp_budget_increase if member_budget is not None else None,
        expires_at=expiry_number,
        now=now_number,
        now_naive=now.utcoffset() is None,
    )


def threshold_alert_decision(
    spend: float | None,
    max_budget: float | None,
    config: Mapping[str, Sequence[str]] | None,
    *,
    team_member: bool = False,
) -> bool:
    minimum: Final = min((int(key) for key in (config or {}) if team_member or key.isdigit()), default=None)
    return common_budgets.alert(
        AlertFacts(spend, max_budget, bool(config), minimum, EMAIL_BUDGET_ALERT_MAX_SPEND_ALERT_PERCENTAGE, team_member)
    )
