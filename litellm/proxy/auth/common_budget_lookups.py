from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from litellm._auth import budget_lookup, common_budgets
from litellm._auth.common_budgets import BudgetFacts
from litellm.proxy._types import (
    Litellm_EntityType,
    LiteLLM_TagTable,
    LiteLLM_TeamMembership,
    LiteLLM_TeamTable,
    LiteLLM_UserTable,
    UserAPIKeyAuth,
)
from litellm.proxy.auth import budget_lookup_adapter
from litellm.proxy.auth.common_budget_adapter import (
    CounterRead,
    emit_alert,
    key_alert_info,
    raise_budget,
    read_budget,
    resolve_effective_team_member_budget,
)
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.redis.stores.spend_counters import model_access_group_counter
from litellm.proxy.spend_tracking.carried_budget_state import carry_organization_budget_state
from litellm.proxy.utils import PrismaClient, ProxyLogging
from litellm.types.proxy.model_access_group_budget import ModelAccessGroupBudget


async def organization_max_budget_check(
    valid_token: UserAPIKeyAuth | None,
    team_object: LiteLLM_TeamTable | None,
    prisma_client: PrismaClient | None,
    user_api_key_cache: UserApiKeyCache,
    proxy_logging_obj: ProxyLogging,
) -> None:
    reading: Final = await budget_lookup_adapter.organization(
        budget_lookup.OrganizationFacts(valid_token is not None, prisma_client is not None, team_object is not None),
        budget_lookup_adapter.LookupIO(valid_token, team_object, prisma_client, user_api_key_cache, proxy_logging_obj),
    )
    if reading is None:
        return
    assert valid_token is not None
    org_id: Final = reading.org_id
    org: Final = reading.row
    carry_organization_budget_state(valid_token=valid_token, org_table=org)
    limit: Final = org.litellm_budget_table.max_budget if org.litellm_budget_table is not None else None
    facts, result = await read_budget(
        BudgetFacts("Organization", limit), CounterRead(f"spend:org:{org_id}", org.spend or 0.0, limit)
    )
    if result.alert is not None:
        info: Final = key_alert_info(valid_token, facts.spend, limit, Litellm_EntityType.ORGANIZATION)
        info.organization_id = org_id
        emit_alert(proxy_logging_obj, result.alert, info)
    if result.denied:
        raise_budget(
            facts,
            Litellm_EntityType.ORGANIZATION,
            org_id,
            f"Budget has been exceeded! Organization={org_id} Current cost: {facts.spend}, Max budget: {limit}",
        )


async def tag_max_budget_check_for_tags(
    tags: Sequence[str],
    prisma_client: PrismaClient | None,
    user_api_key_cache: UserApiKeyCache,
    proxy_logging_obj: ProxyLogging,
    valid_token: UserAPIKeyAuth | None,
) -> None:
    from litellm.proxy.auth.auth_checks import get_tag_objects_batch

    if prisma_client is None or not tags:
        return
    rows: Final = await get_tag_objects_batch(
        tag_names=tags,
        prisma_client=prisma_client,
        user_api_key_cache=user_api_key_cache,
        proxy_logging_obj=proxy_logging_obj,
    )
    for name in tags:
        await check_tag_row(name, rows.get(name))


async def check_tag_row(name: str, row: LiteLLM_TagTable | None) -> None:
    limit: Final = (
        row.litellm_budget_table.max_budget if row is not None and row.litellm_budget_table is not None else None
    )
    facts, result = await read_budget(
        BudgetFacts("Tag", limit),
        CounterRead(f"spend:tag:{name}", (row.spend or 0.0) if row is not None else 0.0, limit, authoritative=True),
    )
    if result.denied:
        raise_budget(
            facts,
            Litellm_EntityType.TAG,
            name,
            f"Budget has been exceeded! Tag={name} Current cost: {facts.spend}, Max budget: {limit}",
        )


async def model_access_group_max_budget_check(
    matched_model_access_groups: Sequence[str],
    prisma_client: PrismaClient | None,
    user_api_key_cache: UserApiKeyCache,
) -> None:
    from litellm.proxy.auth.auth_checks import get_model_access_group_budgets_batch

    if prisma_client is None or not matched_model_access_groups:
        return
    rows: Final = await get_model_access_group_budgets_batch(
        access_group_names=matched_model_access_groups,
        prisma_client=prisma_client,
        user_api_key_cache=user_api_key_cache,
    )
    for name in matched_model_access_groups:
        await check_access_group_row(name, rows.get(name))


async def check_access_group_row(name: str, row: ModelAccessGroupBudget | None) -> None:
    limit: Final = row.max_budget if row is not None else None
    facts, result = await read_budget(
        BudgetFacts("AccessGroup", limit),
        CounterRead(
            model_access_group_counter(name),
            row.spend if row is not None else 0.0,
            limit,
            authoritative=True,
        ),
    )
    if result.denied:
        raise_budget(
            facts,
            Litellm_EntityType.MODEL_ACCESS_GROUP,
            name,
            f"Budget has been exceeded! Model access group={name} Current cost: {facts.spend}, Max budget: {limit}",
        )


async def check_team_member_budget(
    team_object: LiteLLM_TeamTable | None,
    user_object: LiteLLM_UserTable | None,
    valid_token: UserAPIKeyAuth | None,
    prisma_client: PrismaClient | None,
    user_api_key_cache: UserApiKeyCache,
    proxy_logging_obj: ProxyLogging,
    team_membership: LiteLLM_TeamMembership | None = None,
    team_membership_loaded: bool = False,
) -> None:
    from litellm.proxy.auth.budget_alert_adapter import team_member_max_budget_alert_check

    reading: Final = await budget_lookup_adapter.member(
        budget_lookup.MemberFacts(team_object is not None, valid_token is not None, team_membership_loaded),
        budget_lookup_adapter.LookupIO(valid_token, team_object, prisma_client, user_api_key_cache, proxy_logging_obj),
        team_membership,
    )
    if reading is None:
        return
    assert team_object is not None and valid_token is not None and valid_token.user_id is not None
    membership: Final = reading.membership
    team_metadata: Final = reading.metadata
    limit: Final = resolve_effective_team_member_budget(reading.row, reading.default, reading.now)
    facts, result = await read_budget(
        BudgetFacts("TeamMember", limit),
        CounterRead(
            f"spend:team_member:{valid_token.user_id}:{team_object.team_id}",
            (membership.spend or 0.0) if membership is not None else 0.0,
            limit,
        ),
    )
    if result.alert is not None and facts.spend is not None and limit is not None:
        team_member_max_budget_alert_check(
            team_id=team_object.team_id,
            team_alias=team_object.team_alias,
            team_metadata=dict(team_metadata),
            organization_id=team_object.organization_id,
            user_id=valid_token.user_id,
            user_email=user_object.user_email if user_object is not None else None,
            proxy_logging_obj=proxy_logging_obj,
            spend=facts.spend,
            max_budget=limit,
        )
    if result.denied:
        raise_budget(
            facts,
            Litellm_EntityType.TEAM_MEMBER,
            f"{valid_token.user_id}:{team_object.team_id}",
            f"Budget has been exceeded! User={valid_token.user_id} in Team={team_object.team_id} Current cost: {facts.spend}, Max budget: {limit}",
        )


@dataclass(frozen=True, slots=True)
class AccessGroupBudgetReading:
    name: str
    facts: BudgetFacts


async def read_access_group_budget_row(name: str, row: ModelAccessGroupBudget | None) -> AccessGroupBudgetReading:
    limit: Final = row.max_budget if row is not None else None
    facts, _decision = await read_budget(
        BudgetFacts("AccessGroup", limit),
        CounterRead(
            model_access_group_counter(name), row.spend if row is not None else 0.0, limit, authoritative=True
        ),
    )
    return AccessGroupBudgetReading(name, facts)


async def read_access_group_budget_facts(
    groups: Sequence[str],
    prisma_client: PrismaClient | None,
    user_api_key_cache: UserApiKeyCache,
) -> tuple[AccessGroupBudgetReading, ...]:
    from litellm.proxy.auth.auth_checks import get_model_access_group_budgets_batch

    if prisma_client is None or not groups:
        return ()
    rows: Final = await get_model_access_group_budgets_batch(
        access_group_names=groups,
        prisma_client=prisma_client,
        user_api_key_cache=user_api_key_cache,
    )
    return await _read_access_group_rows(tuple(groups), rows)


async def _read_access_group_rows(
    groups: tuple[str, ...],
    rows: Mapping[str, ModelAccessGroupBudget],
) -> tuple[AccessGroupBudgetReading, ...]:
    readings: tuple[AccessGroupBudgetReading, ...] = ()  # rebind-ok: native-selected IO driver accumulator
    for name in groups:
        reading = await read_access_group_budget_row(name, rows.get(name))
        readings = (*readings, reading)
        if common_budgets.check(reading.facts).denied:
            return readings
    return readings
