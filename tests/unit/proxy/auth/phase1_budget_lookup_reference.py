"""Unchanged Phase 1 budget lookup policy with raw IO supplied by tests."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Final

from pydantic import TypeAdapter, ValidationError

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import (
    CallInfo,
    Litellm_EntityType,
    LiteLLM_OrganizationTable,
    LiteLLM_TeamMembership,
    LiteLLM_TeamTable,
    LiteLLM_UserTable,
    UserAPIKeyAuth,
)
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.spend_tracking.carried_budget_state import carry_organization_budget_state
from litellm.proxy.utils import PrismaClient, ProxyLogging
from litellm.utils import get_utc_datetime

from .phase1_budget_auth_reference import (
    get_team_member_default_budget,
    get_team_membership,
    resolve_effective_team_member_budget,
)

TEAM_MEMBER_MAX_BUDGET_ALERT_EMAILS_KEY: Final = "team_member_max_budget_alert_emails"
_TEAM_MEMBER_ALERT_CONFIG_ADAPTER: Final = TypeAdapter(Mapping[str, object])


async def get_org_object(
    org_id: str,
    prisma_client: PrismaClient,
    user_api_key_cache: UserApiKeyCache,
    proxy_logging_obj: ProxyLogging,
    include_budget_table: bool,
) -> LiteLLM_OrganizationTable | None:
    raise AssertionError("Frozen reference raw IO must be supplied by the test: get_org_object")


def _parse_email_list(raw: str | Sequence[object] | None) -> list[str]:  # mutable-ok: unchanged Phase 1 collection contract in independent oracle
    """Parse emails from a list or comma-separated string."""
    if isinstance(raw, list):
        return [e.strip() for e in raw if isinstance(e, str) and e.strip()]
    elif isinstance(raw, str):
        return [e.strip() for e in raw.split(",") if e.strip()]
    return []


def _merge_budget_alert_email_configs(
    global_cfg: Mapping[str, str | Sequence[object] | None] | None,
    per_key_cfg: Mapping[str, str | Sequence[object] | None] | None,
) -> dict[str, list[str]] | None:  # mutable-ok: unchanged Phase 1 collection contract in independent oracle
    """
    Per-threshold additive merge: each threshold's recipient list is the union
    of global + per-key entries (deduped, global-first ordering). Missing
    thresholds on one side are inherited from the other.
    """
    global_cfg_normalized: Final = _normalize_alert_emails(global_cfg)
    per_key_cfg_normalized: Final = _normalize_alert_emails(per_key_cfg)
    if not global_cfg_normalized and (not per_key_cfg_normalized):
        return None
    thresholds: Final = set(global_cfg_normalized) | set(per_key_cfg_normalized)
    return {
        t: list(dict.fromkeys(global_cfg_normalized.get(t, []) + per_key_cfg_normalized.get(t, []))) for t in thresholds
    }


def _is_valid_alert_threshold_pct(pct: str) -> bool:
    return pct.isdigit() and len(pct) <= 3 and (1 <= int(pct) <= 100)


def _alert_recipients(raw: object) -> Sequence[str] | None:
    if isinstance(raw, (str, Sequence)):
        return _parse_email_list(raw)  # pyright: ignore[reportUnknownArgumentType]  # preserve unvalidated Phase 1 oracle values and call shape
    return None


def _valid_alert_threshold_config(raw_config: object) -> Mapping[str, str | Sequence[object] | None] | None:
    try:
        config: Final = _TEAM_MEMBER_ALERT_CONFIG_ADAPTER.validate_python(raw_config)
    except ValidationError:
        return None
    return MappingProxyType(
        {pct: _alert_recipients(emails) for pct, emails in config.items() if _is_valid_alert_threshold_pct(pct)}
    )


def team_member_max_budget_alert_check(
    team_id: str,
    team_alias: str | None,
    team_metadata: Mapping[str, object] | None,
    organization_id: str | None,
    user_id: str,
    user_email: str | None,
    proxy_logging_obj: ProxyLogging,
    spend: float,
    max_budget: float,
) -> None:
    raw_config: Final = team_metadata.get(TEAM_MEMBER_MAX_BUDGET_ALERT_EMAILS_KEY) if team_metadata else None
    alert_email_config: Final = _merge_budget_alert_email_configs(
        global_cfg=None, per_key_cfg=_valid_alert_threshold_config(raw_config)
    )
    if not alert_email_config or spend <= 0:
        return
    min_pct: Final = min(int(pct) for pct in alert_email_config)
    if spend < max_budget * (min_pct / 100.0):
        return
    call_info: Final = CallInfo(
        spend=spend,
        max_budget=max_budget,
        user_id=user_id,
        team_id=team_id,
        team_alias=team_alias,
        organization_id=organization_id,
        user_email=user_email,
        event_group=Litellm_EntityType.TEAM_MEMBER,
        max_budget_alert_emails=alert_email_config,
    )
    asyncio.create_task(proxy_logging_obj.budget_alerts(type="max_budget_alert", user_info=call_info))


async def _check_team_member_budget(  # pyright: ignore[reportUnusedFunction]  # preserve unvalidated Phase 1 oracle values and call shape
    team_object: LiteLLM_TeamTable | None,
    user_object: LiteLLM_UserTable | None,
    valid_token: UserAPIKeyAuth | None,
    prisma_client: PrismaClient | None,
    user_api_key_cache: UserApiKeyCache,
    proxy_logging_obj: ProxyLogging,
    team_membership: LiteLLM_TeamMembership | None = None,
    team_membership_loaded: bool = False,
):
    """Check if team member is over their max budget within the team."""
    if (
        team_object is not None
        and team_object.team_id is not None  # pyright: ignore[reportUnnecessaryComparison]  # preserve unvalidated Phase 1 oracle values and call shape
        and (valid_token is not None)
        and (valid_token.user_id is not None)
    ):
        if not team_membership_loaded:
            team_membership = await get_team_membership(  # rebind-ok: preserve exact Phase 1 evaluation and mutation sequence
                user_id=valid_token.user_id,
                team_id=team_object.team_id,
                prisma_client=prisma_client,
                user_api_key_cache=user_api_key_cache,
                proxy_logging_obj=proxy_logging_obj,
            )
        loaded_membership = team_membership  # rebind-ok: preserve exact Phase 1 evaluation and mutation sequence
        member_budget_row: Final = loaded_membership.litellm_budget_table if loaded_membership is not None else None
        now: Final = get_utc_datetime()
        default_budget_id: Final = (team_object.metadata or {}).get("team_member_budget_id")  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  # preserve unvalidated Phase 1 oracle values and call shape
        team_default_budget: Final = (
            await get_team_member_default_budget(
                budget_id=default_budget_id, prisma_client=prisma_client, user_api_key_cache=user_api_key_cache
            )
            if (member_budget_row is None or member_budget_row.max_budget is None)
            and isinstance(default_budget_id, str)
            else None
        )
        team_member_budget: Final = resolve_effective_team_member_budget(
            member_budget=member_budget_row, team_default_budget=team_default_budget, now=now
        )
        if team_member_budget is not None:
            team_member_spend = (loaded_membership.spend if loaded_membership is not None else 0.0) or 0.0  # rebind-ok: preserve exact Phase 1 evaluation and mutation sequence
            from litellm.proxy.proxy_server import get_current_spend

            team_member_spend = await get_current_spend(  # rebind-ok: preserve exact Phase 1 evaluation and mutation sequence
                counter_key=f"spend:team_member:{valid_token.user_id}:{team_object.team_id}",
                fallback_spend=team_member_spend,
                max_budget=team_member_budget,
            )
            if not math.isfinite(team_member_budget):
                return
            team_member_max_budget_alert_check(
                team_id=team_object.team_id,
                team_alias=team_object.team_alias,
                team_metadata=team_object.metadata,  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]  # preserve unvalidated Phase 1 oracle values and call shape
                organization_id=team_object.organization_id,
                user_id=valid_token.user_id,
                user_email=user_object.user_email if user_object is not None else None,
                proxy_logging_obj=proxy_logging_obj,
                spend=team_member_spend,
                max_budget=team_member_budget,
            )
            if team_member_spend >= team_member_budget:
                raise litellm.BudgetExceededError(
                    current_cost=team_member_spend,
                    max_budget=team_member_budget,
                    message=f"Budget has been exceeded! User={valid_token.user_id} in Team={team_object.team_id} Current cost: {team_member_spend}, Max budget: {team_member_budget}",
                    entity_type=Litellm_EntityType.TEAM_MEMBER.value,
                    entity_id=f"{valid_token.user_id}:{team_object.team_id}",
                )


async def _team_soft_budget_check(  # pyright: ignore[reportUnusedFunction]  # preserve unvalidated Phase 1 oracle values and call shape
    team_object: LiteLLM_TeamTable | None, valid_token: UserAPIKeyAuth | None, proxy_logging_obj: ProxyLogging
):
    """
    Triggers a budget alert if the team is over it's soft budget.
    """
    if (
        team_object is not None
        and team_object.soft_budget is not None
        and (team_object.spend is not None)
        and (team_object.spend >= team_object.soft_budget)
    ):
        verbose_proxy_logger.debug(
            "Crossed Soft Budget for team %s, spend %s, soft_budget %s",
            team_object.team_id,
            team_object.spend,
            team_object.soft_budget,
        )
        if valid_token:
            alert_emails: list[str] | None = None  # mutable-ok: unchanged Phase 1 collection contract in independent oracle  # rebind-ok: preserve exact Phase 1 evaluation and mutation sequence
            if team_object.metadata is not None and isinstance(team_object.metadata, dict):  # pyright: ignore[reportUnknownMemberType, reportUnnecessaryIsInstance]  # preserve unvalidated Phase 1 oracle values and call shape
                soft_budget_alert_emails: Final = team_object.metadata.get("soft_budget_alerting_emails")  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  # preserve unvalidated Phase 1 oracle values and call shape
                if soft_budget_alert_emails is not None:
                    if isinstance(soft_budget_alert_emails, list):
                        alert_emails = [  # rebind-ok: preserve exact Phase 1 evaluation and mutation sequence
                            email for email in soft_budget_alert_emails if isinstance(email, str) and email.strip()  # pyright: ignore[reportUnknownVariableType]  # preserve unvalidated Phase 1 oracle values and call shape
                        ]
                    elif isinstance(soft_budget_alert_emails, str):
                        alert_emails = [email.strip() for email in soft_budget_alert_emails.split(",") if email.strip()]  # rebind-ok: preserve exact Phase 1 evaluation and mutation sequence
                    if alert_emails:
                        alert_emails = [email for email in alert_emails if email]  # rebind-ok: preserve exact Phase 1 evaluation and mutation sequence
                    else:
                        alert_emails = None  # rebind-ok: preserve exact Phase 1 evaluation and mutation sequence
            if alert_emails is None or len(alert_emails) == 0:
                verbose_proxy_logger.debug(
                    "Skipping team soft budget alert for team %s: no alert_emails configured in metadata.soft_budget_alerting_emails",
                    team_object.team_id,
                )
                return
            call_info: Final = CallInfo(
                token=valid_token.token,
                spend=team_object.spend,
                max_budget=team_object.max_budget,
                soft_budget=team_object.soft_budget,
                user_id=valid_token.user_id,
                team_id=valid_token.team_id,
                team_alias=valid_token.team_alias,
                organization_id=valid_token.org_id,
                user_email=None,
                key_alias=valid_token.key_alias,
                event_group=Litellm_EntityType.TEAM,
                alert_emails=alert_emails,
            )
            asyncio.create_task(proxy_logging_obj.budget_alerts(type="soft_budget", user_info=call_info))


async def organization_max_budget_check(
    valid_token: UserAPIKeyAuth | None,
    team_object: LiteLLM_TeamTable | None,
    prisma_client: PrismaClient | None,
    user_api_key_cache: UserApiKeyCache,
    proxy_logging_obj: ProxyLogging,
) -> None:
    """
    Check if the organization is over its max budget.

    This function checks the organization budget using:
    1. First, tries to use valid_token.org_id (if key has organization_id set)
    2. Falls back to team_object.organization_id (if key doesn't have org_id but team does)

    This ensures organization budget checks work even when keys don't have organization_id
    set directly, as long as their team belongs to an organization.

    Raises:
        BudgetExceededError if the organization is over its max budget.
        Triggers a budget alert if the organization is over its max budget.
    """
    if valid_token is None or prisma_client is None:
        return
    org_id: str | None = None  # rebind-ok: preserve exact Phase 1 evaluation and mutation sequence
    if valid_token.org_id is not None:
        org_id = valid_token.org_id  # rebind-ok: preserve exact Phase 1 evaluation and mutation sequence
    elif team_object is not None and team_object.organization_id is not None:
        org_id = team_object.organization_id  # rebind-ok: preserve exact Phase 1 evaluation and mutation sequence
    if org_id is None:
        return
    try:
        org_table: Final = await get_org_object(
            org_id=org_id,
            prisma_client=prisma_client,
            user_api_key_cache=user_api_key_cache,
            proxy_logging_obj=proxy_logging_obj,
            include_budget_table=True,
        )
    except Exception:
        return
    if org_table is None:
        return
    carry_organization_budget_state(valid_token=valid_token, org_table=org_table)
    org_max_budget: float | None = None  # rebind-ok: preserve exact Phase 1 evaluation and mutation sequence
    if org_table.litellm_budget_table is not None:
        org_max_budget = org_table.litellm_budget_table.max_budget  # rebind-ok: preserve exact Phase 1 evaluation and mutation sequence
    if org_max_budget is None:
        return
    from litellm.proxy.proxy_server import get_current_spend

    org_spend: Final = await get_current_spend(
        counter_key=f"spend:org:{org_id}", fallback_spend=org_table.spend or 0.0, max_budget=org_max_budget
    )
    if math.isfinite(org_max_budget) and org_spend >= org_max_budget:
        call_info: Final = CallInfo(
            token=valid_token.token,
            spend=org_spend,
            max_budget=org_max_budget,
            user_id=valid_token.user_id,
            team_id=valid_token.team_id,
            team_alias=valid_token.team_alias,
            organization_id=org_id,
            event_group=Litellm_EntityType.ORGANIZATION,
        )
        asyncio.create_task(proxy_logging_obj.budget_alerts(type="organization_budget", user_info=call_info))
        raise litellm.BudgetExceededError(
            current_cost=org_spend,
            max_budget=org_max_budget,
            message=f"Budget has been exceeded! Organization={org_id} Current cost: {org_spend}, Max budget: {org_max_budget}",
            entity_type=Litellm_EntityType.ORGANIZATION.value,
            entity_id=org_id,
        )


def _normalize_alert_emails(cfg: Mapping[str, str | Sequence[object] | None] | None) -> dict[str, list[str]]:  # mutable-ok: unchanged Phase 1 collection contract in independent oracle
    """Coerce user-supplied threshold→recipients mapping to Dict[str, List[str]].

    Values may legitimately arrive as list, comma-separated string, or None
    from YAML/metadata; _parse_email_list tolerates all three.
    """
    if not cfg:
        return {}
    return {k: _parse_email_list(v) for k, v in cfg.items()}
