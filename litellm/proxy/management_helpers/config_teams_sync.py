"""Sync declarative `teams:` YAML entries into LiteLLM_TeamTable on startup/reload."""

from __future__ import annotations

import contextvars
import uuid
from typing import Any, Mapping, Optional, Sequence

from fastapi import Request

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import NewTeamRequest, UpdateTeamRequest, UserAPIKeyAuth
from litellm.types.proxy.management_endpoints.config_teams import (
    ConfigTeamEntry,
    normalize_budget_config_dict,
)
from litellm.types.utils import BudgetConfig, GenericBudgetConfigType

CONFIG_TEAM_METADATA_KEY = "is_from_config"
CONFIG_TEAM_BUDGET_FIELDS = frozenset(
    {
        "max_budget",
        "soft_budget",
        "budget_duration",
        "budget_limits",
        "team_member_budget",
        "team_member_budget_duration",
        "team_member_rpm_limit",
        "team_member_tpm_limit",
        "model_max_budget",
    }
)
CONFIG_MEMBER_BUDGET_FIELDS = frozenset(
    {
        "max_budget_in_team",
        "budget_duration",
        "model_max_budget_in_team",
        "rpm_limit",
        "tpm_limit",
    }
)

_config_team_sync_active: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "config_team_sync_active",
    default=False,
)


def is_config_team_sync_active() -> bool:
    return _config_team_sync_active.get()


def parse_config_teams(raw_teams: Sequence[Mapping[str, Any]] | None) -> tuple[ConfigTeamEntry, ...]:
    if not raw_teams:
        return ()
    return tuple(ConfigTeamEntry(**dict(entry)) for entry in raw_teams)


def extract_model_list_budgets(
    model_list: Sequence[Mapping[str, Any]] | None,
) -> dict[str, BudgetConfig]:
    if not model_list:
        return {}

    budgets: dict[str, BudgetConfig] = {}
    for model in model_list:
        model_name = model.get("model_name")
        if not isinstance(model_name, str) or not model_name:
            continue
        model_info = model.get("model_info") or {}
        if not isinstance(model_info, Mapping):
            continue
        budget_limit = model_info.get("budget_limit", model_info.get("max_budget"))
        budget_duration = model_info.get("budget_duration", model_info.get("time_period"))
        if budget_limit is None:
            continue
        budgets[model_name] = BudgetConfig(
            budget_limit=budget_limit,
            time_period=budget_duration,
        )
    return budgets


def merge_team_model_max_budget(
    model_list_budgets: Mapping[str, BudgetConfig],
    team_overrides: GenericBudgetConfigType | Mapping[str, Any] | None,
) -> dict[str, dict[str, float | str | None]]:
    merged: dict[str, BudgetConfig] = dict(model_list_budgets)
    if team_overrides:
        normalized = normalize_budget_config_dict(dict(team_overrides))
        merged.update(normalized)
    return {
        model_name: {
            "budget_limit": config.max_budget,
            "time_period": config.budget_duration,
        }
        for model_name, config in merged.items()
        if config.max_budget is not None
    }


def team_is_from_config(metadata: Mapping[str, Any] | None) -> bool:
    if not metadata:
        return False
    return bool(metadata.get(CONFIG_TEAM_METADATA_KEY))


def budget_fields_in_payload(payload: Mapping[str, Any], field_names: frozenset[str]) -> tuple[str, ...]:
    return tuple(sorted(name for name in field_names if name in payload and payload[name] is not None))


def _master_user_auth(master_key: Optional[str]) -> UserAPIKeyAuth:
    from litellm.proxy._types import LitellmUserRoles, hash_token

    token = hash_token(master_key) if master_key else ""
    return UserAPIKeyAuth(token=token, user_role=LitellmUserRoles.PROXY_ADMIN)


def _http_request() -> Request:
    return Request(scope={"type": "http", "method": "POST"})


async def sync_config_teams(
    *,
    config_teams: Sequence[ConfigTeamEntry],
    model_list: Sequence[Mapping[str, Any]] | None,
    prisma_client: Any,
    master_key: Optional[str],
) -> tuple[str, ...]:
    """Create/update teams declared in YAML. Returns synced team_ids."""
    if not config_teams:
        return ()
    if prisma_client is None:
        verbose_proxy_logger.warning("Skipping config teams sync: no database connected")
        return ()

    from litellm.repositories.team_repository import TeamRepository

    model_list_budgets = extract_model_list_budgets(model_list)
    user_api_key_dict = _master_user_auth(master_key)
    team_repo = TeamRepository(prisma_client)
    synced_ids: list[str] = []
    sync_token = _config_team_sync_active.set(True)
    try:
        for entry in config_teams:
            synced_id = await _sync_one_config_team(
                entry=entry,
                model_list_budgets=model_list_budgets,
                team_repo=team_repo,
                user_api_key_dict=user_api_key_dict,
            )
            if synced_id is not None:
                synced_ids.append(synced_id)
    finally:
        _config_team_sync_active.reset(sync_token)

    return tuple(synced_ids)


async def _sync_one_config_team(
    *,
    entry: ConfigTeamEntry,
    model_list_budgets: Mapping[str, BudgetConfig],
    team_repo: Any,
    user_api_key_dict: UserAPIKeyAuth,
) -> Optional[str]:
    from litellm.proxy.management_endpoints.team_endpoints import new_team, update_team

    if entry.max_budget_was_aliased:
        verbose_proxy_logger.warning(
            "Config team %s: max_budget is aliased to team_member_budget (per-user/SA); "
            "shared team pools are not used for config teams",
            entry.team_id or entry.team_alias,
        )

    team_id = entry.team_id or str(uuid.uuid4())
    model_max_budget = merge_team_model_max_budget(model_list_budgets, entry.model_max_budget)
    metadata = {CONFIG_TEAM_METADATA_KEY: True}

    existing = None
    if entry.team_id is not None:
        existing = await team_repo.find_by_id(entry.team_id, id_field="team_id")
    elif entry.team_alias is not None:
        alias_matches = await team_repo.find_many(where={"team_alias": entry.team_alias})
        config_matches = tuple(team for team in alias_matches if team_is_from_config(team.metadata))
        if len(config_matches) == 1:
            existing = config_matches[0]
            team_id = existing.team_id
        elif alias_matches and not config_matches:
            verbose_proxy_logger.warning(
                "Skipping team_alias=%s: matches a non-config team; set team_id explicitly to manage it from config",
                entry.team_alias,
            )
            return None

    if existing is None:
        create_data = NewTeamRequest(
            team_id=team_id,
            team_alias=entry.team_alias,
            team_member_budget=entry.team_member_budget,
            team_member_budget_duration=entry.team_member_budget_duration,
            model_max_budget=model_max_budget or None,
            models=entry.models or [],
            metadata=metadata,
        )
        await new_team(
            data=create_data,
            http_request=_http_request(),
            user_api_key_dict=user_api_key_dict,
        )
        verbose_proxy_logger.info(
            "Created config team team_id=%s team_alias=%s",
            team_id,
            entry.team_alias,
        )
        return team_id

    existing_metadata = dict(existing.metadata or {})
    existing_metadata[CONFIG_TEAM_METADATA_KEY] = True
    update_kwargs: dict[str, Any] = {
        "team_id": existing.team_id,
        "metadata": existing_metadata,
    }
    if entry.team_alias is not None:
        update_kwargs["team_alias"] = entry.team_alias
    if entry.team_member_budget is not None:
        update_kwargs["team_member_budget"] = entry.team_member_budget
    if entry.team_member_budget_duration is not None:
        update_kwargs["team_member_budget_duration"] = entry.team_member_budget_duration
    if model_max_budget:
        update_kwargs["model_max_budget"] = model_max_budget
    # Prisma rejects models=null; only set when config explicitly provides a list
    if entry.models is not None:
        update_kwargs["models"] = entry.models
    update_data = UpdateTeamRequest(**update_kwargs)
    await update_team(
        data=update_data,
        http_request=_http_request(),
        user_api_key_dict=user_api_key_dict,
    )
    verbose_proxy_logger.info(
        "Updated config team team_id=%s team_alias=%s",
        existing.team_id,
        entry.team_alias,
    )
    return existing.team_id


async def apply_team_member_budget_to_sa_key(
    *,
    data: Any,
    team_table: Any,
    prisma_client: Any,
    user_api_key_cache: Any,
) -> None:
    """Copy team member budget defaults onto SA keys that have no explicit max_budget."""
    if getattr(data, "user_id", None) is not None:
        return
    if getattr(data, "max_budget", None) is not None:
        return
    if team_table is None:
        return

    metadata = getattr(team_table, "metadata", None) or {}
    budget_id = metadata.get("team_member_budget_id") if isinstance(metadata, Mapping) else None
    if not isinstance(budget_id, str):
        return

    from litellm.proxy.auth.auth_checks import get_team_member_default_budget

    budget = await get_team_member_default_budget(
        budget_id=budget_id,
        prisma_client=prisma_client,
        user_api_key_cache=user_api_key_cache,
    )
    if budget is None or budget.max_budget is None:
        return

    data.max_budget = budget.max_budget
    if getattr(data, "budget_duration", None) is None and budget.budget_duration is not None:
        data.budget_duration = budget.budget_duration
