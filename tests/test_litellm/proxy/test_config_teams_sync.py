"""Unit tests for declarative config team sync and budget merge."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from litellm.proxy.management_helpers.config_teams_sync import (
    CONFIG_TEAM_METADATA_KEY,
    apply_team_member_budget_to_sa_key,
    extract_model_list_budgets,
    merge_team_model_max_budget,
    parse_config_teams,
    sync_config_teams,
    team_is_from_config,
)
from litellm.types.proxy.management_endpoints.config_teams import ConfigTeamEntry
from litellm.types.utils import BudgetConfig


def test_parse_teams_yaml_requires_team_id_or_alias() -> None:
    with pytest.raises(ValidationError):
        parse_config_teams([{"team_member_budget": 100}])


def test_max_budget_aliases_to_team_member_budget() -> None:
    teams = parse_config_teams(
        [
            {
                "team_id": "t1",
                "max_budget": 50,
                "budget_duration": "30d",
            }
        ]
    )
    assert len(teams) == 1
    assert teams[0].team_member_budget == 50
    assert teams[0].team_member_budget_duration == "30d"
    assert teams[0].max_budget is None
    assert teams[0].max_budget_was_aliased is True


def test_merge_model_info_budgets_into_team() -> None:
    model_budgets = extract_model_list_budgets(
        [
            {
                "model_name": "claude-sonnet-4-6",
                "model_info": {"budget_limit": 20, "budget_duration": "1d"},
            },
            {"model_name": "free-model"},
        ]
    )
    merged = merge_team_model_max_budget(model_budgets, None)
    assert merged == {
        "claude-sonnet-4-6": {"budget_limit": 20.0, "time_period": "1d"},
    }


def test_team_model_max_budget_overrides_model_info() -> None:
    model_budgets = extract_model_list_budgets(
        [
            {
                "model_name": "claude-sonnet-4-6",
                "model_info": {"budget_limit": 20, "budget_duration": "1d"},
            }
        ]
    )
    merged = merge_team_model_max_budget(
        model_budgets,
        {"claude-sonnet-4-6": {"budget_limit": 5, "time_period": "1d"}},
    )
    assert merged["claude-sonnet-4-6"]["budget_limit"] == 5.0


def test_model_without_budget_info_omitted() -> None:
    model_budgets = extract_model_list_budgets(
        [
            {"model_name": "a", "model_info": {}},
            {"model_name": "b", "model_info": {"budget_duration": "1d"}},
        ]
    )
    assert model_budgets == {}


def test_team_is_from_config() -> None:
    assert team_is_from_config({CONFIG_TEAM_METADATA_KEY: True}) is True
    assert team_is_from_config({}) is False
    assert team_is_from_config(None) is False


@pytest.mark.asyncio
async def test_sync_creates_missing_team() -> None:
    created: dict[str, Any] = {}

    async def fake_new_team(*, data, http_request, user_api_key_dict):
        created["data"] = data
        return data

    team_repo = MagicMock()
    team_repo.find_by_id = AsyncMock(return_value=None)
    team_repo.find_many = AsyncMock(return_value=[])

    with (
        patch(
            "litellm.repositories.team_repository.TeamRepository",
            return_value=team_repo,
        ),
        patch(
            "litellm.proxy.management_endpoints.team_endpoints.new_team",
            new=fake_new_team,
        ),
        patch(
            "litellm.proxy.management_endpoints.team_endpoints.update_team",
            new=AsyncMock(),
        ),
    ):
        synced = await sync_config_teams(
            config_teams=parse_config_teams(
                [
                    {
                        "team_id": "systems",
                        "team_alias": "Systems",
                        "team_member_budget": 100,
                        "team_member_budget_duration": "30d",
                    }
                ]
            ),
            model_list=[
                {
                    "model_name": "claude-sonnet-4-6",
                    "model_info": {"budget_limit": 20, "budget_duration": "1d"},
                }
            ],
            prisma_client=MagicMock(),
            master_key="sk-test",
        )

    assert synced == ("systems",)
    data = created["data"]
    assert data.team_id == "systems"
    assert data.team_member_budget == 100
    assert data.team_member_budget_duration == "30d"
    assert data.max_budget is None
    assert data.metadata == {CONFIG_TEAM_METADATA_KEY: True}
    assert data.model_max_budget["claude-sonnet-4-6"]["budget_limit"] == 20.0


@pytest.mark.asyncio
async def test_sync_updates_existing_config_team() -> None:
    updated: dict[str, Any] = {}

    async def fake_update_team(*, data, http_request, user_api_key_dict):
        updated["data"] = data
        return {"team_id": data.team_id, "data": data}

    existing = SimpleNamespace(
        team_id="systems",
        team_alias="Old",
        metadata={CONFIG_TEAM_METADATA_KEY: True},
    )
    team_repo = MagicMock()
    team_repo.find_by_id = AsyncMock(return_value=existing)
    team_repo.find_many = AsyncMock(return_value=[])

    with (
        patch(
            "litellm.repositories.team_repository.TeamRepository",
            return_value=team_repo,
        ),
        patch(
            "litellm.proxy.management_endpoints.team_endpoints.new_team",
            new=AsyncMock(),
        ),
        patch(
            "litellm.proxy.management_endpoints.team_endpoints.update_team",
            new=fake_update_team,
        ),
    ):
        synced = await sync_config_teams(
            config_teams=parse_config_teams(
                [
                    {
                        "team_id": "systems",
                        "team_alias": "Systems",
                        "team_member_budget": 75,
                        "team_member_budget_duration": "30d",
                        "model_max_budget": {
                            "claude-opus-4-8": {"budget_limit": 10, "time_period": "1d"},
                        },
                    }
                ]
            ),
            model_list=[
                {
                    "model_name": "claude-sonnet-4-6",
                    "model_info": {"budget_limit": 20, "budget_duration": "1d"},
                }
            ],
            prisma_client=MagicMock(),
            master_key="sk-test",
        )

    assert synced == ("systems",)
    data = updated["data"]
    assert data.team_alias == "Systems"
    assert data.team_member_budget == 75
    assert "models" not in data.model_fields_set
    sonnet = data.model_max_budget["claude-sonnet-4-6"]
    opus = data.model_max_budget["claude-opus-4-8"]
    assert getattr(sonnet, "max_budget", None) == 20.0 or (
        isinstance(sonnet, dict) and sonnet.get("budget_limit") == 20.0
    )
    assert getattr(opus, "max_budget", None) == 10.0 or (
        isinstance(opus, dict) and opus.get("budget_limit") == 10.0
    )
    assert data.metadata[CONFIG_TEAM_METADATA_KEY] is True


@pytest.mark.asyncio
async def test_sync_update_includes_models_when_configured() -> None:
    updated: dict[str, Any] = {}

    async def fake_update_team(*, data, http_request, user_api_key_dict):
        updated["data"] = data
        return {"team_id": data.team_id, "data": data}

    existing = SimpleNamespace(
        team_id="systems",
        team_alias="Systems",
        metadata={CONFIG_TEAM_METADATA_KEY: True},
    )
    team_repo = MagicMock()
    team_repo.find_by_id = AsyncMock(return_value=existing)

    with (
        patch(
            "litellm.repositories.team_repository.TeamRepository",
            return_value=team_repo,
        ),
        patch(
            "litellm.proxy.management_endpoints.team_endpoints.new_team",
            new=AsyncMock(),
        ),
        patch(
            "litellm.proxy.management_endpoints.team_endpoints.update_team",
            new=fake_update_team,
        ),
    ):
        await sync_config_teams(
            config_teams=parse_config_teams(
                [
                    {
                        "team_id": "systems",
                        "team_alias": "Systems",
                        "team_member_budget": 100,
                        "models": ["claude-haiku-4-5"],
                    }
                ]
            ),
            model_list=None,
            prisma_client=MagicMock(),
            master_key="sk-test",
        )

    assert updated["data"].models == ["claude-haiku-4-5"]
    assert "models" in updated["data"].model_fields_set


@pytest.mark.asyncio
async def test_sync_does_not_clobber_non_config_team() -> None:
    existing = SimpleNamespace(
        team_id="other",
        team_alias="Systems",
        metadata={},
    )
    team_repo = MagicMock()
    team_repo.find_by_id = AsyncMock(return_value=None)
    team_repo.find_many = AsyncMock(return_value=[existing])
    new_team = AsyncMock()
    update_team = AsyncMock()

    with (
        patch(
            "litellm.repositories.team_repository.TeamRepository",
            return_value=team_repo,
        ),
        patch(
            "litellm.proxy.management_endpoints.team_endpoints.new_team",
            new=new_team,
        ),
        patch(
            "litellm.proxy.management_endpoints.team_endpoints.update_team",
            new=update_team,
        ),
    ):
        synced = await sync_config_teams(
            config_teams=parse_config_teams([{"team_alias": "Systems", "team_member_budget": 100}]),
            model_list=None,
            prisma_client=MagicMock(),
            master_key="sk-test",
        )

    assert synced == ()
    new_team.assert_not_called()
    update_team.assert_not_called()


@pytest.mark.asyncio
async def test_apply_team_member_budget_to_sa_key() -> None:
    data = SimpleNamespace(user_id=None, max_budget=None, budget_duration=None)
    team_table = SimpleNamespace(metadata={"team_member_budget_id": "b1"})
    budget = SimpleNamespace(max_budget=100.0, budget_duration="30d")

    with patch(
        "litellm.proxy.auth.auth_checks.get_team_member_default_budget",
        new=AsyncMock(return_value=budget),
    ):
        await apply_team_member_budget_to_sa_key(
            data=data,
            team_table=team_table,
            prisma_client=MagicMock(),
            user_api_key_cache=MagicMock(),
        )

    assert data.max_budget == 100.0
    assert data.budget_duration == "30d"


@pytest.mark.asyncio
async def test_apply_team_member_budget_preserves_explicit_max_budget() -> None:
    data = SimpleNamespace(user_id=None, max_budget=5.0, budget_duration=None)
    team_table = SimpleNamespace(metadata={"team_member_budget_id": "b1"})

    with patch(
        "litellm.proxy.auth.auth_checks.get_team_member_default_budget",
        new=AsyncMock(return_value=SimpleNamespace(max_budget=100.0, budget_duration="30d")),
    ) as get_budget:
        await apply_team_member_budget_to_sa_key(
            data=data,
            team_table=team_table,
            prisma_client=MagicMock(),
            user_api_key_cache=MagicMock(),
        )

    assert data.max_budget == 5.0
    get_budget.assert_not_called()


@pytest.mark.asyncio
async def test_human_key_does_not_copy_member_budget() -> None:
    data = SimpleNamespace(user_id="user-1", max_budget=None, budget_duration=None)
    team_table = SimpleNamespace(metadata={"team_member_budget_id": "b1"})

    with patch(
        "litellm.proxy.auth.auth_checks.get_team_member_default_budget",
        new=AsyncMock(return_value=SimpleNamespace(max_budget=100.0, budget_duration="30d")),
    ) as get_budget:
        await apply_team_member_budget_to_sa_key(
            data=data,
            team_table=team_table,
            prisma_client=MagicMock(),
            user_api_key_cache=MagicMock(),
        )

    assert data.max_budget is None
    get_budget.assert_not_called()


def test_model_info_accepts_budget_limit_and_duration() -> None:
    from litellm.types.router import ModelInfo

    info = ModelInfo(budget_limit=20, budget_duration="1d")
    assert info.budget_limit == 20
    assert info.budget_duration == "1d"


def test_model_info_budget_not_required() -> None:
    from litellm.types.router import ModelInfo

    info = ModelInfo()
    assert info.budget_limit is None
    assert info.budget_duration is None


def test_config_team_entry_explicit_member_budget_wins_over_alias() -> None:
    entry = ConfigTeamEntry(team_id="t1", team_member_budget=100, max_budget=50)
    assert entry.team_member_budget == 100
    assert entry.max_budget_was_aliased is True


def test_budget_config_aliases() -> None:
    config = BudgetConfig(budget_limit=20, time_period="1d")
    assert config.max_budget == 20
    assert config.budget_duration == "1d"
