"""Unit tests for config-team budget mutation locks."""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from litellm.proxy.management_endpoints.team_endpoints import _reject_config_team_budget_mutation
from litellm.proxy.management_helpers.config_teams_sync import (
    CONFIG_MEMBER_BUDGET_FIELDS,
    CONFIG_TEAM_BUDGET_FIELDS,
    CONFIG_TEAM_METADATA_KEY,
    _config_team_sync_active,
)


def test_team_update_rejects_budget_fields_when_from_config() -> None:
    with pytest.raises(HTTPException) as exc:
        _reject_config_team_budget_mutation(
            team_metadata={CONFIG_TEAM_METADATA_KEY: True},
            payload={"team_member_budget": 50.0},
            field_names=CONFIG_TEAM_BUDGET_FIELDS,
        )
    assert exc.value.status_code == 400
    assert "config" in str(exc.value.detail).lower()


def test_team_member_update_rejects_budget_fields_when_team_from_config() -> None:
    with pytest.raises(HTTPException) as exc:
        _reject_config_team_budget_mutation(
            team_metadata={CONFIG_TEAM_METADATA_KEY: True},
            payload={"max_budget_in_team": 10.0, "model_max_budget_in_team": {"m": {"budget_limit": 1}}},
            field_names=CONFIG_MEMBER_BUDGET_FIELDS,
        )
    assert exc.value.status_code == 400


def test_non_config_team_budget_update_still_works() -> None:
    _reject_config_team_budget_mutation(
        team_metadata={},
        payload={"team_member_budget": 50.0},
        field_names=CONFIG_TEAM_BUDGET_FIELDS,
    )


def test_config_sync_bypasses_budget_lock() -> None:
    token = _config_team_sync_active.set(True)
    try:
        _reject_config_team_budget_mutation(
            team_metadata={CONFIG_TEAM_METADATA_KEY: True},
            payload={"team_member_budget": 50.0},
            field_names=CONFIG_TEAM_BUDGET_FIELDS,
        )
    finally:
        _config_team_sync_active.reset(token)


def test_non_budget_fields_allowed_on_config_team() -> None:
    _reject_config_team_budget_mutation(
        team_metadata={CONFIG_TEAM_METADATA_KEY: True},
        payload={"team_alias": "Renamed"},
        field_names=CONFIG_TEAM_BUDGET_FIELDS,
    )
