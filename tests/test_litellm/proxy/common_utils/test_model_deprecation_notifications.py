"""Tests for the model deprecation email notifications module."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date
from typing import Final
from unittest.mock import MagicMock

import pytest

from litellm.models.team import LiteLLM_TeamTable
from litellm.proxy.common_utils.model_deprecation_notifications import (
    AffectedModel,
    resolve_affected_teams,
    select_milestone,
)
from litellm.types.proxy.model_deprecation import ModelDeprecationInfo


class TestSelectMilestone:
    @pytest.mark.parametrize(
        ("days_until", "expected"),
        [(45, None), (30, 30), (25, 30), (7, 7), (3, 7), (0, 0), (-12, 0)],
    )
    def test_should_pick_the_most_urgent_threshold_reached(self, days_until, expected):
        assert select_milestone(days_until, (30, 7, 0)) == expected

    def test_should_return_none_when_no_thresholds_configured(self):
        assert select_milestone(-5, ()) is None

    def test_should_not_depend_on_threshold_order(self):
        assert select_milestone(5, (0, 7, 30)) == 7


def _info(model_name: str, days_until: int, provider: str = "openai") -> ModelDeprecationInfo:
    return ModelDeprecationInfo(
        model_name=model_name,
        litellm_model=model_name,
        deprecation_date=date(2026, 10, 1),
        days_until_deprecation=days_until,
        status="deprecated" if days_until < 0 else ("imminent" if days_until <= 30 else "upcoming"),
        litellm_provider=provider,
    )


def _router(deployments: Sequence[Mapping[str, object]]) -> MagicMock:
    router: Final = MagicMock()
    router.get_model_list.return_value = deployments
    router.get_model_access_groups.return_value = {}
    router.model_group_alias = {}
    return router


def _team(team_id: str, models: Sequence[str] = (), blocked: bool = False) -> LiteLLM_TeamTable:
    return LiteLLM_TeamTable(team_id=team_id, models=list(models), blocked=blocked)


THRESHOLDS: Final = (30, 7, 0)


class TestResolveAffectedTeams:
    @pytest.mark.asyncio
    async def test_should_include_team_listing_the_model_explicitly(self):
        affected: Final = await resolve_affected_teams(
            (_info("gpt-old", 5),), _router([]), (_team("t1", ("gpt-old",)), _team("t2", ("other",))), THRESHOLDS
        )
        assert set(affected) == {"t1"}
        assert affected["t1"] == (AffectedModel(info=_info("gpt-old", 5), display_name="gpt-old", milestone=7),)

    @pytest.mark.asyncio
    async def test_should_include_unrestricted_team(self):
        affected: Final = await resolve_affected_teams((_info("gpt-old", 5),), _router([]), (_team("t1"),), THRESHOLDS)
        assert set(affected) == {"t1"}

    @pytest.mark.asyncio
    async def test_should_match_wildcard_allowlists(self):
        affected: Final = await resolve_affected_teams(
            (_info("openai/gpt-old", 5),),
            _router([]),
            (_team("t1", ("openai/*",)), _team("t2", ("azure/*",))),
            THRESHOLDS,
        )
        assert set(affected) == {"t1"}

    @pytest.mark.asyncio
    async def test_should_skip_blocked_teams(self):
        affected: Final = await resolve_affected_teams(
            (_info("gpt-old", 5),), _router([]), (_team("t1", blocked=True),), THRESHOLDS
        )
        assert dict(affected) == {}

    @pytest.mark.asyncio
    async def test_should_skip_models_that_have_not_reached_a_threshold(self):
        affected: Final = await resolve_affected_teams((_info("gpt-old", 45),), _router([]), (_team("t1"),), THRESHOLDS)
        assert dict(affected) == {}

    @pytest.mark.asyncio
    async def test_should_attribute_team_scoped_deployment_to_its_owner_only(self):
        deployments: Final = [
            {
                "model_name": "model_name_t2_abc",
                "litellm_params": {"model": "openai/gpt-old"},
                "model_info": {"id": "1", "team_id": "t2", "team_public_model_name": "gpt-old"},
            }
        ]
        affected: Final = await resolve_affected_teams(
            (_info("model_name_t2_abc", -3),), _router(deployments), (_team("t1"), _team("t2")), THRESHOLDS
        )
        assert set(affected) == {"t2"}
        assert affected["t2"][0].display_name == "gpt-old"
        assert affected["t2"][0].milestone == 0

    @pytest.mark.asyncio
    async def test_should_collect_every_reachable_model_per_team(self):
        affected: Final = await resolve_affected_teams(
            (_info("gpt-old", 5), _info("gpt-older", -1)),
            _router([]),
            (_team("t1", ("gpt-old", "gpt-older")),),
            THRESHOLDS,
        )
        assert [m.info.model_name for m in affected["t1"]] == ["gpt-old", "gpt-older"]

    @pytest.mark.asyncio
    async def test_should_treat_an_ordinary_deployment_as_shared(self):
        deployments: Final = [
            {"model_name": "gpt-old", "litellm_params": {"model": "openai/gpt-old"}, "model_info": {"id": "1"}}
        ]
        affected: Final = await resolve_affected_teams(
            (_info("gpt-old", 5),), _router(deployments), (_team("t1", ("gpt-old",)),), THRESHOLDS
        )
        assert set(affected) == {"t1"}
        assert affected["t1"][0].display_name == "gpt-old"

    @pytest.mark.asyncio
    async def test_should_fall_back_to_the_internal_name_when_a_scoped_deployment_has_no_public_name(self):
        deployments: Final = [
            {
                "model_name": "model_name_t2_abc",
                "litellm_params": {"model": "openai/gpt-old"},
                "model_info": {"id": "1", "team_id": "t2"},
            }
        ]
        affected: Final = await resolve_affected_teams(
            (_info("model_name_t2_abc", -3),), _router(deployments), (_team("t2"),), THRESHOLDS
        )
        assert affected["t2"][0].display_name == "model_name_t2_abc"

    @pytest.mark.asyncio
    async def test_should_deny_a_team_whose_allowlist_is_malformed_and_keep_going(self):
        affected: Final = await resolve_affected_teams(
            (_info("openai/gpt-old", 5),),
            _router([]),
            (_team("t1", ("openai/[*",)), _team("t2", ("openai/gpt-old",))),
            THRESHOLDS,
        )
        assert set(affected) == {"t2"}
