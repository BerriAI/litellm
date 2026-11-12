"""Share math and the active-team registry behind guaranteed team shares."""

import pytest

from litellm import DualCache
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.hooks.team_shares import TeamShareRegistry, compute_team_share, team_burst_multiplier

WEIGHTS = {"interactive": 0.6, "batch": 0.4}


@pytest.mark.parametrize(
    ("members", "team", "expected_fraction"),
    [
        (
            {
                "assistant": "interactive",
                "api": "interactive",
                "state": "batch",
                "dec": "batch",
                "wt": "batch",
                "paa": "batch",
            },
            "assistant",
            0.3,
        ),
        (
            {
                "assistant": "interactive",
                "api": "interactive",
                "state": "batch",
                "dec": "batch",
                "wt": "batch",
                "paa": "batch",
            },
            "state",
            0.1,
        ),
        ({"state": "batch", "dec": "batch"}, "state", 0.5),
        ({"state": "batch"}, "state", 1.0),
        ({"assistant": "interactive", "state": "batch"}, "assistant", 0.6),
        ({"assistant": "interactive", "state": "batch"}, "state", 0.4),
    ],
)
def test_share_is_class_weight_over_present_classes_split_equally_among_its_teams(members, team, expected_fraction):
    share = compute_team_share(team_id=team, members=members, class_weights=WEIGHTS, default_weight=0.25)
    assert share.fraction == pytest.approx(expected_fraction)


def test_shares_of_all_active_teams_sum_to_the_whole_model():
    members = {"a": "interactive", "b": "interactive", "c": "batch", "d": "batch", "e": "batch", "f": "default"}
    total = sum(
        compute_team_share(team_id=team, members=members, class_weights=WEIGHTS, default_weight=0.25).fraction
        for team in members
    )
    assert total == pytest.approx(1.0)


def test_share_in_tokens_scales_model_tpm_and_reports_active_counts():
    members = {"assistant": "interactive", "state": "batch", "dec": "batch"}
    share = compute_team_share(team_id="state", members=members, class_weights=WEIGHTS, default_weight=0.25)
    assert share.tokens(500_000_000) == 100_000_000
    assert share.requests(None) is None
    assert (share.active_teams_in_class, share.active_classes) == (2, 2)


def test_unknown_class_uses_default_weight():
    members = {"assistant": "interactive", "mystery": "default"}
    share = compute_team_share(team_id="mystery", members=members, class_weights=WEIGHTS, default_weight=0.2)
    assert share.fraction == pytest.approx(0.2 / 0.8)


@pytest.mark.parametrize(
    ("team_metadata", "default", "expected"),
    [
        ({"max_borrow_multiplier": 2}, None, 2.0),
        ({"max_borrow_multiplier": 0.5}, None, 1.0),
        ({}, 3.0, 3.0),
        ({}, None, None),
        ({"max_borrow_multiplier": "2"}, 1.5, 1.5),
        ({"max_borrow_multiplier": True}, None, None),
    ],
)
def test_burst_multiplier_comes_from_team_metadata_then_default_and_never_drops_below_one(
    team_metadata, default, expected
):
    user = UserAPIKeyAuth(team_id="t", team_metadata=team_metadata)
    assert team_burst_multiplier(user, default) == expected


@pytest.mark.asyncio
async def test_registry_lists_every_team_seen_inside_the_window_with_its_class():
    registry = TeamShareRegistry(DualCache(), window_size=60, local_ttl=60)
    await registry.members("luna", team_id="assistant", priority="interactive", now=1000.0)
    await registry.members("luna", team_id="state", priority="batch", now=1010.0)
    members = await registry.members("luna", team_id="dec", priority="batch", now=1020.0)
    assert dict(members) == {"assistant": "interactive", "state": "batch", "dec": "batch"}


@pytest.mark.asyncio
async def test_registry_drops_a_team_idle_for_a_whole_window():
    registry = TeamShareRegistry(DualCache(), window_size=60, local_ttl=60)
    await registry.members("luna", team_id="assistant", priority="interactive", now=1000.0)
    await registry.members("luna", team_id="state", priority="batch", now=1050.0)
    members = await registry.members("luna", team_id="state", priority="batch", now=1061.0)
    assert dict(members) == {"state": "batch"}


@pytest.mark.asyncio
async def test_registry_keeps_a_team_that_sends_at_least_once_per_window():
    registry = TeamShareRegistry(DualCache(), window_size=60, local_ttl=60)
    for now in (1000.0, 1030.0, 1060.0, 1090.0):
        await registry.members("luna", team_id="assistant", priority="interactive", now=now)
    members = await registry.members("luna", team_id="state", priority="batch", now=1100.0)
    assert dict(members) == {"assistant": "interactive", "state": "batch"}


@pytest.mark.asyncio
async def test_registry_follows_a_team_that_changes_class():
    registry = TeamShareRegistry(DualCache(), window_size=60, local_ttl=60)
    await registry.members("luna", team_id="assistant", priority="batch", now=1000.0)
    members = await registry.members("luna", team_id="assistant", priority="interactive", now=1001.0)
    assert dict(members) == {"assistant": "interactive"}


@pytest.mark.asyncio
async def test_registry_is_shared_between_handlers_on_one_cache_and_scoped_per_model():
    cache = DualCache()
    first = TeamShareRegistry(cache, window_size=60, local_ttl=60)
    second = TeamShareRegistry(cache, window_size=60, local_ttl=60)
    await first.members("luna", team_id="assistant", priority="interactive", now=1000.0)
    await second.members("opus", team_id="paa", priority="batch", now=1000.0)
    luna = await second.members("luna", team_id="state", priority="batch", now=1001.0)
    assert dict(luna) == {"assistant": "interactive", "state": "batch"}
