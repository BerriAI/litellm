"""Share math and the active-team registry behind
``priority_reservation_settings.priority_class_share_config``."""

from typing import Final

import pytest

from litellm.caching.caching import DualCache
from litellm.proxy.hooks.priority_class_share import ActiveTeamRegistry, pool_member, team_share

WEIGHTS: Final = {"interactive": 0.6, "batch": 0.4}


def test_two_interactive_and_four_batch_teams_split_the_model_sixty_forty():
    members: Final = {
        "i1": "interactive",
        "i2": "interactive",
        "b1": "batch",
        "b2": "batch",
        "b3": "batch",
        "b4": "batch",
    }
    interactive: Final = team_share("interactive", members, WEIGHTS, default_weight=0.2)
    batch: Final = team_share("batch", members, WEIGHTS, default_weight=0.2)
    assert int(500_000_000 * interactive.fraction) == 150_000_000
    assert int(500_000_000 * batch.fraction) == 50_000_000
    assert (interactive.active_teams_in_class, interactive.active_classes) == (2, 2)


def test_only_active_class_takes_the_whole_model():
    members: Final = {"b1": "batch", "b2": "batch", "b3": "batch", "b4": "batch"}
    assert int(500_000_000 * team_share("batch", members, WEIGHTS, default_weight=0.2).fraction) == 125_000_000


def test_a_single_active_team_gets_everything():
    share: Final = team_share("interactive", {"i1": "interactive"}, WEIGHTS, default_weight=0.2)
    assert share.fraction == 1.0
    assert (share.active_teams_in_class, share.active_classes) == (1, 1)


def test_a_class_without_a_configured_weight_uses_the_default_weight():
    members: Final = {"i1": "interactive", "d1": "default"}
    assert team_share("default", members, WEIGHTS, default_weight=0.2).fraction == pytest.approx(0.2 / 0.8)
    assert team_share("interactive", members, WEIGHTS, default_weight=0.2).fraction == pytest.approx(0.6 / 0.8)


def test_active_team_fractions_always_sum_to_one():
    members: Final = {"i1": "interactive", "i2": "interactive", "b1": "batch", "d1": "default", "d2": "default"}
    total: Final = sum(team_share(cls, members, WEIGHTS, default_weight=0.2).fraction for cls in members.values())
    assert total == pytest.approx(1.0)


def test_zero_weights_split_the_model_equally_between_classes():
    members: Final = {"a": "interactive", "b": "batch"}
    assert team_share("batch", members, {"interactive": 0.0, "batch": 0.0}, default_weight=0.0).fraction == 0.5


class _Clock:
    def __init__(self) -> None:
        self.now = 1_000.0

    def __call__(self) -> float:
        return self.now


@pytest.mark.asyncio
async def test_registry_lists_every_team_seen_in_the_window_oldest_first():
    clock: Final = _Clock()
    registry: Final = ActiveTeamRegistry(DualCache(), time_provider=clock)
    assert dict(await registry.touch("m", "a", "interactive", window=60)) == {"a": "interactive"}
    clock.now += 10
    assert dict(await registry.touch("m", "b", "batch", window=60)) == {"a": "interactive", "b": "batch"}


@pytest.mark.asyncio
async def test_registry_forgets_a_team_once_its_last_request_leaves_the_window():
    clock: Final = _Clock()
    registry: Final = ActiveTeamRegistry(DualCache(), time_provider=clock)
    await registry.touch("m", "a", "interactive", window=60)
    clock.now += 59
    assert "a" in await registry.touch("m", "b", "batch", window=60)
    clock.now += 1
    assert dict(await registry.touch("m", "b", "batch", window=60)) == {"b": "batch"}


@pytest.mark.asyncio
async def test_registry_refreshes_a_teams_last_seen_time_on_every_request():
    clock: Final = _Clock()
    registry: Final = ActiveTeamRegistry(DualCache(), time_provider=clock)
    await registry.touch("m", "a", "interactive", window=60)
    clock.now += 40
    await registry.touch("m", "a", "interactive", window=60)
    clock.now += 40
    assert "a" in await registry.touch("m", "b", "batch", window=60)


@pytest.mark.asyncio
async def test_registry_keeps_models_apart():
    registry: Final = ActiveTeamRegistry(DualCache(), time_provider=_Clock())
    await registry.touch("m1", "a", "interactive", window=60)
    assert dict(await registry.touch("m2", "b", "batch", window=60)) == {"b": "batch"}


@pytest.mark.asyncio
async def test_registry_keeps_a_teams_newest_class_when_it_changed_class_mid_window():
    clock: Final = _Clock()
    registry: Final = ActiveTeamRegistry(DualCache(), time_provider=clock)
    await registry.touch("m", "a", "interactive", window=60)
    clock.now += 1
    assert dict(await registry.touch("m", "a", "batch", window=60)) == {"a": "batch"}


@pytest.mark.asyncio
async def test_registry_lists_the_caller_under_the_class_it_just_sent_with_even_when_another_worker_wrote_later():
    clock: Final = _Clock()
    registry: Final = ActiveTeamRegistry(DualCache(), time_provider=clock)
    clock.now += 5
    await registry.touch("m", "a", "batch", window=60)
    clock.now -= 5
    members: Final = await registry.touch("m", "a", "interactive", window=60)
    assert dict(members) == {"a": "interactive"}
    assert team_share("interactive", members, WEIGHTS, 1.0).fraction == 1.0


@pytest.mark.asyncio
async def test_registry_keeps_a_team_id_that_contains_the_separator_whole():
    registry: Final = ActiveTeamRegistry(DualCache(), time_provider=_Clock())
    members: Final = await registry.touch("m", "owned\x1fextra", "batch", window=60)
    assert dict(members) == {"owned\x1fextra": "batch"}
    assert team_share("batch", members, {"interactive": 0.6, "batch": 0.4}, 1.0).fraction == 1.0


def test_a_pool_member_keeps_its_class_present_without_taking_a_team_slice():
    members: Final = {pool_member("interactive"): "interactive", "a": "interactive", "c": "batch"}
    share: Final = team_share("interactive", members, {"interactive": 0.6, "batch": 0.4}, 1.0)
    assert share.fraction == 0.6
    assert share.active_teams_in_class == 1
    assert share.active_classes == 2


@pytest.mark.asyncio
async def test_registry_keeps_pool_markers_of_different_classes_apart():
    registry: Final = ActiveTeamRegistry(DualCache(), time_provider=_Clock())
    await registry.touch("m", pool_member("interactive"), "interactive", window=60)
    members: Final = await registry.touch("m", pool_member("batch"), "batch", window=60)
    assert set(members.values()) == {"interactive", "batch"}


@pytest.mark.asyncio
async def test_registry_keeps_a_team_that_was_refreshed_after_the_first_window_opened():
    clock: Final = _Clock()
    registry: Final = ActiveTeamRegistry(DualCache(), time_provider=clock)
    await registry.touch("m", "a", "interactive", window=60)
    clock.now += 50
    await registry.touch("m", "a", "interactive", window=60)
    clock.now += 50
    assert dict(await registry.touch("m", "b", "batch", window=60)) == {"a": "interactive", "b": "batch"}
