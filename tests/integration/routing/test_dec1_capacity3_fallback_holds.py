"""The hold a borrower takes on a router fallback target, including the empty hold when it has no limit there.

When a model group fails at the provider and the router falls back to another group, ``RouterFallbackRateLimitCheck``
(litellm/proxy/auth/fallback_rate_limit.py:38) asks the dynamic limiter's ``async_fallback_target_allowed``
(dynamic_rate_limiter_v3.py:1503). With ``commit`` the limiter first gives every borrower that has no per-model limit
on the target an empty hold (``_hold_fallback_for_borrowers_without_a_share``, :1588, the last fix of #51705). At
settlement (``_team_model_settlement``, :317) a served group with a hold returns the admitting reservation and charges
the target's usage to the hold's own descriptor, which for an empty hold is nothing. Before the fix the fallback's
usage was charged to the borrower's share on the model that had failed.

A borrower with a limit on the target takes a real hold there, counted on the target's model window and settled to
its usage. A borrower without one is not checked at all, so the target's model window is not counted (:1516).

Every provider answer reports 18 tokens. Primary deployments fail every call (``FlakyProvider(failures=-1)``) and the
router has ``num_retries: 0``.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Final

from pydantic import JsonValue

from tests.integration._support.client import Gateway, Scenario
from tests.integration._support.wire import wire_server
from tests.integration.authorization.dec1_limcap4_support import (
    PROVIDER_TOTAL,
    FlakyProvider,
    counter,
    counter_settles,
)
from tests.integration.routing.dec1_capacity3_support import borrower_scope, borrower_value
from tests.integration.routing.dec1_capacity_support import (
    MODEL_SCOPE,
    capacity_proxy,
    chat,
    entry,
    group_name,
    hashed,
    provider,
)


@contextmanager
def _fallback_rig(
    gateway: Gateway, directory: Path, *, borrow: list[str], threshold: float, max_borrow_multiple: float | None = None
) -> Iterator[tuple[Gateway, Scenario, str, str]]:
    """``primary`` (always failing) falls back to ``spare`` (healthy), both ``tpm: 100000, rpm: 100``, with the given
    borrowing entities and saturation threshold, on the shared Redis."""
    primary, spare = group_name("c3primary"), group_name("c3spare")
    borrowing: Final[dict[str, JsonValue]] = {
        "borrow_idle_capacity": list(borrow),
        **({"max_borrow_multiple": max_borrow_multiple} if max_borrow_multiple is not None else {}),
    }
    with (
        wire_server(FlakyProvider(failures=-1)) as broken,
        wire_server(provider()) as healthy,
        capacity_proxy(
            gateway,
            directory,
            model_list=[
                entry(primary, broken.url, rpm=100, tpm=100_000),
                entry(spare, healthy.url, rpm=100, tpm=100_000),
            ],
            priority_settings={
                "saturation_threshold": threshold,
                "saturation_check_cache_ttl": 0,
                "model_limit_borrowing_config": borrowing,
            },
            router_settings={"fallbacks": [{primary: [spare]}]},
            shared=True,
        ) as candidate,
        candidate.scenario() as scenario,
    ):
        yield candidate, scenario, primary, spare


def _team_key(scenario: Scenario, limits: Mapping[str, int], **metadata: JsonValue) -> tuple[str, str]:
    team: Final = scenario.team(metadata={"model_tpm_limit": dict(limits), **metadata})
    return team, scenario.key(team_id=team)


def _spare_settled(spare: str) -> None:
    assert counter_settles(MODEL_SCOPE, spare, "tokens", PROVIDER_TOTAL) == PROVIDER_TOTAL


def test_a_fallback_to_a_model_the_team_has_no_limit_on_charges_nothing_to_its_share_on_the_failed_model(
    gateway: Gateway, tmp_path: Path
) -> None:
    """Teams borrow on a saturated proxy (``saturation_threshold: 0.0``) and the team has ``model_tpm_limit`` only on
    ``primary``. ``primary`` fails and ``spare`` answers 18 tokens. Once ``spare``'s model window settled, the team's
    share counter on the failed model ``{model_per_team_saturated:<team_id>:<primary>}:tokens`` holds 0: the admission
    hold was returned and ``spare``'s usage went to the empty hold (:1588). Before #51705 it held 18."""
    with _fallback_rig(gateway, tmp_path, borrow=["team"], threshold=0.0) as (candidate, scenario, primary, spare):
        team, key = _team_key(scenario, {primary: 1000})
        assert chat(candidate, primary, key).status_code == 200
        _spare_settled(spare)
        assert counter(borrower_scope("team"), borrower_value(team, primary), "tokens") == 0


def test_the_failed_models_share_keeps_the_one_admitted_request_after_a_fallback(
    gateway: Gateway, tmp_path: Path
) -> None:
    """Same rig. The empty hold returns tokens only: ``{model_per_team_saturated:<team_id>:<primary>}:requests`` still
    holds the 1 request admission counted (:1287), so the failed attempt keeps using the team's request share."""
    with _fallback_rig(gateway, tmp_path, borrow=["team"], threshold=0.0) as (candidate, scenario, primary, spare):
        team, key = _team_key(scenario, {primary: 1000})
        assert chat(candidate, primary, key).status_code == 200
        _spare_settled(spare)
        assert counter(borrower_scope("team"), borrower_value(team, primary), "requests") == 1


def test_a_fallback_target_without_a_team_limit_is_not_counted_on_its_model_window(
    gateway: Gateway, tmp_path: Path
) -> None:
    """Same rig. ``async_fallback_target_allowed`` takes the empty hold and then returns before counting, because the
    team has no share on ``spare`` (:1514-1517). ``{model_saturation_check:<spare>}:requests`` is never written even
    though ``spare`` served and settled 18 tokens on its window."""
    with _fallback_rig(gateway, tmp_path, borrow=["team"], threshold=0.0) as (candidate, scenario, primary, spare):
        _team, key = _team_key(scenario, {primary: 1000})
        assert chat(candidate, primary, key).status_code == 200
        _spare_settled(spare)
        assert counter(MODEL_SCOPE, spare, "requests") is None


def test_a_fallback_to_a_model_the_key_has_no_limit_on_charges_nothing_to_the_keys_share(
    gateway: Gateway, tmp_path: Path
) -> None:
    """Keys borrow (``borrow_idle_capacity: ["key"]``) on a saturated proxy and the key has ``model_tpm_limit`` only on
    ``primary``. After the fallback to ``spare`` the key's share ``{model_per_key_saturated:<hash>:<primary>}:tokens``
    holds 0: the empty hold covers every borrower kind (:1595), not only teams."""
    with _fallback_rig(gateway, tmp_path, borrow=["key"], threshold=0.0) as (candidate, scenario, primary, spare):
        key: Final = scenario.key(model_tpm_limit={primary: 1000})
        assert chat(candidate, primary, key).status_code == 200
        _spare_settled(spare)
        assert counter(borrower_scope("key"), borrower_value(hashed(key), primary), "tokens") == 0


def test_an_idle_teams_borrow_cap_on_the_failed_model_is_returned_after_a_fallback(
    gateway: Gateway, tmp_path: Path
) -> None:
    """Teams borrow on an idle proxy (``saturation_threshold: 0.9``) with ``max_borrow_multiple: 2``, so admission
    holds tokens on the team's borrow cap counter rather than its share (:260-266). The team has a limit only on
    ``primary``. After the fallback the cap counter ``{model_per_team_max_borrow_multiple:<team_id>:<primary>}:tokens``
    holds 0, not ``spare``'s 18."""
    with _fallback_rig(gateway, tmp_path, borrow=["team"], threshold=0.9, max_borrow_multiple=2) as (
        candidate,
        scenario,
        primary,
        spare,
    ):
        team, key = _team_key(scenario, {primary: 1000})
        assert chat(candidate, primary, key).status_code == 200
        _spare_settled(spare)
        assert counter(borrower_scope("team", capped=True), borrower_value(team, primary), "tokens") == 0


def test_a_fallback_to_a_model_the_team_has_a_limit_on_settles_on_the_targets_share(
    gateway: Gateway, tmp_path: Path
) -> None:
    """The contrast. The team has ``model_tpm_limit`` on both groups on a saturated proxy. After the fallback the team's
    share on ``spare`` settles to the answer, ``{model_per_team_saturated:<team_id>:<spare>}:tokens`` = 18, and its
    share on ``primary`` is back to 0 (:338-347)."""
    with _fallback_rig(gateway, tmp_path, borrow=["team"], threshold=0.0) as (candidate, scenario, primary, spare):
        team, key = _team_key(scenario, {primary: 1000, spare: 1000})
        assert chat(candidate, primary, key).status_code == 200
        target: Final = counter_settles(borrower_scope("team"), borrower_value(team, spare), "tokens", PROVIDER_TOTAL)
        assert target == PROVIDER_TOTAL
        assert counter(borrower_scope("team"), borrower_value(team, primary), "tokens") == 0


def test_a_fallback_to_a_model_the_team_has_a_limit_on_is_counted_on_the_targets_share_requests(
    gateway: Gateway, tmp_path: Path
) -> None:
    """Same contrast. The committed hold on ``spare`` adds one request to the team's share there:
    ``{model_per_team_saturated:<team_id>:<spare>}:requests`` = 1 (:1550-1556)."""
    with _fallback_rig(gateway, tmp_path, borrow=["team"], threshold=0.0) as (candidate, scenario, primary, spare):
        team, key = _team_key(scenario, {primary: 1000, spare: 1000})
        assert chat(candidate, primary, key).status_code == 200
        _spare_settled(spare)
        assert counter(borrower_scope("team"), borrower_value(team, spare), "requests") == 1
