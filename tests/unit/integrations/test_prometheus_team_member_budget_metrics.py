"""Team member budget gauges: one series per (team, member) with an effective budget."""

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from types import MappingProxyType
from typing import Final
from unittest.mock import MagicMock

import pytest
from prometheus_client import REGISTRY

import litellm
from litellm.integrations.prometheus import PrometheusLogger, TeamMemberSweepIO
from litellm.models.budget import LiteLLM_BudgetTable, LiteLLM_BudgetTableFull
from litellm.models.team import LiteLLM_TeamTable
from litellm.models.team_membership import LiteLLM_TeamMembership
from litellm.proxy import proxy_server
from litellm.proxy.spend_tracking.team_member_budget import hours_until
from litellm.redis.stores.spend_counters.types import team_member_counter
from litellm.types.proxy.carried_budget_state import TeamMemberBudgetSnapshot

NOW: Final = datetime(2026, 10, 1, 12, 0, tzinfo=timezone.utc)
MAX_METRIC: Final = "litellm_team_member_max_budget_metric"
REMAINING_METRIC: Final = "litellm_remaining_team_member_budget_metric"
HOURS_METRIC: Final = "litellm_team_member_budget_remaining_hours_metric"
ALICE_IN_SEARCH: Final = {
    "team": "t-search",
    "team_alias": "search",
    "user": "alice",
    "user_email": "alice@example.com",
}


def _unregister_everything() -> None:
    for collector in tuple(REGISTRY._collector_to_names):
        REGISTRY.unregister(collector)


@pytest.fixture(autouse=True)
def clean_registry(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(litellm, "prometheus_metrics_config", None)
    _unregister_everything()
    yield
    _unregister_everything()


def _sample(metric: str, labels: Mapping[str, str]) -> float | None:
    return REGISTRY.get_sample_value(metric, dict(labels))


def _set_alice(logger: PrometheusLogger, budget_reset_at: datetime | None) -> None:
    logger._set_team_member_budget_metrics(
        team_id="t-search",
        team_alias="search",
        user_id="alice",
        user_email="alice@example.com",
        max_budget=10.0,
        spend=4.0,
        budget_reset_at=budget_reset_at,
        now=NOW,
    )


def test_setter_emits_three_series_with_team_and_user_labels() -> None:
    logger: Final = PrometheusLogger()

    _set_alice(logger, budget_reset_at=NOW + timedelta(minutes=150))

    assert {
        MAX_METRIC: _sample(MAX_METRIC, ALICE_IN_SEARCH),
        REMAINING_METRIC: _sample(REMAINING_METRIC, ALICE_IN_SEARCH),
        HOURS_METRIC: _sample(HOURS_METRIC, ALICE_IN_SEARCH),
    } == {MAX_METRIC: 10.0, REMAINING_METRIC: 6.0, HOURS_METRIC: pytest.approx(2.5)}


def test_member_without_reset_time_gets_no_hours_series() -> None:
    logger: Final = PrometheusLogger()

    _set_alice(logger, budget_reset_at=None)

    assert _sample(MAX_METRIC, ALICE_IN_SEARCH) == 10.0
    assert _sample(REMAINING_METRIC, ALICE_IN_SEARCH) == 6.0
    assert _sample(HOURS_METRIC, ALICE_IN_SEARCH) is None


def test_include_labels_trims_member_series_to_the_configured_labels(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        litellm,
        "prometheus_metrics_config",
        [
            {"group": "member-max", "metrics": [MAX_METRIC], "include_labels": ["team", "user"]},
            {"group": "member-remaining", "metrics": [REMAINING_METRIC]},
            {"group": "member-hours", "metrics": [HOURS_METRIC], "include_labels": ["user"]},
        ],
    )
    logger: Final = PrometheusLogger()

    _set_alice(logger, budget_reset_at=NOW + timedelta(hours=1))

    assert _sample(MAX_METRIC, {"team": "t-search", "user": "alice"}) == 10.0
    assert _sample(REMAINING_METRIC, ALICE_IN_SEARCH) == 6.0
    assert _sample(HOURS_METRIC, {"user": "alice"}) == pytest.approx(1.0)


MEMBER_GAUGES_ONLY: Final = [{"group": "team-member-budgets", "metrics": [MAX_METRIC, REMAINING_METRIC, HOURS_METRIC]}]
MEMBER_RESET_AT: Final = datetime(2026, 10, 2, 0, 0, tzinfo=timezone.utc)


async def _log_alice_request(logger: PrometheusLogger, carried: Mapping[str, object], response_cost: float) -> None:
    await logger._increment_remaining_budget_metrics(
        user_api_team="t-search",
        user_api_team_alias="search",
        user_api_key="hashed",
        user_api_key_alias=None,
        litellm_params={"metadata": {**carried, "user_api_key_user_email": "alice@example.com"}},
        response_cost=response_cost,
        user_id="alice",
    )


@pytest.mark.asyncio
async def test_member_gauge_spend_equals_the_enforcement_counter_plus_cost(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "prometheus_metrics_config", MEMBER_GAUGES_ONLY)
    logger: Final = PrometheusLogger()
    carried: Final = TeamMemberBudgetSnapshot(max_budget=10.0, spend=3.0, budget_reset_at=MEMBER_RESET_AT)

    before: Final = datetime.now(timezone.utc)
    await _log_alice_request(logger, carried.metadata_entries(), response_cost=0.5)
    after: Final = datetime.now(timezone.utc)

    assert _sample(MAX_METRIC, ALICE_IN_SEARCH) == 10.0
    assert _sample(REMAINING_METRIC, ALICE_IN_SEARCH) == 6.5
    hours: Final = _sample(HOURS_METRIC, ALICE_IN_SEARCH)
    assert hours is not None
    assert hours_until(MEMBER_RESET_AT, after) <= hours <= hours_until(MEMBER_RESET_AT, before)


@pytest.mark.asyncio
async def test_request_without_member_snapshot_emits_no_member_series(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "prometheus_metrics_config", MEMBER_GAUGES_ONLY)
    logger: Final = PrometheusLogger()

    await _log_alice_request(logger, {}, response_cost=0.5)

    assert {
        MAX_METRIC: _sample(MAX_METRIC, ALICE_IN_SEARCH),
        REMAINING_METRIC: _sample(REMAINING_METRIC, ALICE_IN_SEARCH),
        HOURS_METRIC: _sample(HOURS_METRIC, ALICE_IN_SEARCH),
    } == {MAX_METRIC: None, REMAINING_METRIC: None, HOURS_METRIC: None}


@pytest.mark.asyncio
@pytest.mark.parametrize(("team_id", "user_id"), [(None, "alice"), ("t-search", None)])
async def test_member_snapshot_without_team_or_user_emits_no_member_series(
    monkeypatch: pytest.MonkeyPatch, team_id: str | None, user_id: str | None
) -> None:
    monkeypatch.setattr(litellm, "prometheus_metrics_config", MEMBER_GAUGES_ONLY)
    logger: Final = PrometheusLogger()
    carried: Final = TeamMemberBudgetSnapshot(max_budget=10.0, spend=3.0, budget_reset_at=None)

    await logger._increment_remaining_budget_metrics(
        user_api_team=team_id,
        user_api_team_alias="search",
        user_api_key="hashed",
        user_api_key_alias=None,
        litellm_params={"metadata": dict(carried.metadata_entries())},
        response_cost=0.5,
        user_id=user_id,
    )

    samples_by_family: Final = {family.name: family.samples for family in REGISTRY.collect()}
    assert samples_by_family[MAX_METRIC] == []


SEARCH_DEFAULT: Final = "search-default"
ADS_DEFAULT: Final = "ads-default"


def _team(team_id: str, alias: str, default_budget_id: str | None) -> LiteLLM_TeamTable:
    metadata: Final = {} if default_budget_id is None else {"team_member_budget_id": default_budget_id}
    return LiteLLM_TeamTable(team_id=team_id, team_alias=alias, metadata=metadata)


def _member(user_id: str, team_id: str, spend: float = 0.0, own_budget: float | None = None) -> LiteLLM_TeamMembership:
    own_row: Final = (
        None
        if own_budget is None
        else LiteLLM_BudgetTableFull(
            budget_id=f"{user_id}-{team_id}",
            max_budget=own_budget,
            budget_reset_at=NOW + timedelta(hours=3),
            created_at=NOW - timedelta(days=30),
        )
    )
    return LiteLLM_TeamMembership(
        user_id=user_id,
        team_id=team_id,
        spend=spend,
        budget_id=own_row.budget_id if own_row is not None else None,
        litellm_budget_table=own_row,
    )


@dataclass(frozen=True, slots=True)
class _Directory:
    memberships: tuple[LiteLLM_TeamMembership, ...]
    teams: tuple[LiteLLM_TeamTable, ...] = (
        _team("t-search", "search", SEARCH_DEFAULT),
        _team("t-ads", "ads", None),
    )
    defaults: Mapping[str, LiteLLM_BudgetTable] = MappingProxyType(
        {SEARCH_DEFAULT: LiteLLM_BudgetTable(budget_id=SEARCH_DEFAULT, max_budget=5.0)}
    )
    counters: Mapping[str, float] | None = None
    failing_defaults: frozenset[str] = frozenset()

    def sweep_io(self) -> TeamMemberSweepIO:
        async def fetch_page(page_size: int, page: int) -> tuple[Sequence[LiteLLM_TeamMembership], int | None]:
            start: Final = (page - 1) * page_size
            return self.memberships[start : start + page_size], len(self.memberships) if page == 1 else None

        async def fetch_teams(team_ids: Sequence[str]) -> Mapping[str, LiteLLM_TeamTable]:
            return MappingProxyType({team.team_id: team for team in self.teams if team.team_id in team_ids})

        async def fetch_default(budget_id: str) -> LiteLLM_BudgetTable | None:
            if budget_id in self.failing_defaults:
                raise ConnectionError(f"budget table unreachable for {budget_id}")
            return self.defaults.get(budget_id)

        async def fetch_emails(user_ids: Sequence[str]) -> Mapping[str, str | None]:
            return MappingProxyType({user_id: f"{user_id}@example.com" for user_id in user_ids})

        async def read_counters(counter_keys: Sequence[str]) -> Mapping[str, float | None] | None:
            counters: Final = self.counters
            return None if counters is None else MappingProxyType({key: counters.get(key) for key in counter_keys})

        return TeamMemberSweepIO(
            fetch_page=fetch_page,
            fetch_teams=fetch_teams,
            fetch_default=fetch_default,
            fetch_emails=fetch_emails,
            read_counters=read_counters,
        )


def _labels(user_id: str, team_id: str, alias: str) -> Mapping[str, str]:
    return {"team": team_id, "team_alias": alias, "user": user_id, "user_email": f"{user_id}@example.com"}


def _series(metric: str) -> Mapping[tuple[str, str], float]:
    samples_by_family: Final = {family.name: family.samples for family in REGISTRY.collect()}
    return MappingProxyType(
        {(sample.labels["team"], sample.labels["user"]): sample.value for sample in samples_by_family[metric]}
    )


@pytest.fixture
def db_connected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(  # test-quality-ok: [TQ008] the shared budget pager only runs when the proxy has a DB
        proxy_server, "prisma_client", object()
    )


@pytest.mark.asyncio
async def test_sweep_emits_one_series_per_team_and_omits_unbudgeted(db_connected: None) -> None:
    directory: Final = _Directory(
        memberships=(
            _member("alice", "t-search", spend=2.0, own_budget=20.0),
            _member("bob", "t-search", spend=1.0),
            _member("carol", "t-search", spend=0.5),
            _member("carol", "t-ads", spend=0.25),
        )
    )

    await PrometheusLogger()._sweep_team_member_budget_metrics(directory.sweep_io(), now=NOW)

    assert _series(MAX_METRIC) == {("t-search", "alice"): 20.0, ("t-search", "bob"): 5.0, ("t-search", "carol"): 5.0}
    assert _series(REMAINING_METRIC) == {
        ("t-search", "alice"): 18.0,
        ("t-search", "bob"): 4.0,
        ("t-search", "carol"): 4.5,
    }
    assert _series(HOURS_METRIC) == {("t-search", "alice"): pytest.approx(3.0)}
    assert _sample(MAX_METRIC, _labels("bob", "t-search", "search")) == 5.0


@pytest.mark.asyncio
async def test_sweep_spend_is_the_higher_of_counter_and_stored(db_connected: None) -> None:
    directory: Final = _Directory(
        memberships=(_member("alice", "t-search", spend=2.0, own_budget=20.0), _member("bob", "t-search", spend=3.0)),
        counters=MappingProxyType(
            {team_member_counter("alice", "t-search"): 7.0, team_member_counter("bob", "t-search"): 1.0}
        ),
    )

    await PrometheusLogger()._sweep_team_member_budget_metrics(directory.sweep_io(), now=NOW)

    assert _series(REMAINING_METRIC) == {("t-search", "alice"): 13.0, ("t-search", "bob"): 2.0}


@pytest.mark.asyncio
async def test_sweep_uses_stored_spend_without_redis(db_connected: None) -> None:
    directory: Final = _Directory(memberships=(_member("bob", "t-search", spend=3.0),), counters=None)

    await PrometheusLogger()._sweep_team_member_budget_metrics(directory.sweep_io(), now=NOW)

    assert _series(REMAINING_METRIC) == {("t-search", "bob"): 2.0}


@pytest.mark.asyncio
async def test_sweep_covers_every_page(db_connected: None) -> None:
    directory: Final = _Directory(memberships=tuple(_member(f"user-{index:03d}", "t-search") for index in range(123)))

    await PrometheusLogger()._sweep_team_member_budget_metrics(directory.sweep_io(), now=NOW)

    assert len(_series(MAX_METRIC)) == 123


@pytest.mark.asyncio
async def test_sweep_with_member_gauges_disabled_makes_no_queries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        litellm, "prometheus_metrics_config", [{"group": "core-only", "metrics": ["litellm_requests_metric"]}]
    )
    prisma: Final = MagicMock()
    monkeypatch.setattr(  # test-quality-ok: [TQ008] the sweep reads the proxy_server prisma global, no seam
        proxy_server, "prisma_client", prisma
    )

    await PrometheusLogger()._initialize_team_member_budget_metrics()

    assert prisma.mock_calls == []


@pytest.mark.asyncio
async def test_sweep_keeps_going_when_one_team_default_lookup_fails(db_connected: None) -> None:
    directory: Final = _Directory(
        memberships=(
            _member("alice", "t-search", own_budget=20.0),
            _member("bob", "t-search"),
            _member("carol", "t-ads"),
        ),
        teams=(_team("t-search", "search", SEARCH_DEFAULT), _team("t-ads", "ads", ADS_DEFAULT)),
        defaults=MappingProxyType({ADS_DEFAULT: LiteLLM_BudgetTable(budget_id=ADS_DEFAULT, max_budget=4.0)}),
        failing_defaults=frozenset({SEARCH_DEFAULT}),
    )

    await PrometheusLogger()._sweep_team_member_budget_metrics(directory.sweep_io(), now=NOW)

    assert _series(MAX_METRIC) == {("t-search", "alice"): 20.0, ("t-ads", "carol"): 4.0}
