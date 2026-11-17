"""Contract: what the scheduled budget reset does to every budgeted level, in Postgres and in Redis.

`ResetBudgetJob.reset_budget` (`litellm/proxy/common_utils/reset_budget_job.py:782-809`) runs on one pod per tick
(a Redis lease) and walks, in order: keys, users and teams with their own `budget_reset_at`
(`_reset_budget_for_litellm_keys_chunk` and siblings, `:1527-1880`), then every due `LiteLLM_BudgetTable` row and
everything linked to it (`_reset_expired_budget_cascade`, `:1316-1348`): team memberships, keys without their own
`budget_duration`, organizations, projects, tags, model access groups, customer organizations and end users, then
the `budget_limits` windows (`reset_budget_windows`, covered in `test_dec1_budgets4_window_resets.py`).

For every reset row the job writes `spend = 0` (or the overage above `max_budget` with
`litellm_settings.budget_rollover`, `:238-257`) and the next calendar boundary as `budget_reset_at`, commits, and
only then deletes the row's Redis counter and its charges-only twin (`_invalidate_spend_counter`, `:865-882`, and
`_invalidate_caches`, `:917-946`). Deleting, not zeroing, is the contract: the next request reseeds the counter from
the committed row (`spend_counter_reseed.py::coalesced`), the only value that includes increments that raced the
reset. Rows linked to a due budget row are reset only when their spend is above 0 (`_SPENT_ROWS_WHERE`, `:189`), and
a key with its own `budget_duration` is never reset by its budget row (`_LINKED_KEYS_WHERE`, `:188`).

Each test drives one reset by moving a `budget_reset_at` into the past on an owned proxy whose job ticks every 2 to
3 seconds (see `dec1_budgets4_support`). Price: one served call 0.02 (`hold_model`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pytest
from pydantic import JsonValue

from tests.integration._support.client import Gateway, eventually
from tests.integration.spend.dec1_budgets3_support import (
    end_user_counter,
    org_counter,
    project_counter,
    tag_counter,
    team_counter,
    team_member_counter,
    user_counter,
)
from tests.integration.spend.dec1_budgets4_support import (
    RESET_SECONDS,
    Due,
    budget_id_of,
    committed,
    make_budget_row_due,
    make_due,
    next_utc_midnight,
    reset_proxy,
    wait_for_row_reset,
)
from tests.integration.spend.dec1_budgets_support import (
    CHAT_COST,
    background,
    call,
    counter,
    counter_settles,
    held_provider,
    hold_model,
    key_counter,
    redis_client,
    refused,
)
from tests.integration.spend.dec1_spend_support import (
    hashed,
    new_customer,
    new_tag,
    scripted_upstream,
    served,
    spend_of,
    spend_reaches,
)

pytestmark: Final = pytest.mark.shares_worker_state


def _project_spend(rows: list[dict[str, JsonValue]]) -> float | None:
    value: Final = rows[0]["spend"] if rows else None
    return None if value is None else float(str(value))


@pytest.mark.timeout(300)
def test_a_due_key_reset_zeroes_its_stored_spend_and_deletes_its_counter(gateway: Gateway, tmp_path: Path) -> None:
    """A key with `max_budget` 1.0 and `budget_duration` `1d` is served once (stored spend and `spend:key:<hash>`
    both 0.02). Its `budget_reset_at` is moved into the past. After the next tick the stored spend is exactly 0 and
    the Redis counter is absent, not 0: the job deletes it after the commit (`reset_budget_job.py:1576-1580`,
    `:865-882`) so the next read reseeds from the committed row."""
    with (
        scripted_upstream() as wire,
        reset_proxy(gateway, tmp_path) as owned,
        owned.scenario() as scenario,
        redis_client() as cache,
    ):
        model: Final = hold_model(scenario, wire)
        key: Final = scenario.key(models=[model], max_budget=1.0, budget_duration="1d")
        served(call(owned, model, key, max_tokens=5))
        spend_reaches(owned, "key", hashed(key), CHAT_COST)
        counter_settles(cache, key_counter(key), CHAT_COST)
        make_due(Due.KEY, hashed(key))
        wait_for_row_reset(owned, Due.KEY, hashed(key))
        assert spend_of(owned, "key", hashed(key)) == pytest.approx(0.0)
        assert eventually(lambda: cache.exists(key_counter(key)), lambda found: found == 0, seconds=RESET_SECONDS) == 0


@pytest.mark.timeout(300)
def test_a_due_key_reset_writes_the_next_utc_midnight_as_its_reset_time(gateway: Gateway, tmp_path: Path) -> None:
    """A `1d` key made due one second ago is given the next 00:00 UTC as its new `budget_reset_at`
    (`_reset_budget_common`, `reset_budget_job.py:2094-2099`, `compute_budget_reset_at`), not "now plus one day" and
    not the old reset time plus one day. A port that drifts the schedule by the job's lateness breaks every
    customer's billing day."""
    with reset_proxy(gateway, tmp_path) as owned, owned.scenario() as scenario:
        key: Final = scenario.key(max_budget=1.0, budget_duration="1d")
        make_due(Due.KEY, hashed(key))
        assert wait_for_row_reset(owned, Due.KEY, hashed(key)) == next_utc_midnight()


@pytest.mark.timeout(300)
def test_a_key_reset_also_deletes_the_charges_only_twin_counter(gateway: Gateway, tmp_path: Path) -> None:
    """On a proxy with the Prometheus callback (which turns on the charges-only twins,
    `committed_spend.py:57-58`), a served `1d` key has `spend:committed:key:<hash>` at 0.02. After its reset both
    `spend:key:<hash>` and the twin are gone (`spend_counter_store.delete((counter, *committed_spend_keys(...)))`,
    `reset_budget_job.py:876`), so the remaining budget gauge does not keep reporting the last period's spend."""
    with (
        scripted_upstream() as wire,
        reset_proxy(gateway, tmp_path, litellm_settings={"callbacks": ["prometheus"]}) as owned,
        owned.scenario() as scenario,
        redis_client() as cache,
    ):
        model: Final = hold_model(scenario, wire)
        key: Final = scenario.key(models=[model], max_budget=1.0, budget_duration="1d")
        served(call(owned, model, key, max_tokens=5))
        counter_settles(cache, committed(key_counter(key)), CHAT_COST)
        make_due(Due.KEY, hashed(key))
        wait_for_row_reset(owned, Due.KEY, hashed(key))
        assert (
            eventually(
                lambda: cache.exists(key_counter(key), committed(key_counter(key))),
                lambda found: found == 0,
                seconds=RESET_SECONDS,
            )
            == 0
        )


@pytest.mark.timeout(300)
def test_after_a_key_reset_the_next_hold_starts_from_zero(gateway: Gateway, tmp_path: Path) -> None:
    """After the reset of a served `1d` key, a held call with `max_tokens: 10` reads exactly 0.04 on the recreated
    `spend:key:<hash>` (the committed row's 0 plus the hold), not 0.06: the counter is reseeded from the reset row,
    never from the key object cached before the reset (`spend_counter_reseed.py:303-337`)."""
    with (
        held_provider() as provider,
        reset_proxy(gateway, tmp_path) as owned,
        owned.scenario() as scenario,
        redis_client() as cache,
    ):
        model: Final = hold_model(scenario, provider)
        key: Final = scenario.key(models=[model], max_budget=1.0, budget_duration="1d")
        served(call(owned, model, key, max_tokens=5))
        spend_reaches(owned, "key", hashed(key), CHAT_COST)
        make_due(Due.KEY, hashed(key))
        wait_for_row_reset(owned, Due.KEY, hashed(key))
        eventually(lambda: cache.exists(key_counter(key)), lambda found: found == 0, seconds=RESET_SECONDS)
        with background() as run:
            in_flight: Final = run(lambda: call(owned, model, key, hold=True, max_tokens=10))
            provider.wait_for(1)
            assert counter(cache, key_counter(key)) == pytest.approx(0.04)
            provider.release()
            served(in_flight.result(timeout=60))


@pytest.mark.timeout(300)
def test_a_reset_readmits_a_key_that_was_refused_for_its_budget(gateway: Gateway, tmp_path: Path) -> None:
    """A `1d` key with `max_budget` 0.02 is served once and then refused with HTTP 422 `budget_exceeded`. After its
    reset (stored spend 0, counter deleted), the same key is served again on the same proxy, with no cache
    eviction by the test: the job evicts the cached key object on every worker (`_key_cache_keys`,
    `reset_budget_job.py:272-273`)."""
    with (
        scripted_upstream() as wire,
        reset_proxy(gateway, tmp_path) as owned,
        owned.scenario() as scenario,
    ):
        model: Final = hold_model(scenario, wire)
        key: Final = scenario.key(models=[model], max_budget=CHAT_COST, budget_duration="1d")
        served(call(owned, model, key, max_tokens=5))
        spend_reaches(owned, "key", hashed(key), CHAT_COST)
        refused(call(owned, model, key, max_tokens=5))
        make_due(Due.KEY, hashed(key))
        wait_for_row_reset(owned, Due.KEY, hashed(key))
        response: Final = eventually(
            lambda: call(owned, model, key, max_tokens=5),
            lambda answer: answer.status_code == 200,
            seconds=RESET_SECONDS,
        )
        assert response.status_code == 200, response.text


@pytest.mark.timeout(300)
def test_a_due_user_reset_deletes_the_user_counter(gateway: Gateway, tmp_path: Path) -> None:
    """An internal user with `max_budget` 1.0 and `budget_duration` `1d` spends 0.02 through a personal key
    (`spend:user:<user_id>` 0.02). After the user's reset the stored user spend is 0 and the counter is absent
    (`reset_budget_job.py:1697`)."""
    with (
        scripted_upstream() as wire,
        reset_proxy(gateway, tmp_path) as owned,
        owned.scenario() as scenario,
        redis_client() as cache,
    ):
        model: Final = hold_model(scenario, wire)
        user_id: Final = scenario.user(user_role="internal_user", max_budget=1.0, budget_duration="1d")
        key: Final = scenario.key(user_id=user_id, models=[model])
        served(call(owned, model, key, max_tokens=5))
        counter_settles(cache, user_counter(user_id), CHAT_COST)
        spend_reaches(owned, "user", user_id, CHAT_COST)
        make_due(Due.USER, user_id)
        wait_for_row_reset(owned, Due.USER, user_id)
        assert spend_of(owned, "user", user_id) == pytest.approx(0.0)
        assert (
            eventually(lambda: cache.exists(user_counter(user_id)), lambda found: found == 0, seconds=RESET_SECONDS)
            == 0
        )


@pytest.mark.timeout(300)
def test_a_team_reset_leaves_the_counter_of_its_keys_own_budget_alone(gateway: Gateway, tmp_path: Path) -> None:
    """A team with `max_budget` 1.0 and `budget_duration` `1d` has a key with its own `max_budget` 1.0 and no
    duration. After one served call both counters read 0.02. When the team is reset, `spend:team:<team_id>` is
    deleted (`reset_budget_job.py:1818`) while `spend:key:<hash>` still reads 0.02 and the key's stored spend stays
    0.02: a team reset is not a reset of its keys."""
    with (
        scripted_upstream() as wire,
        reset_proxy(gateway, tmp_path) as owned,
        owned.scenario() as scenario,
        redis_client() as cache,
    ):
        model: Final = hold_model(scenario, wire)
        team_id: Final = scenario.team(models=[model], max_budget=1.0, budget_duration="1d")
        key: Final = scenario.key(team_id=team_id, models=[model], max_budget=1.0)
        served(call(owned, model, key, max_tokens=5))
        counter_settles(cache, team_counter(team_id), CHAT_COST)
        counter_settles(cache, key_counter(key), CHAT_COST)
        spend_reaches(owned, "key", hashed(key), CHAT_COST)
        make_due(Due.TEAM, team_id)
        wait_for_row_reset(owned, Due.TEAM, team_id)
        eventually(lambda: cache.exists(team_counter(team_id)), lambda found: found == 0, seconds=RESET_SECONDS)
        assert counter(cache, key_counter(key)) == pytest.approx(CHAT_COST)
        assert spend_of(owned, "key", hashed(key)) == pytest.approx(CHAT_COST)


@pytest.mark.timeout(300)
def test_a_due_team_member_budget_row_resets_the_membership_and_its_counter(gateway: Gateway, tmp_path: Path) -> None:
    """A team with `team_member_budget` 1.0 and `team_member_budget_duration` `1d` has a member whose key is served
    once (`spend:team_member:<user>:<team>` 0.02, `LiteLLM_TeamMembership.spend` 0.02). When the membership's budget
    row is made due, the cascade zeroes the membership spend and deletes the member counter
    (`_team_membership_counter_key`, `reset_budget_job.py:260-261`, `_commit_budget_cascade_once`, `:1281-1300`).
    UNVERIFIED: that `member_add` links the membership to the team's default member budget row
    (`team_endpoints.py:774-777` stores its id in the team metadata); the test reads the link from the
    membership row."""
    with (
        scripted_upstream() as wire,
        reset_proxy(gateway, tmp_path) as owned,
        owned.scenario() as scenario,
        redis_client() as cache,
    ):
        model: Final = hold_model(scenario, wire)
        team_id: Final = scenario.team(models=[model], team_member_budget=1.0, team_member_budget_duration="1d")
        user_id: Final = scenario.member(team_id)
        key: Final = scenario.key(team_id=team_id, user_id=user_id, models=[model])
        served(call(owned, model, key, max_tokens=5))
        counter_settles(cache, team_member_counter(user_id, team_id), CHAT_COST)
        membership: Final = 'SELECT spend FROM "LiteLLM_TeamMembership" WHERE user_id = %s AND team_id = %s'
        eventually(
            lambda: _project_spend(owned.rows(membership, (user_id, team_id))),
            lambda spend: spend is not None and spend >= CHAT_COST - 1e-9,
            seconds=RESET_SECONDS,
        )
        budget_id: Final = budget_id_of(owned, "membership", user_id, team_id)
        make_budget_row_due(budget_id)
        wait_for_row_reset(owned, Due.BUDGET, budget_id)
        assert eventually(
            lambda: _project_spend(owned.rows(membership, (user_id, team_id))),
            lambda spend: spend == pytest.approx(0.0),
            seconds=RESET_SECONDS,
        ) == pytest.approx(0.0)
        assert (
            eventually(
                lambda: cache.exists(team_member_counter(user_id, team_id)),
                lambda found: found == 0,
                seconds=RESET_SECONDS,
            )
            == 0
        )


@pytest.mark.timeout(300)
def test_a_due_organization_budget_row_resets_the_organization_and_its_counter(
    gateway: Gateway, tmp_path: Path
) -> None:
    """An organization created with `max_budget` 1.0 and `budget_duration` `1d` (its own `LiteLLM_BudgetTable` row)
    spends 0.02 through a team key. When that budget row is due, the organization's stored spend becomes 0 and
    `spend:org:<organization_id>` is deleted (`_org_counter_key`, `reset_budget_job.py:276-277`), and the budget
    row's `budget_reset_at` moves to the next 00:00 UTC (`budget_resets`, `:1101-1108`)."""
    with (
        scripted_upstream() as wire,
        reset_proxy(gateway, tmp_path) as owned,
        owned.scenario() as scenario,
        redis_client() as cache,
    ):
        model: Final = hold_model(scenario, wire)
        org_id: Final = scenario.organization(models=[model], max_budget=1.0, budget_duration="1d")
        team_id: Final = scenario.team(organization_id=org_id, models=[model])
        key: Final = scenario.key(team_id=team_id, models=[model])
        served(call(owned, model, key, max_tokens=5))
        counter_settles(cache, org_counter(org_id), CHAT_COST)
        spend_reaches(owned, "org", org_id, CHAT_COST)
        budget_id: Final = budget_id_of(owned, "org", org_id)
        make_budget_row_due(budget_id)
        assert wait_for_row_reset(owned, Due.BUDGET, budget_id) == next_utc_midnight()
        assert eventually(
            lambda: spend_of(owned, "org", org_id), lambda spend: spend == pytest.approx(0.0), seconds=RESET_SECONDS
        ) == pytest.approx(0.0)
        assert (
            eventually(lambda: cache.exists(org_counter(org_id)), lambda found: found == 0, seconds=RESET_SECONDS) == 0
        )


@pytest.mark.timeout(300)
def test_an_organization_whose_budget_row_is_not_due_keeps_its_spend(gateway: Gateway, tmp_path: Path) -> None:
    """Two organizations with `1d` budget rows each spend 0.02. Only the first row is made due. After the first
    organization was reset, the second still has stored spend 0.02 and `spend:org:<id>` 0.02: the cascade only
    touches rows linked to the due budget ids (`_budget_link_where`, `reset_budget_job.py:319-326`)."""
    with (
        scripted_upstream() as wire,
        reset_proxy(gateway, tmp_path) as owned,
        owned.scenario() as scenario,
        redis_client() as cache,
    ):
        model: Final = hold_model(scenario, wire)
        orgs: Final = [scenario.organization(models=[model], max_budget=1.0, budget_duration="1d") for _ in range(2)]
        for org_id in orgs:
            team_id = scenario.team(organization_id=org_id, models=[model])
            served(call(owned, model, scenario.key(team_id=team_id, models=[model]), max_tokens=5))
            spend_reaches(owned, "org", org_id, CHAT_COST)
        make_budget_row_due(budget_id_of(owned, "org", orgs[0]))
        eventually(lambda: spend_of(owned, "org", orgs[0]), lambda spend: spend == 0, seconds=RESET_SECONDS)
        assert spend_of(owned, "org", orgs[1]) == pytest.approx(CHAT_COST)
        assert counter(cache, org_counter(orgs[1])) == pytest.approx(CHAT_COST)


@pytest.mark.timeout(300)
def test_a_due_tag_budget_row_deletes_the_tag_counter(gateway: Gateway, tmp_path: Path) -> None:
    """A tag created with `max_budget` 1.0 and `budget_duration` `1d` is charged 0.02 by one tagged call
    (`spend:tag:<name>` 0.02). When the tag's budget row is due, the tag's stored spend becomes 0 and its counter is
    deleted (`_tag_counter_key`, `reset_budget_job.py:287-288`). Tag counters are never reseeded from the table
    (`spend_counter_reseed.py:125-127`), so the next tagged call starts the counter from the reloaded tag object."""
    with (
        scripted_upstream() as wire,
        reset_proxy(gateway, tmp_path) as owned,
        owned.scenario() as scenario,
        redis_client() as cache,
    ):
        model: Final = hold_model(scenario, wire)
        tag: Final = new_tag(owned, scenario, max_budget=1.0, budget_duration="1d")
        key: Final = scenario.key(models=[model])
        served(call(owned, model, key, max_tokens=5, metadata={"tags": [tag]}))
        counter_settles(cache, tag_counter(tag), CHAT_COST)
        spend_reaches(owned, "tag", tag, CHAT_COST)
        make_budget_row_due(budget_id_of(owned, "tag", tag))
        eventually(lambda: spend_of(owned, "tag", tag), lambda spend: spend == 0, seconds=RESET_SECONDS)
        assert eventually(lambda: cache.exists(tag_counter(tag)), lambda found: found == 0, seconds=RESET_SECONDS) == 0


@pytest.mark.timeout(300)
def test_a_due_project_budget_row_resets_the_project_and_its_counter(gateway: Gateway, tmp_path: Path) -> None:
    """A project created with `max_budget` 1.0 and `budget_duration` `1d` is charged 0.02 by its key
    (`spend:project:<id>` 0.02). When its budget row is due, `LiteLLM_ProjectTable.spend` becomes 0 and the counter
    is deleted (`_project_counter_key`, `reset_budget_job.py:303-304`). UNVERIFIED: that `/project/new` stores
    `budget_duration` on the project's budget row (`NewProjectRequest` extends `LiteLLM_BudgetTable`)."""
    with (
        scripted_upstream() as wire,
        reset_proxy(gateway, tmp_path) as owned,
        owned.scenario() as scenario,
        redis_client() as cache,
    ):
        model: Final = hold_model(scenario, wire)
        team_id: Final = scenario.team(models=[model])
        project_id: Final = scenario.project(team_id, models=[model], max_budget=1.0, budget_duration="1d")
        key: Final = scenario.key(team_id=team_id, project_id=project_id, models=[model])
        served(call(owned, model, key, max_tokens=5))
        counter_settles(cache, project_counter(project_id), CHAT_COST)
        spend_sql: Final = 'SELECT spend FROM "LiteLLM_ProjectTable" WHERE project_id = %s'
        eventually(
            lambda: _project_spend(owned.rows(spend_sql, (project_id,))),
            lambda spend: spend is not None and spend >= CHAT_COST - 1e-9,
            seconds=RESET_SECONDS,
        )
        make_budget_row_due(budget_id_of(owned, "project", project_id))
        eventually(
            lambda: _project_spend(owned.rows(spend_sql, (project_id,))),
            lambda spend: spend == pytest.approx(0.0),
            seconds=RESET_SECONDS,
        )
        assert (
            eventually(
                lambda: cache.exists(project_counter(project_id)), lambda found: found == 0, seconds=RESET_SECONDS
            )
            == 0
        )


@pytest.mark.timeout(300)
def test_a_due_customer_budget_row_resets_the_end_user_and_both_its_counters(gateway: Gateway, tmp_path: Path) -> None:
    """A customer created with `max_budget` 1.0 and `budget_duration` `1d` spends 0.02 as the body `user` on a proxy
    with the Prometheus callback. When the customer's budget row is due, `LiteLLM_EndUserTable.spend` becomes 0 and
    both `spend:end_user:<id>` and `spend:committed:end_user:<id>` are deleted (`_reset_enduser_page`,
    `reset_budget_job.py:1157-1174`, end user counters have a twin, `committed_spend.py:38`)."""
    with (
        scripted_upstream() as wire,
        reset_proxy(gateway, tmp_path, litellm_settings={"callbacks": ["prometheus"]}) as owned,
        owned.scenario() as scenario,
        redis_client() as cache,
    ):
        model: Final = hold_model(scenario, wire)
        customer: Final = new_customer(owned, scenario, max_budget=1.0, budget_duration="1d")
        key: Final = scenario.key(models=[model])
        served(call(owned, model, key, max_tokens=5, user=customer))
        counter_settles(cache, end_user_counter(customer), CHAT_COST)
        spend_reaches(owned, "end_user", customer, CHAT_COST)
        make_budget_row_due(budget_id_of(owned, "end_user", customer))
        eventually(lambda: spend_of(owned, "end_user", customer), lambda spend: spend == 0, seconds=RESET_SECONDS)
        assert (
            eventually(
                lambda: cache.exists(end_user_counter(customer), committed(end_user_counter(customer))),
                lambda found: found == 0,
                seconds=RESET_SECONDS,
            )
            == 0
        )


@pytest.mark.timeout(300)
def test_a_key_linked_to_a_budget_row_resets_with_that_row(gateway: Gateway, tmp_path: Path) -> None:
    """A budget row with `max_budget` 1.0 and `budget_duration` `1d` is created with `/budget/new`, and a key with
    `budget_id` of that row, its own `max_budget` 1.0 and no `budget_duration` spends 0.02. When the row is due the
    key's stored spend becomes 0 and `spend:key:<hash>` is deleted: keys without their own duration ride their
    budget row's schedule (`_LINKED_KEYS_WHERE`, `reset_budget_job.py:188`, `:1057-1061`)."""
    with (
        scripted_upstream() as wire,
        reset_proxy(gateway, tmp_path) as owned,
        owned.scenario() as scenario,
        redis_client() as cache,
    ):
        model: Final = hold_model(scenario, wire)
        budget_id: Final = scenario.budget(max_budget=1.0, budget_duration="1d")
        key: Final = scenario.key(models=[model], budget_id=budget_id, max_budget=1.0)
        served(call(owned, model, key, max_tokens=5))
        spend_reaches(owned, "key", hashed(key), CHAT_COST)
        make_budget_row_due(budget_id)
        eventually(lambda: spend_of(owned, "key", hashed(key)), lambda spend: spend == 0, seconds=RESET_SECONDS)
        assert eventually(lambda: cache.exists(key_counter(key)), lambda found: found == 0, seconds=RESET_SECONDS) == 0


@pytest.mark.timeout(300)
def test_a_key_with_its_own_duration_is_not_reset_by_its_budget_row(gateway: Gateway, tmp_path: Path) -> None:
    """The same setup, but the key also has its own `budget_duration` `1d` (its own `budget_reset_at` tomorrow).
    Making only the budget row due resets the row (its `budget_reset_at` moves to the next midnight) and leaves the
    key's stored spend and `spend:key:<hash>` at 0.02: such a key runs only on its own schedule, so it is never
    reset twice (`_collect_budget_cascade` docstring, `reset_budget_job.py:1039-1044`)."""
    with (
        scripted_upstream() as wire,
        reset_proxy(gateway, tmp_path) as owned,
        owned.scenario() as scenario,
        redis_client() as cache,
    ):
        model: Final = hold_model(scenario, wire)
        budget_id: Final = scenario.budget(max_budget=1.0, budget_duration="1d")
        key: Final = scenario.key(models=[model], budget_id=budget_id, max_budget=1.0, budget_duration="1d")
        served(call(owned, model, key, max_tokens=5))
        spend_reaches(owned, "key", hashed(key), CHAT_COST)
        counter_settles(cache, key_counter(key), CHAT_COST)
        make_budget_row_due(budget_id)
        wait_for_row_reset(owned, Due.BUDGET, budget_id)
        assert spend_of(owned, "key", hashed(key)) == pytest.approx(CHAT_COST)
        assert counter(cache, key_counter(key)) == pytest.approx(CHAT_COST)


@pytest.mark.timeout(300)
def test_with_budget_rollover_a_due_organization_keeps_only_its_overage(gateway: Gateway, tmp_path: Path) -> None:
    """With `litellm_settings.budget_rollover: true`, an organization with `max_budget` 0.01 and `budget_duration`
    `1d` is served one call that costs 0.02 (the hold shrinks to the 0.01 of room, the settle charges 0.02). After its
    budget row resets, the organization's stored spend is 0.01 (0.02 minus the cap), not 0
    (`_row_carried_spend`, `reset_budget_job.py:254-257`, `_rollover_cap`, `:242-245`), so the next call is still
    refused with the organization wording. Rounds before this one pinned rollover for keys, users and teams only."""
    with (
        scripted_upstream() as wire,
        reset_proxy(gateway, tmp_path, litellm_settings={"budget_rollover": True}) as owned,
        owned.scenario() as scenario,
    ):
        model: Final = hold_model(scenario, wire)
        org_id: Final = scenario.organization(models=[model], max_budget=0.01, budget_duration="1d")
        team_id: Final = scenario.team(organization_id=org_id, models=[model])
        key: Final = scenario.key(team_id=team_id, models=[model])
        served(call(owned, model, key, max_tokens=5))
        spend_reaches(owned, "org", org_id, CHAT_COST)
        make_budget_row_due(budget_id_of(owned, "org", org_id))
        carried: Final = eventually(
            lambda: spend_of(owned, "org", org_id),
            lambda spend: spend is not None and spend < CHAT_COST - 1e-9,
            seconds=RESET_SECONDS,
        )
        assert carried == pytest.approx(0.01)
        assert f"Organization={org_id}" in refused(call(owned, model, key, max_tokens=5))
