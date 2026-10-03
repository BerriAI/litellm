import os
from concurrent.futures import ThreadPoolExecutor
from typing import Final

import psycopg
import pytest
from psycopg import sql
from pydantic import JsonValue

from tests.integration._support.client import Gateway, Scenario, eventually, string_value
from tests.integration._support.database import read_rows, write_rows


def _member(scenario: Scenario, team_id: str, amount: int, duration: str | None = "30d") -> str:
    user_id: Final = scenario.user()
    scenario.gateway.post("/team/member_add", {"team_id": team_id, "member": {"user_id": user_id, "role": "user"}})
    scenario.gateway.post(
        "/team/member_update",
        {"team_id": team_id, "user_id": user_id, "max_budget_in_team": amount, "budget_duration": duration},
    )
    return user_id


def _budgets(team_id: str) -> list[dict[str, JsonValue]]:
    return read_rows(
        "SELECT m.user_id, m.spend, b.max_budget, b.budget_duration, b.rpm_limit, "
        'b.temp_budget_increase, b.budget_reset_at::text FROM "LiteLLM_TeamMembership" m '
        'LEFT JOIN "LiteLLM_BudgetTable" b ON b.budget_id=m.budget_id WHERE m.team_id=%s ORDER BY m.user_id',
        (team_id,),
    )


@pytest.mark.parametrize(
    ("mode", "cleared"), (("keep", ()), ("raise", (0, 50)), ("lower", (150,)), ("both", (0, 50, 150)))
)
def test_default_and_selected_amounts_update_together_without_changing_other_limits(
    gateway: Gateway, mode: str, cleared: tuple[int, ...]
) -> None:
    with gateway.scenario() as scenario:
        team: Final = scenario.team(team_member_budget=75, team_member_budget_duration="30d")
        users: Final = {amount: _member(scenario, team, amount) for amount in (0, 50, 100, 150)}
        different_period: Final = _member(scenario, team, 50, "7d")
        never_resets: Final = _member(scenario, team, 150, None)
        gateway.post("/team/member_update", {"team_id": team, "user_id": users[50], "rpm_limit": 9})
        write_rows('UPDATE "LiteLLM_TeamMembership" SET spend=23 WHERE team_id=%s AND user_id=%s', (team, users[50]))
        write_rows(
            'UPDATE "LiteLLM_BudgetTable" SET temp_budget_increase=200, '
            "temp_budget_expiry=NOW()+INTERVAL '1 day' WHERE budget_id IN "
            '(SELECT budget_id FROM "LiteLLM_TeamMembership" WHERE team_id=%s AND user_id=%s)',
            (team, users[50]),
        )
        before: Final = _budgets(team)
        response: Final = gateway.post(
            "/team/update", {"team_id": team, "team_member_budget": 100, "team_member_budget_update_mode": mode}
        )
        assert response["member_budgets_updated"] == len(cleared), response
        changed_users: Final = frozenset(users[amount] for amount in cleared)
        assert _budgets(team) == [
            {**row, "max_budget": None}
            if row["user_id"] in changed_users
            else {**row, "max_budget": 100}
            if row["user_id"] not in {*users.values(), different_period, never_resets}
            else row
            for row in before
        ]
        gateway.post("/team/update", {"team_id": team, "team_member_budget": 125})
        result: Final = gateway.post(
            f"/management/v1/teams/{team}/members/bulk_update",
            {"members": [{"user_id": user_id, "rpm_limit": 9} for user_id in users.values()]},
        )
        rows: Final = result["data"]
        assert isinstance(rows, list), result
        assert {row["user_id"]: row["max_budget"] for row in rows} == {
            user_id: 125 if amount in cleared else amount for amount, user_id in users.items()
        }


def test_resetting_an_amount_only_override_leaves_the_member_inheriting_the_default(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        team: Final = scenario.team(team_member_budget=75)
        user: Final = _member(scenario, team, 50, None)
        response: Final = gateway.post(
            "/team/update", {"team_id": team, "team_member_budget": 100, "team_member_budget_update_mode": "raise"}
        )
        assert response["member_budgets_updated"] == 1
        assert read_rows(
            'SELECT budget_id FROM "LiteLLM_TeamMembership" WHERE team_id=%s AND user_id=%s', (team, user)
        ) == [{"budget_id": None}]


def test_a_failed_member_write_rolls_back_the_new_default(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        team: Final = scenario.team(team_member_budget=75, team_member_budget_duration="30d")
        user: Final = _member(scenario, team, 50)
        before: Final = _budgets(team)
        budget_id: Final = string_value(
            read_rows('SELECT budget_id FROM "LiteLLM_TeamMembership" WHERE team_id=%s AND user_id=%s', (team, user))[
                0
            ]["budget_id"]
        )
        constraint: Final = sql.Identifier(f"test_member_budget_{user.replace('-', '_')}")
        with psycopg.connect(os.environ["DATABASE_URL"], autocommit=True) as connection:
            connection.execute(
                sql.SQL(
                    'ALTER TABLE "LiteLLM_BudgetTable" ADD CONSTRAINT {} CHECK (budget_id <> {} OR max_budget IS NOT NULL)'
                ).format(constraint, sql.Literal(budget_id))
            )
            try:
                response: Final = gateway.request(
                    "POST",
                    "/team/update",
                    {"team_id": team, "team_member_budget": 100, "team_member_budget_update_mode": "raise"},
                )
                assert response.status_code >= 400, response.text
                assert _budgets(team) == before
            finally:
                connection.execute(sql.SQL('ALTER TABLE "LiteLLM_BudgetTable" DROP CONSTRAINT {}').format(constraint))


@pytest.mark.parametrize(("amount", "period"), ((150, "30d"), (50, "7d")))
def test_raise_rechecks_amount_and_period_after_a_concurrent_edit(gateway: Gateway, amount: int, period: str) -> None:
    with gateway.scenario() as scenario:
        team: Final = scenario.team(team_member_budget=75, team_member_budget_duration="30d")
        user: Final = _member(scenario, team, 50)
        with psycopg.connect(os.environ["DATABASE_URL"]) as blocker, ThreadPoolExecutor(max_workers=1) as executor:
            blocker.execute(
                'UPDATE "LiteLLM_BudgetTable" SET max_budget=%s, budget_duration=%s WHERE budget_id IN '
                '(SELECT budget_id FROM "LiteLLM_TeamMembership" WHERE team_id=%s AND user_id=%s)',
                (amount, period, team, user),
            )
            future: Final = executor.submit(
                gateway.request,
                "POST",
                "/team/update",
                {"team_id": team, "team_member_budget": 100, "team_member_budget_update_mode": "raise"},
            )
            try:
                eventually(
                    lambda: read_rows(
                        "SELECT pid FROM pg_stat_activity WHERE %s::int=ANY(pg_blocking_pids(pid))",
                        (str(blocker.info.backend_pid),),
                    ),
                    bool,
                    seconds=10,
                )
            finally:
                blocker.commit()
            response: Final = future.result(timeout=30)
        assert response.status_code == 200, response.text
        assert response.json()["member_budgets_updated"] == 0, response.text
        member: Final = next(row for row in _budgets(team) if row["user_id"] == user)
        assert (member["max_budget"], member["budget_duration"]) == (amount, period)


@pytest.mark.parametrize("surface", ("single", "bulk"))
def test_concurrent_member_edits_follow_the_membership_after_a_shared_budget_is_cloned(
    gateway: Gateway, surface: str
) -> None:
    with gateway.scenario() as scenario:
        team: Final = scenario.team(team_member_budget=75, team_member_budget_duration="30d")
        user: Final = _member(scenario, team, 50)
        other_team: Final = scenario.team(team_member_budget=75, team_member_budget_duration="30d")
        other_user: Final = _member(scenario, other_team, 50)
        write_rows(
            'UPDATE "LiteLLM_TeamMembership" SET budget_id=(SELECT budget_id FROM "LiteLLM_TeamMembership" '
            "WHERE team_id=%s AND user_id=%s) WHERE team_id=%s AND user_id=%s",
            (team, user, other_team, other_user),
        )
        other_before: Final = _budgets(other_team)
        endpoint: Final = (
            "/team/member_update" if surface == "single" else f"/management/v1/teams/{team}/members/bulk_update"
        )
        payload: Final = (
            {"team_id": team, "user_id": user, "rpm_limit": 9}
            if surface == "single"
            else {"members": [{"user_id": user, "rpm_limit": 9}]}
        )
        with psycopg.connect(os.environ["DATABASE_URL"]) as blocker, ThreadPoolExecutor(max_workers=2) as executor:
            blocker.execute(
                'SELECT budget_id FROM "LiteLLM_BudgetTable" WHERE budget_id=(SELECT budget_id '
                'FROM "LiteLLM_TeamMembership" WHERE team_id=%s AND user_id=%s) FOR UPDATE',
                (team, user),
            )
            backfill: Final = executor.submit(
                gateway.request,
                "POST",
                "/team/update",
                {"team_id": team, "team_member_budget": 100, "team_member_budget_update_mode": "raise"},
            )
            try:
                eventually(
                    lambda: read_rows(
                        "SELECT pid FROM pg_stat_activity WHERE %s::int=ANY(pg_blocking_pids(pid))",
                        (str(blocker.info.backend_pid),),
                    ),
                    bool,
                    seconds=10,
                )
                edit: Final = executor.submit(gateway.request, "POST", endpoint, payload)
                eventually(
                    lambda: read_rows(
                        "SELECT pid FROM pg_stat_activity WHERE datname=current_database() "
                        "AND cardinality(pg_blocking_pids(pid))>0",
                        (),
                    ),
                    lambda rows: len(rows) >= 2,
                    seconds=10,
                )
            finally:
                blocker.commit()
            backfill_response: Final = backfill.result(timeout=30)
            edit_response: Final = edit.result(timeout=30)
        assert backfill_response.status_code == 200, backfill_response.text
        assert backfill_response.json()["member_budgets_updated"] == 1, backfill_response.text
        assert edit_response.status_code == 200, edit_response.text
        member: Final = next(row for row in _budgets(team) if row["user_id"] == user)
        assert (member["max_budget"], member["rpm_limit"]) == (None, 9)
        assert _budgets(other_team) == other_before


def test_first_default_attaches_unlinked_members_and_resets_matching_overrides(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        team: Final = scenario.team()
        user: Final = _member(scenario, team, 50)
        inherited: Final = scenario.user()
        gateway.post("/team/member_add", {"team_id": team, "member": {"user_id": inherited, "role": "user"}})
        response: Final = gateway.post(
            "/team/update",
            {
                "team_id": team,
                "team_member_budget": 100,
                "team_member_budget_duration": "30d",
                "team_member_budget_update_mode": "raise",
            },
        )
        assert response["member_budgets_updated"] == 1, response
        rows: Final = {row["user_id"]: row for row in _budgets(team)}
        assert rows[user]["max_budget"] is None
        assert (rows[inherited]["max_budget"], rows[inherited]["budget_duration"]) == (100, "30d")


@pytest.mark.parametrize("amount", (None, 0))
def test_bulk_action_requires_a_positive_default(gateway: Gateway, amount: int | None) -> None:
    with gateway.scenario() as scenario:
        team: Final = scenario.team(team_member_budget=75, team_member_budget_duration="30d")
        _member(scenario, team, 50)
        before: Final = _budgets(team)
        response: Final = gateway.request(
            "POST",
            "/team/update",
            {"team_id": team, "team_member_budget": amount, "team_member_budget_update_mode": "both"},
        )
        assert response.status_code == 400, response.text
        assert _budgets(team) == before


@pytest.mark.parametrize(
    "concurrent_update",
    (
        pytest.param({"team_member_budget": 200}, id="positive-budget"),
        pytest.param({"team_member_budget": 0}, id="zero-budget"),
        pytest.param({"team_member_rpm_limit": 9}, id="rate-only"),
        pytest.param({"team_member_budget": None}, id="clear-budget"),
        pytest.param({"metadata": {"concurrent_setting": True}}, id="metadata-only"),
    ),
)
def test_concurrent_team_updates_preserve_member_default_inheritance(
    gateway: Gateway, concurrent_update: dict[str, JsonValue]
) -> None:
    with gateway.scenario() as scenario:
        team: Final = scenario.team()
        custom: Final = _member(scenario, team, 50)
        inherited: Final = scenario.member(team)
        with psycopg.connect(os.environ["DATABASE_URL"]) as blocker, ThreadPoolExecutor(max_workers=2) as executor:
            blocker.execute(
                'SELECT budget_id FROM "LiteLLM_BudgetTable" WHERE budget_id=(SELECT budget_id '
                'FROM "LiteLLM_TeamMembership" WHERE team_id=%s AND user_id=%s) FOR UPDATE',
                (team, custom),
            )
            atomic: Final = executor.submit(
                gateway.request,
                "POST",
                "/team/update",
                {
                    "team_id": team,
                    "team_member_budget": 100,
                    "team_member_budget_duration": "30d",
                    "team_member_budget_update_mode": "raise",
                },
            )
            try:
                blocked_atomic: Final = eventually(
                    lambda: read_rows(
                        "SELECT pid::text FROM pg_stat_activity WHERE %s::int=ANY(pg_blocking_pids(pid))",
                        (str(blocker.info.backend_pid),),
                    ),
                    bool,
                    seconds=10,
                )
                atomic_pid: Final = string_value(blocked_atomic[0]["pid"])
                concurrent: Final = executor.submit(
                    gateway.request, "POST", "/team/update", {"team_id": team, **concurrent_update}
                )
                eventually(
                    lambda: read_rows(
                        "SELECT pid FROM pg_stat_activity WHERE %s::int=ANY(pg_blocking_pids(pid))",
                        (atomic_pid,),
                    ),
                    bool,
                    seconds=10,
                )
            finally:
                blocker.commit()
            atomic_response: Final = atomic.result(timeout=30)
            concurrent_response: Final = concurrent.result(timeout=30)
        assert atomic_response.status_code == 200, atomic_response.text
        assert atomic_response.json()["member_budgets_updated"] == 1, atomic_response.text
        assert concurrent_response.status_code == 200, concurrent_response.text
        assert read_rows(
            'SELECT b.budget_id IS NOT NULL AS has_default FROM "LiteLLM_TeamTable" t '
            "LEFT JOIN \"LiteLLM_BudgetTable\" b ON b.budget_id=t.metadata->>'team_member_budget_id' "
            "WHERE t.team_id=%s",
            (team,),
        ) == [{"has_default": True}]
        gateway.post("/team/update", {"team_id": team, "team_member_budget": 300})
        assert read_rows(
            "SELECT m.budget_id=t.metadata->>'team_member_budget_id' AS inherits_default "
            'FROM "LiteLLM_TeamMembership" m JOIN "LiteLLM_TeamTable" t USING(team_id) '
            "WHERE m.team_id=%s AND m.user_id=%s",
            (team, inherited),
        ) == [{"inherits_default": True}]
        amounts: Final = {row["user_id"]: row["max_budget"] for row in _budgets(team)}
        assert (amounts[custom], amounts[inherited]) == (None, 300)
        resolved: Final = gateway.post(
            f"/management/v1/teams/{team}/members/bulk_update", {"members": [{"user_id": custom}]}
        )
        results: Final = resolved["data"]
        assert isinstance(results, list), resolved
        assert len(results) == 1, resolved
        member: Final = results[0]
        assert isinstance(member, dict), resolved
        assert (member["user_id"], member["max_budget"], member["max_budget_source"]) == (
            custom,
            300,
            "team_default",
        )


@pytest.mark.parametrize(
    ("update", "expected_key_duration", "expected_budget"),
    (
        pytest.param({"team_member_budget": 100}, "1h", 100, id="budget-only"),
        pytest.param({"team_member_key_duration": ""}, "", 75, id="metadata-backed-field"),
    ),
)
def test_omitted_metadata_preserves_a_concurrent_metadata_change(
    gateway: Gateway,
    update: dict[str, JsonValue],
    expected_key_duration: str,
    expected_budget: int,
) -> None:
    with gateway.scenario() as scenario:
        team: Final = scenario.team(
            team_member_budget=75,
            metadata={"label": "before", "team_member_key_duration": "1h"},
        )
        with psycopg.connect(os.environ["DATABASE_URL"]) as blocker, ThreadPoolExecutor(max_workers=2) as executor:
            blocker.execute('SELECT team_id FROM "LiteLLM_TeamTable" WHERE team_id=%s FOR UPDATE', (team,))
            metadata_writer: Final = executor.submit(
                gateway.request,
                "POST",
                "/team/update",
                {"team_id": team, "metadata": {"label": "after", "team_member_key_duration": "1h"}},
            )
            try:
                blocked_metadata: Final = eventually(
                    lambda: read_rows(
                        "SELECT pid::text FROM pg_stat_activity WHERE %s::int=ANY(pg_blocking_pids(pid))",
                        (str(blocker.info.backend_pid),),
                    ),
                    bool,
                    seconds=10,
                )
                metadata_pid: Final = string_value(blocked_metadata[0]["pid"])
                other_writer: Final = executor.submit(
                    gateway.request, "POST", "/team/update", {"team_id": team, **update}
                )
                eventually(
                    lambda: read_rows(
                        "SELECT pid::text FROM pg_stat_activity WHERE %s::int=ANY(pg_blocking_pids(pid))",
                        (metadata_pid,),
                    ),
                    bool,
                    seconds=10,
                )
            finally:
                blocker.commit()
            metadata_response: Final = metadata_writer.result(timeout=30)
            other_response: Final = other_writer.result(timeout=30)
        assert metadata_response.status_code == 200, metadata_response.text
        assert other_response.status_code == 200, other_response.text
        assert read_rows(
            "SELECT t.metadata->>'label' AS label, "
            "t.metadata->>'team_member_key_duration' AS key_duration, b.max_budget AS default_amount "
            'FROM "LiteLLM_TeamTable" t LEFT JOIN "LiteLLM_BudgetTable" b '
            "ON b.budget_id=t.metadata->>'team_member_budget_id' WHERE t.team_id=%s",
            (team,),
        ) == [{"label": "after", "key_duration": expected_key_duration, "default_amount": expected_budget}]
