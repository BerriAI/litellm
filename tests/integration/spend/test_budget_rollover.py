from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from types import MappingProxyType
from typing import Final

import psycopg
import pytest
from integration._support.client import Gateway, eventually, object_value, string_value
from integration._support.database import write_rows
from integration._support.process import OwnedState, owned_proxy
from pydantic import JsonValue, TypeAdapter

from litellm.models.budget import LiteLLM_BudgetTableFull
from litellm.proxy.common_utils.reset_budget_job import ResetBudgetJob
from litellm.repositories.budget_repository import BudgetRepository
from tests.integration._support.prisma_clients import connected_proxy_prisma

_FAST_RESET_TICK: Final = MappingProxyType(
    {"PROXY_BUDGET_RESCHEDULER_MIN_TIME": "2", "PROXY_BUDGET_RESCHEDULER_MAX_TIME": "2"}
)
_MEMBERS: Final = TypeAdapter(list[dict[str, JsonValue]])


@pytest.fixture(scope="module")
def rollover_proxy(session_gateway: Gateway, tmp_path_factory: pytest.TempPathFactory) -> Iterator[Gateway]:
    with owned_proxy(
        session_gateway,
        tmp_path_factory.mktemp("budget-rollover"),
        _FAST_RESET_TICK,
        workers=2,
        state=OwnedState.PRIVATE,
    ) as proxy:
        yield proxy


@pytest.fixture(scope="module")
def overage_rollover_proxy(session_gateway: Gateway, tmp_path_factory: pytest.TempPathFactory) -> Iterator[Gateway]:
    directory: Final = tmp_path_factory.mktemp("temporary-budget-rollover")
    config: Final = directory / "proxy.yaml"
    config.write_text(
        "model_list: []\n"
        "general_settings:\n"
        "  master_key: os.environ/LITELLM_MASTER_KEY\n"
        "  database_url: os.environ/DATABASE_URL\n"
        "litellm_settings:\n"
        "  budget_rollover: true\n"
    )
    with owned_proxy(
        session_gateway, directory, _FAST_RESET_TICK, config=config, workers=2, state=OwnedState.PRIVATE
    ) as proxy:
        yield proxy


def _personal_close_period(proxy: Gateway, user_id: str, spend: str) -> None:
    write_rows(
        "UPDATE \"LiteLLM_UserTable\" SET spend = %s, budget_reset_at = now() - interval '1 minute' WHERE user_id = %s",
        (spend, user_id),
        database_url=proxy.database_url,
    )
    eventually(
        lambda: proxy.rows(
            'SELECT (budget_reset_at > now()) AS advanced FROM "LiteLLM_UserTable" WHERE user_id = %s',
            (user_id,),
        ),
        lambda rows: rows == [{"advanced": True}],
        seconds=30,
    )


def _member_close_period(proxy: Gateway, budget_id: str) -> None:
    write_rows(
        "UPDATE \"LiteLLM_BudgetTable\" SET budget_reset_at = now() - interval '1 minute' WHERE budget_id = %s",
        (budget_id,),
        database_url=proxy.database_url,
    )
    eventually(
        lambda: proxy.rows(
            'SELECT (budget_reset_at > now()) AS advanced FROM "LiteLLM_BudgetTable" WHERE budget_id = %s',
            (budget_id,),
        ),
        lambda rows: rows == [{"advanced": True}],
        seconds=30,
    )


def _member_budget(gateway: Gateway, team_id: str, user_id: str) -> dict[str, JsonValue]:
    rows: Final = _MEMBERS.validate_python(gateway.get("/team/info", {"team_id": team_id})["team_memberships"])
    return next(row for row in rows if row["user_id"] == user_id)


@pytest.mark.parametrize(("expiry_offset", "carried_spend"), ((1, 0.0), (-1, 30.0)), ids=("active", "expired"))
def test_user_budget_reset_only_counts_active_temporary_increase(
    overage_rollover_proxy: Gateway, expiry_offset: int, carried_spend: float
) -> None:
    proxy: Final = overage_rollover_proxy
    with proxy.scenario() as scenario:
        user_id: Final = scenario.user(max_budget=100, budget_duration="30d")
        expiry: Final = datetime.now(timezone.utc) + timedelta(hours=expiry_offset)
        proxy.post(
            "/user/update",
            {"user_id": user_id, "temp_budget_increase": 50, "temp_budget_expiry": expiry.isoformat()},
        )

        _personal_close_period(proxy, user_id, "130")

        assert proxy.rows('SELECT spend, max_budget FROM "LiteLLM_UserTable" WHERE user_id = %s', (user_id,)) == [
            {"spend": carried_spend, "max_budget": 100.0}
        ]


def test_user_edits_only_return_rollover_fields_when_enabled(rollover_proxy: Gateway) -> None:
    proxy: Final = rollover_proxy
    with proxy.scenario() as scenario:
        user_id: Final = scenario.user(max_budget=10, budget_duration="30d")
        for patch in ({"user_alias": "Legacy user"}, {"max_budget": 20}, {"budget_duration": "7d"}):
            updated = object_value(proxy.post("/user/update", {"user_id": user_id, **patch})["data"])
            assert {key: updated[key] for key in patch} == patch
            assert {"rollover_max_budget", "rollover_credit"}.isdisjoint(updated), updated

        enabled: Final = object_value(
            proxy.post("/user/update", {"user_id": user_id, "rollover_max_budget": 30})["data"]
        )
        assert enabled["rollover_max_budget"] == 30
        disabled: Final = object_value(
            proxy.post("/user/update", {"user_id": user_id, "rollover_max_budget": None})["data"]
        )
        assert {"rollover_max_budget", "rollover_credit"}.isdisjoint(disabled), disabled


@pytest.mark.parametrize("personal", (False, True), ids=("shared-budget", "personal-budget"))
@pytest.mark.parametrize("enable_rollover", (False, True), ids=("uncapped", "concurrently-capped"))
def test_budget_update_respects_a_competing_write(
    rollover_proxy: Gateway, personal: bool, enable_rollover: bool
) -> None:
    proxy: Final = rollover_proxy
    assert proxy.database_url is not None
    with proxy.scenario() as scenario, ThreadPoolExecutor(max_workers=1) as executor:
        budget_id: Final = (
            scenario.user(max_budget=10, budget_duration="30d")
            if personal
            else string_value(proxy.post("/budget/new", {"max_budget": 10, "budget_duration": "30d"})["budget_id"])
        )
        if not personal:
            scenario.cleanups.callback(scenario.delete_budget, budget_id)
        endpoint: Final = "/user/update" if personal else "/budget/update"
        id_key: Final = "user_id" if personal else "budget_id"
        table: Final = "LiteLLM_UserTable" if personal else "LiteLLM_BudgetTable"

        with psycopg.connect(proxy.database_url) as blocker:
            writer_fields: Final = "max_budget = 11, rollover_max_budget = 20" if enable_rollover else "max_budget = 11"
            blocker.execute(f'UPDATE "{table}" SET {writer_fields} WHERE {id_key} = %s', (budget_id,))
            update: Final = executor.submit(proxy.request, "POST", endpoint, {id_key: budget_id, "max_budget": 25})
            blocker_pid: Final = str(blocker.info.backend_pid)
            eventually(
                lambda: proxy.rows(
                    "SELECT EXISTS (SELECT 1 FROM pg_stat_activity WHERE %s::int = ANY(pg_blocking_pids(pid)))"
                    " AS blocked",
                    (blocker_pid,),
                ),
                lambda rows: rows == [{"blocked": True}],
            )
            if not enable_rollover:
                eventually(update.done, bool, seconds=6, return_last_on_timeout=True)
        result: Final = update.result(timeout=15)
        assert result.status_code == (400 if enable_rollover else 200), result.text
        if enable_rollover:
            assert "rollover" in result.text, result.text
        assert proxy.rows(f'SELECT max_budget FROM "{table}" WHERE {id_key} = %s', (budget_id,)) == [
            {"max_budget": 11.0 if enable_rollover else 25.0}
        ]


async def test_a_capped_member_reset_does_not_overwrite_a_newer_period_edit(
    rollover_proxy: Gateway, monkeypatch: pytest.MonkeyPatch
) -> None:
    proxy: Final = rollover_proxy
    assert proxy.database_url is not None
    monkeypatch.setenv("DATABASE_URL", proxy.database_url)
    monkeypatch.delenv("DATABASE_URL_READ_REPLICA", raising=False)
    window_sql: Final = (
        "SELECT budget_duration, budget_reset_at::text AS reset_at, budget_reset_count AS closes "
        'FROM "LiteLLM_BudgetTable" WHERE budget_id = %s'
    )
    with proxy.scenario() as scenario:
        team: Final = scenario.team(
            team_member_budget=100.0, team_member_budget_duration="30d", team_member_rollover_max_budget=250.0
        )
        member: Final = scenario.member(team)
        budget_id: Final = string_value(_member_budget(proxy, team, member)["budget_id"])
        write_rows(
            'UPDATE "LiteLLM_TeamMembership" SET spend = 30 WHERE team_id = %s AND user_id = %s',
            (team, member),
        )
        write_rows(
            """UPDATE "LiteLLM_BudgetTable" SET budget_reset_at = now() + interval '6 hours' WHERE budget_id = %s""",
            (budget_id,),
        )
        async with connected_proxy_prisma() as client:
            job: Final = ResetBudgetJob(client.proxy_logging_obj, client)
            rows: Final = await BudgetRepository(client).table.find_many(where={"budget_id": budget_id})
            budgets: Final = tuple(LiteLLM_BudgetTableFull.model_validate(row, from_attributes=True) for row in rows)
            cascade: Final = await job._collect_budget_cascade(budgets, datetime.now(timezone.utc) + timedelta(days=1))
            assert cascade.budget_ids == (budget_id,)
            proxy.post(
                "/team/update", {"team_id": team, "team_member_budget": 100.0, "team_member_budget_duration": "7d"}
            )
            edited: Final = proxy.rows(window_sql, (budget_id,))
            assert len(edited) == 1 and edited[0]["budget_duration"] == "7d", edited
            await job._commit_budget_cascade(cascade)
        assert proxy.rows(window_sql, (budget_id,)) == edited
        assert proxy.rows(
            'SELECT spend, rollover_credit FROM "LiteLLM_TeamMembership" WHERE team_id = %s AND user_id = %s',
            (team, member),
        ) == [{"spend": 30.0, "rollover_credit": None}]

        _member_close_period(proxy, budget_id)
        later: Final = _member_budget(proxy, team, member)
        assert (later["spend"], later["rollover_credit"]) == (0.0, 70.0)


@pytest.mark.parametrize(("cap", "credits"), ((250.0, (70.0, 150.0, 150.0)), (None, (0.0, 0.0, 0.0))))
def test_personal_monthly_budget_carries_unused_allowance_only_when_enabled(
    rollover_proxy: Gateway, cap: float | None, credits: tuple[float, ...]
) -> None:
    proxy: Final = rollover_proxy
    with proxy.scenario() as scenario:
        user_id: Final = scenario.user(
            user_role="internal_user",
            max_budget=100.0,
            budget_duration="monthly",
            **({"rollover_max_budget": cap} if cap is not None else {}),
        )
        for spend, expected in zip(("30", "0", "0"), credits, strict=True):
            _personal_close_period(proxy, user_id, spend)
            info = object_value(proxy.get("/user/info", {"user_id": user_id})["user_info"])
            assert info["spend"] == 0.0
            assert info["max_budget"] == 100.0
            if cap is not None:
                assert info.get("rollover_credit") == expected, info
                assert info["effective_max_budget"] == 100.0 + expected
            else:
                assert not info.get("rollover_credit")

        model: Final = scenario.model(input_cost_per_token=0.001, output_cost_per_token=0.001)
        key: Final = scenario.key(user_id=user_id, models=[model])
        write_rows(
            'UPDATE "LiteLLM_UserTable" SET spend = 120 WHERE user_id = %s',
            (user_id,),
            database_url=proxy.database_url,
        )
        response: Final = proxy.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": "rollover"}]},
            key=key,
        )
        if cap is None:
            assert response.status_code == 422 and f"User={user_id}" in response.text, response.text
        else:
            assert response.status_code == 200, response.text
            v2: Final = proxy.get("/v2/user/info", {"user_id": user_id})
            assert (v2["rollover_credit"], v2["effective_max_budget"]) == (150.0, 250.0)
            proxy.post("/user/update", {"user_id": user_id, "rollover_max_budget": None})
            after_clear: Final = proxy.get("/v2/user/info", {"user_id": user_id})
            assert (after_clear.get("rollover_credit"), after_clear.get("effective_max_budget")) == (None, None)
            refused: Final = proxy.request(
                "POST",
                "/v1/chat/completions",
                {"model": model, "messages": [{"role": "user", "content": "cleared"}]},
                key=key,
            )
            assert refused.status_code == 422 and f"User={user_id}" in refused.text, refused.text


@pytest.mark.parametrize("cap", (250.0, None))
def test_members_get_a_full_midcycle_allowance_and_roll_over_their_own_spend(
    rollover_proxy: Gateway, cap: float | None
) -> None:
    proxy: Final = rollover_proxy
    with proxy.scenario() as scenario:
        team: Final = scenario.team(
            max_budget=155.0,
            team_member_budget=100.0,
            team_member_budget_duration="monthly",
            **({"team_member_rollover_max_budget": cap} if cap is not None else {}),
        )
        first: Final = scenario.member(team)
        first_budget: Final = _member_budget(proxy, team, first)
        budget_id: Final = string_value(first_budget["budget_id"])
        write_rows(
            'UPDATE "LiteLLM_TeamMembership" SET spend = 30 WHERE team_id = %s AND user_id = %s',
            (team, first),
            database_url=proxy.database_url,
        )
        _member_close_period(proxy, budget_id)
        second: Final = scenario.member(team)
        second_budget: Final = _member_budget(proxy, team, second)
        assert second_budget["spend"] == 0.0
        if cap is not None:
            assert second_budget.get("effective_max_budget") == 100.0, second_budget
            assert _member_budget(proxy, team, first).get("rollover_credit") == 70.0
        else:
            assert not _member_budget(proxy, team, first).get("rollover_credit")
        _member_close_period(proxy, budget_id)
        first_after: Final = _member_budget(proxy, team, first)
        second_after: Final = _member_budget(proxy, team, second)
        assert first_after["spend"] == second_after["spend"] == 0.0
        if cap is not None:
            assert first_after["rollover_credit"] == 150.0
            assert second_after["rollover_credit"] == 100.0
            assert proxy.rows(
                'SELECT rollover_credit FROM "LiteLLM_TeamMembership" WHERE team_id = %s AND user_id = %s',
                (team, first),
            ) == [{"rollover_credit": 150.0}]
        else:
            assert not first_after.get("rollover_credit")
            assert not second_after.get("rollover_credit")

        model: Final = scenario.model(input_cost_per_token=0.001, output_cost_per_token=0.001)
        key: Final = scenario.key(team_id=team, user_id=second, models=[model])
        if cap is not None:
            members: Final = proxy.get(f"/management/v1/teams/{team}/members")
            roster: Final = _MEMBERS.validate_python(members["data"])
            second_roster: Final = next(row for row in roster if row["user_id"] == second)
            assert (second_roster["rollover_credit"], second_roster["effective_max_budget"]) == (100.0, 200.0)
            write_rows(
                'UPDATE "LiteLLM_TeamMembership" SET spend = 120 WHERE team_id = %s AND user_id = %s',
                (team, second),
                database_url=proxy.database_url,
            )
            admitted: Final = proxy.request(
                "POST",
                "/v1/chat/completions",
                {"model": model, "messages": [{"role": "user", "content": "rollover"}]},
                key=key,
            )
            assert admitted.status_code == 200, admitted.text
            await_spend: Final = eventually(
                lambda: proxy.rows('SELECT spend FROM "LiteLLM_TeamTable" WHERE team_id = %s', (team,)),
                lambda rows: bool(rows) and rows[0]["spend"] != 0.0,
                seconds=60,
            )
            assert len(await_spend) == 1
            proxy.post("/team/update", {"team_id": team, "max_budget": 0.001})
        else:
            write_rows(
                'UPDATE "LiteLLM_TeamTable" SET spend = 156 WHERE team_id = %s',
                (team,),
                database_url=proxy.database_url,
            )
        blocked: Final = proxy.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": "control"}]},
            key=key,
        )
        assert blocked.status_code == 422 and f"Team={team}" in blocked.text, blocked.text
        if cap is not None:
            proxy.post("/team/update", {"team_id": team, "max_budget": 155.0, "team_member_rollover_max_budget": None})
            cleared: Final = _member_budget(proxy, team, second)
            assert (cleared.get("rollover_credit"), cleared.get("effective_max_budget")) == (None, None)
            refused: Final = proxy.request(
                "POST",
                "/v1/chat/completions",
                {"model": model, "messages": [{"role": "user", "content": "cleared"}]},
                key=key,
            )
            assert refused.status_code == 422 and "TeamMember=" in refused.text, refused.text
            proxy.post("/team/update", {"team_id": team, "team_member_rollover_max_budget": cap})
            reenabled: Final = _member_budget(proxy, team, second)
            assert (reenabled["rollover_credit"], reenabled["effective_max_budget"]) == (0.0, 100.0)


def test_member_keeps_rollover_when_a_rate_limit_edit_creates_a_private_budget(rollover_proxy: Gateway) -> None:
    proxy: Final = rollover_proxy
    with proxy.scenario() as scenario:
        team: Final = scenario.team(
            team_member_budget=100.0,
            team_member_budget_duration="monthly",
            team_member_rollover_max_budget=250.0,
        )
        member: Final = scenario.member(team)
        shared_id: Final = string_value(_member_budget(proxy, team, member)["budget_id"])
        _member_close_period(proxy, shared_id)
        proxy.post("/team/member_update", {"team_id": team, "user_id": member, "rpm_limit": 20})
        private: Final = _member_budget(proxy, team, member)
        private_id: Final = string_value(private["budget_id"])
        assert private_id != shared_id, private
        assert (private.get("rollover_credit"), private.get("effective_max_budget")) == (100.0, 200.0), private
        _member_close_period(proxy, private_id)
        renewed: Final = _member_budget(proxy, team, member)
        assert (renewed["rollover_credit"], renewed["effective_max_budget"]) == (150.0, 250.0), renewed


@pytest.mark.parametrize(
    ("path", "payload"),
    (
        ("/user/new", {"max_budget": 100.0, "rollover_max_budget": 250.0}),
        ("/user/new", {"max_budget": 100.0, "budget_duration": "monthly", "rollover_max_budget": 100.0}),
        ("/team/new", {"team_member_budget": 100.0, "team_member_rollover_max_budget": 250.0}),
        (
            "/team/new",
            {
                "team_member_budget": 100.0,
                "team_member_budget_duration": "monthly",
                "team_member_rollover_max_budget": 100.0,
            },
        ),
    ),
)
def test_rollover_requires_a_reset_period_and_cap_above_the_base(
    rollover_proxy: Gateway, path: str, payload: dict[str, JsonValue]
) -> None:
    response: Final = rollover_proxy.request("POST", path, payload)
    assert response.status_code == 400 and "rollover requires a reset period" in response.text, response.text


@pytest.mark.parametrize("bulk", (False, True), ids=("single-user", "all-users"))
def test_user_budget_edits_cannot_disable_saved_rollover_implicitly(rollover_proxy: Gateway, bulk: bool) -> None:
    proxy: Final = rollover_proxy
    with proxy.scenario() as scenario:
        user_id: Final = scenario.user(max_budget=100, budget_duration="monthly", rollover_max_budget=250)
        rejected: Final = (
            proxy.request("POST", "/user/bulk_update", {"all_users": True, "user_updates": {"budget_duration": None}})
            if bulk
            else proxy.request("POST", "/user/update", {"user_id": user_id, "max_budget": 300})
        )
        assert rejected.status_code == 400 and "rollover" in rejected.text, rejected.text

        intact: Final = proxy.get("/v2/user/info", {"user_id": user_id})
        assert (intact["rollover_max_budget"], intact["effective_max_budget"]) == (250, 100), intact
        proxy.post("/user/update", {"user_id": user_id, "max_budget": 300, "rollover_max_budget": None})
        updated: Final = proxy.get("/v2/user/info", {"user_id": user_id})
        assert updated["max_budget"] == 300 and "rollover_credit" not in updated, updated


@pytest.mark.parametrize("scope", ("team-default", "shared-membership", "private-membership"))
def test_team_budget_edits_cannot_disable_saved_rollover_implicitly(rollover_proxy: Gateway, scope: str) -> None:
    proxy: Final = rollover_proxy
    with proxy.scenario() as scenario:
        team: Final = scenario.team(
            team_member_budget=100, team_member_budget_duration="monthly", team_member_rollover_max_budget=250
        )
        member: Final = scenario.member(team)
        if scope == "private-membership":
            proxy.post("/team/member_update", {"team_id": team, "user_id": member, "rpm_limit": 20})
        rejected: Final = (
            proxy.request("POST", "/team/update", {"team_id": team, "team_member_budget": 300})
            if scope == "team-default"
            else proxy.request(
                "POST", "/team/member_update", {"team_id": team, "user_id": member, "budget_duration": None}
            )
        )
        assert rejected.status_code == 400 and "rollover" in rejected.text, rejected.text

        intact: Final = _member_budget(proxy, team, member)
        assert intact["effective_max_budget"] == 100, intact
        if scope == "team-default":
            proxy.post(
                "/team/update", {"team_id": team, "team_member_budget": 300, "team_member_rollover_max_budget": None}
            )
            updated: Final = _member_budget(proxy, team, member)
            assert "rollover_credit" not in updated, updated


def test_rejected_team_period_edit_keeps_the_inheriting_members_reset_schedule(rollover_proxy: Gateway) -> None:
    proxy: Final = rollover_proxy
    with proxy.scenario() as scenario:
        team: Final = scenario.team(team_member_budget=100, team_member_budget_duration="monthly")
        member: Final = scenario.member(team)
        proxy.post("/team/member_update", {"team_id": team, "user_id": member, "rpm_limit": 20})
        proxy.post(
            f"/management/v1/teams/{team}/members/bulk_update",
            {"members": [{"user_id": member, "max_budget_in_team": None}]},
        )
        proxy.post("/team/update", {"team_id": team, "team_member_rollover_max_budget": 250})
        budget_id: Final = string_value(_member_budget(proxy, team, member)["budget_id"])
        schedule_query: Final = (
            'SELECT max_budget, budget_duration, budget_reset_at::text FROM "LiteLLM_BudgetTable" WHERE budget_id = %s'
        )
        (before,) = proxy.rows(schedule_query, (budget_id,))
        assert before["max_budget"] is None and before["budget_duration"] == "monthly" and before["budget_reset_at"]

        response: Final = proxy.request("POST", "/team/update", {"team_id": team, "team_member_budget_duration": None})
        assert response.status_code == 400 and "rollover" in response.text, response.text
        assert proxy.rows(schedule_query, (budget_id,)) == [before]

        proxy.post("/team/update", {"team_id": team, "team_member_budget_duration": "7d"})
        (after,) = proxy.rows(schedule_query, (budget_id,))
        assert (after["max_budget"], after["budget_duration"]) == (None, "7d")
        assert after["budget_reset_at"]


def test_management_responses_omit_rollover_cap_without_opt_in(rollover_proxy: Gateway) -> None:
    proxy: Final = rollover_proxy
    with proxy.scenario() as scenario:
        user: Final = proxy.post(
            "/user/new", {"max_budget": 100.0, "budget_duration": "monthly", "auto_create_key": False}
        )
        scenario.cleanups.callback(scenario.delete_user, string_value(user["user_id"]))
        assert "rollover_max_budget" not in user

        budget: Final = proxy.post("/budget/new", {"max_budget": 100.0, "budget_duration": "monthly"})
        budget_id: Final = string_value(budget["budget_id"])
        scenario.cleanups.callback(scenario.delete_budget, budget_id)
        assert "rollover_max_budget" not in budget

        response: Final = proxy.request("POST", "/budget/info", {"budgets": [budget_id]})
        assert response.status_code == 200, response.text
        rows: Final = _MEMBERS.validate_json(response.content)
        assert len(rows) == 1 and "rollover_max_budget" not in rows[0], rows


def test_team_can_save_legacy_member_rate_limits_without_enabling_rollover(rollover_proxy: Gateway) -> None:
    proxy: Final = rollover_proxy
    with proxy.scenario() as scenario:
        team: Final = scenario.team(team_member_rpm_limit=-1, team_member_tpm_limit=-1)
        initial: Final = proxy.get("/team/info", {"team_id": team})
        budget: Final = object_value(object_value(initial["team_info"])["team_member_budget_table"])
        assert (budget["rpm_limit"], budget["tpm_limit"]) == (-1, -1), budget

        alias: Final = f"renamed-{team}"
        proxy.post(
            "/team/update",
            {
                "team_id": team,
                "team_alias": alias,
                "team_member_rpm_limit": budget["rpm_limit"],
                "team_member_tpm_limit": budget["tpm_limit"],
            },
        )
        updated: Final = object_value(proxy.get("/team/info", {"team_id": team})["team_info"])
        assert updated["team_alias"] == alias, updated
        saved_budget: Final = object_value(updated["team_member_budget_table"])
        assert (saved_budget["rpm_limit"], saved_budget["tpm_limit"]) == (-1, -1), saved_budget


def test_budget_info_preserves_the_utc_timestamp_format(rollover_proxy: Gateway) -> None:
    proxy: Final = rollover_proxy
    with proxy.scenario() as scenario:
        reset_at: Final = "2040-01-01T00:00:00Z"
        budget: Final = proxy.post("/budget/new", {"budget_reset_at": reset_at})
        budget_id: Final = string_value(budget["budget_id"])
        scenario.cleanups.callback(scenario.delete_budget, budget_id)
        response: Final = proxy.request("POST", "/budget/info", {"budgets": [budget_id]})
        assert response.status_code == 200, response.text
        budgets: Final = _MEMBERS.validate_json(response.content)
        assert [row["budget_reset_at"] for row in budgets] == [reset_at], budgets
