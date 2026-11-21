from collections.abc import Iterator
from types import MappingProxyType
from typing import Final

import pytest
from integration._support.client import Gateway, eventually, object_value, string_value
from integration._support.database import write_rows
from integration._support.process import OwnedState, owned_proxy
from pydantic import JsonValue, TypeAdapter

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
            info: Final = object_value(proxy.get("/user/info", {"user_id": user_id})["user_info"])
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
