import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from litellm.proxy._types import TeamMemberBudgetBackfillResult, UpdateTeamRequest, UserAPIKeyAuth
from litellm.proxy.management_helpers.team_member_default_budget import (
    MAX_MEMBER_BUDGETS_MOVED_PER_UPDATE,
    MovedMemberBudgets,
    TooManyMemberBudgetsToMove,
    move_member_budgets_to_default,
    plan_team_member_default_budget_write,
    write_team_member_default_budget,
)


def _plan(existing_budget_id: str | None = None, team_budget_duration: str | None = None, **fields: object):
    return plan_team_member_default_budget_write(
        data=UpdateTeamRequest(team_id="team-1", **fields),
        existing_budget_id=existing_budget_id,
        team_alias="My Team",
        team_budget_duration=team_budget_duration,
    )


def test_a_request_without_member_defaults_writes_nothing():
    assert _plan(existing_budget_id="default-1", max_budget=10.0) is None


def test_clearing_a_default_the_team_never_had_writes_nothing():
    assert _plan(team_member_budget=None) is None


def test_clearing_an_existing_default_updates_it_without_relinking_members():
    write = _plan(existing_budget_id="default-1", team_member_budget=None, team_member_model_max_budget=None)

    assert write is not None
    assert write.budget_id == "default-1"
    assert dict(write.update) == {"max_budget": None, "model_max_budget": {}}
    assert write.links_members is False


def test_a_first_default_gets_a_new_row_that_inherits_the_team_reset_period():
    before = datetime.now(timezone.utc)

    write = _plan(team_budget_duration="30d", team_member_budget=25.0)

    assert write is not None
    assert write.budget_id.startswith("team-My-Team-budget-")
    assert write.create["max_budget"] == 25.0
    assert write.create["budget_duration"] == "30d"
    reset_at = write.create["budget_reset_at"]
    assert isinstance(reset_at, datetime) and reset_at > before
    assert write.links_members is True


def test_an_explicit_null_reset_period_is_not_replaced_by_the_team_period():
    write = _plan(team_budget_duration="30d", team_member_budget=25.0, team_member_budget_duration=None)

    assert write is not None
    assert "budget_duration" not in write.create
    assert write.update["budget_duration"] is None
    assert write.update["budget_reset_at"] is None


def test_changing_the_reset_period_restarts_the_window():
    before = datetime.now(timezone.utc)

    write = _plan(existing_budget_id="default-1", team_member_budget_duration="7d")

    assert write is not None
    assert write.update["budget_duration"] == "7d"
    reset_at = write.update["budget_reset_at"]
    assert isinstance(reset_at, datetime) and reset_at > before


@pytest.mark.parametrize("team_member_budget", [None, 0.0])
def test_backfill_without_a_positive_default_in_the_same_request_is_rejected(team_member_budget):
    with pytest.raises(ValidationError, match="team_member_budget_backfill needs a positive team_member_budget"):
        UpdateTeamRequest(
            team_id="team-1", team_member_budget=team_member_budget, team_member_budget_backfill="raise_to_default"
        )


@pytest.mark.parametrize("backfill", [None, "none"])
def test_keeping_custom_budgets_needs_no_default(backfill):
    assert UpdateTeamRequest(team_id="team-1", team_member_budget_backfill=backfill).team_member_budget is None


def test_per_model_member_budgets_are_stored_as_plain_json():
    caps = {"gpt-4o": {"max_budget": 5.0, "budget_duration": "1d"}}

    write = _plan(existing_budget_id="default-1", team_member_model_max_budget=caps)

    assert write is not None
    assert json.loads(json.dumps(write.update["model_max_budget"]))["gpt-4o"]["max_budget"] == 5.0


class _InMemoryTeam:
    """One team's memberships and budget rows, changed the way the backfill's statements change Postgres."""

    def __init__(self, default: dict[str, object], members: dict[str, dict[str, object] | None]) -> None:
        self.budgets: dict[str, dict[str, object]] = {"default": {"budget_id": "default", **default}}
        self.memberships: dict[str, str | None] = {}
        for user_id, budget in members.items():
            if budget is None:
                self.memberships[user_id] = "default"
                continue
            self.budgets[f"own-{user_id}"] = {"budget_id": f"own-{user_id}", **budget}
            self.memberships[user_id] = f"own-{user_id}"
        self.litellm_budgettable = SimpleNamespace(
            find_unique=self._find_budget, upsert=self._upsert_budget, create=self._create_budget
        )
        self.litellm_teammembership = SimpleNamespace(upsert=self._link_membership)

    def share(self, budget_id: str, *user_ids: str) -> None:
        for user_id in user_ids:
            self.memberships[user_id] = budget_id

    async def _create_budget(self, data: dict[str, object], include: dict[str, bool] | None = None):
        budget_id = f"copy-{len(self.budgets)}"
        self.budgets[budget_id] = {"budget_id": budget_id, **data}
        return SimpleNamespace(budget_id=budget_id)

    async def _link_membership(self, where: dict[str, dict[str, str]], data: dict[str, dict[str, object]]) -> None:
        self.memberships[where["user_id_team_id"]["user_id"]] = data["update"]["litellm_budget_table"]["connect"][
            "budget_id"
        ]

    def budget_of(self, user_id: str) -> dict[str, object]:
        return self.budgets[self.memberships[user_id]]

    async def _find_budget(self, where: dict[str, str]):
        stored = self.budgets.get(where["budget_id"])
        if stored is None:
            return None
        row = {**_ROW_DEFAULTS, **stored}
        return SimpleNamespace(**row, model_dump=lambda: row)

    async def _upsert_budget(self, where: dict[str, str], data: dict[str, dict[str, object]]) -> None:
        current = self.budgets.get(where["budget_id"])
        self.budgets[where["budget_id"]] = {**(current or {}), **(data["update"] if current else data["create"])}

    def _on_chosen_side(self, budget: dict[str, object], direction: str, amount: float) -> bool:
        own = budget.get("max_budget")
        if budget["budget_id"] == "default" or own is None:
            return False
        return {"raise_to_default": own < amount, "lower_to_default": own > amount, "all_to_default": own != amount}[
            direction
        ]

    async def query_raw(self, sql: str, *args: object) -> list[dict[str, object]]:
        if "HAVING COUNT(*) > 1" in sql:
            budget_ids = list(self.memberships.values())
            return [{"budget_id": b} for b in set(args[0]) if budget_ids.count(b) > 1]
        _team, _default, direction, amount = args[:4]
        chosen = [
            (u, self.budgets[b])
            for u, b in self.memberships.items()
            if b and self._on_chosen_side(self.budgets[b], direction, amount)
        ]
        if "GROUP BY" in sql:
            durations = [budget.get("budget_duration") for _, budget in chosen]
            return [{"duration": d, "members": durations.count(d)} for d in set(durations)]
        durations, include_none = args[4], args[5]
        return [
            {**_ROW_DEFAULTS, **budget, "user_id": user_id}
            for user_id, budget in chosen
            if budget.get("budget_duration") in durations or (include_none and budget.get("budget_duration") is None)
        ]

    async def execute_raw(self, sql: str, *args: object) -> int:
        if "INSERT INTO" in sql:
            for user_id in args[2]:
                self.memberships.setdefault(user_id, args[1])
        elif "budget_id IS NULL" in sql:
            self.memberships = {u: b or args[1] for u, b in self.memberships.items()}
        elif "SET budget_reset_at" in sql:
            for user_id in args[2]:
                if self.memberships[user_id] != args[1]:
                    self.budget_of(user_id)["budget_reset_at"] = self.budgets[args[1]].get("budget_reset_at")
        elif "SET max_budget = NULL" in sql:
            for budget_id in args[0]:
                self.budgets[budget_id]["max_budget"] = None
        elif "SET budget_id = $2" in sql:
            for user_id in args[2]:
                self.memberships[user_id] = args[1]
        return 0


_ROW_DEFAULTS = {
    "soft_budget": None,
    "max_parallel_requests": None,
    "tpm_limit": None,
    "rpm_limit": None,
    "model_max_budget": None,
    "budget_duration": None,
    "allowed_models": None,
    "temp_budget_increase": None,
    "temp_budget_expiry": None,
}
_DEFAULT_RESET = datetime(2030, 1, 1, tzinfo=timezone.utc)
_ADMIN = UserAPIKeyAuth(user_id="admin")


async def _backfill(team: _InMemoryTeam, backfill: str):
    return await move_member_budgets_to_default(
        team, team_id="team-1", default_budget_id="default", backfill=backfill, changed_by=_ADMIN
    )


def _team(**members: dict[str, object] | None) -> _InMemoryTeam:
    return _InMemoryTeam({"max_budget": 50.0, "budget_duration": "30d", "budget_reset_at": _DEFAULT_RESET}, members)


@pytest.mark.asyncio
async def test_raise_moves_members_below_the_default_on_its_schedule_and_counts_the_rest():
    team = _team(
        below={"max_budget": 10.0, "budget_duration": "30d"},
        own_rpm={"max_budget": 10.0, "budget_duration": "30d", "rpm_limit": 5, "budget_reset_at": "2030-02-01"},
        weekly={"max_budget": 10.0, "budget_duration": "7d"},
        above={"max_budget": 80.0, "budget_duration": "30d"},
        on_default=None,
    )

    moved = await _backfill(team, "raise_to_default")

    assert isinstance(moved, MovedMemberBudgets)
    assert sorted(moved.user_ids) == ["below", "own_rpm"]
    assert moved.result() == TeamMemberBudgetBackfillResult(moved_to_default=2, skipped_other_reset_period=1)
    assert team.memberships["below"] == "default"
    assert team.budget_of("own_rpm") == {
        "budget_id": "own-own_rpm",
        "max_budget": None,
        "budget_duration": "30d",
        "rpm_limit": 5,
        "budget_reset_at": _DEFAULT_RESET,
    }
    assert team.budget_of("weekly")["max_budget"] == 10.0
    assert team.budget_of("above")["max_budget"] == 80.0


@pytest.mark.asyncio
async def test_a_model_list_matching_the_default_still_keeps_the_members_own_row():
    team = _team(scoped={"max_budget": 10.0, "budget_duration": "30d", "allowed_models": ["gpt-5.4-mini"]})
    team.budgets["default"]["allowed_models"] = ["gpt-5.4-mini"]

    await _backfill(team, "raise_to_default")

    assert team.memberships["scoped"] == "own-scoped"
    assert team.budget_of("scoped")["max_budget"] is None


@pytest.mark.asyncio
async def test_more_members_than_one_update_may_move_changes_nobody():
    members = {
        f"m{i}": {"max_budget": 10.0, "budget_duration": "30d"} for i in range(MAX_MEMBER_BUDGETS_MOVED_PER_UPDATE + 1)
    }
    team = _team(**members)

    moved = await _backfill(team, "raise_to_default")

    assert moved == TooManyMemberBudgetsToMove(count=MAX_MEMBER_BUDGETS_MOVED_PER_UPDATE + 1)
    assert all(budget_id != "default" for budget_id in team.memberships.values())


@pytest.mark.asyncio
async def test_members_sharing_a_budget_with_their_own_limits_each_get_a_copy_on_the_default():
    team = _team()
    team.budgets["shared"] = {"budget_id": "shared", "max_budget": 10.0, "budget_duration": "30d", "rpm_limit": 5}
    team.share("shared", "first", "second")

    moved = await _backfill(team, "raise_to_default")

    assert isinstance(moved, MovedMemberBudgets)
    assert sorted(moved.user_ids) == ["first", "second"]
    copies = {team.memberships["first"], team.memberships["second"]}
    assert len(copies) == 2 and "shared" not in copies
    assert [(team.budgets[c]["max_budget"], team.budgets[c]["rpm_limit"]) for c in sorted(copies)] == [
        (None, 5),
        (None, 5),
    ]
    assert team.budgets["shared"]["max_budget"] == 10.0


@pytest.mark.asyncio
async def test_only_members_on_another_schedule_are_counted_and_left_alone():
    team = _team(weekly={"max_budget": 10.0, "budget_duration": "7d"})

    moved = await _backfill(team, "raise_to_default")

    assert moved == MovedMemberBudgets(user_ids=(), skipped_other_reset_period=1)
    assert team.memberships["weekly"] == "own-weekly"


@pytest.mark.asyncio
async def test_a_missing_default_row_moves_nobody():
    team = _team(below={"max_budget": 10.0, "budget_duration": "30d"})
    del team.budgets["default"]

    assert await _backfill(team, "raise_to_default") == MovedMemberBudgets(user_ids=(), skipped_other_reset_period=0)


def test_a_first_default_on_a_team_without_an_alias_gets_a_generic_id():
    write = plan_team_member_default_budget_write(
        data=UpdateTeamRequest(team_id="team-1", team_member_budget=5.0),
        existing_budget_id=None,
        team_alias=None,
        team_budget_duration=None,
    )

    assert write is not None
    assert write.budget_id.startswith("team-budget-")


@pytest.mark.asyncio
async def test_keeping_custom_budgets_moves_nobody():
    team = _team(below={"max_budget": 10.0, "budget_duration": "30d"})

    assert await _backfill(team, "none") == MovedMemberBudgets(user_ids=(), skipped_other_reset_period=0)
    assert team.memberships["below"] == "own-below"


@pytest.mark.asyncio
async def test_writing_a_default_links_roster_members_without_a_budget_to_it():
    team = _team(own={"max_budget": 10.0, "budget_duration": "30d"})
    team.memberships["unbudgeted"] = None
    write = _plan(
        existing_budget_id="default",
        team_member_budget=60.0,
        team_member_model_max_budget={"gpt-4o": {"max_budget": 1.0}},
    )
    assert write is not None

    await write_team_member_default_budget(
        team, write, team_id="team-1", member_user_ids=("own", "unbudgeted", "no-row"), changed_by="admin"
    )

    assert team.budgets["default"]["max_budget"] == 60.0
    assert team.budgets["default"]["model_max_budget"].data == {
        "gpt-4o": {"max_budget": 1.0, "budget_duration": None, "tpm_limit": None, "rpm_limit": None}
    }
    assert (team.memberships["unbudgeted"], team.memberships["no-row"], team.memberships["own"]) == (
        "default",
        "default",
        "own-own",
    )


@pytest.mark.asyncio
async def test_clearing_the_default_does_not_link_members_without_a_budget_to_it():
    team = _team()
    team.memberships["unbudgeted"] = None
    write = _plan(existing_budget_id="default", team_member_budget=None)
    assert write is not None

    await write_team_member_default_budget(
        team, write, team_id="team-1", member_user_ids=("unbudgeted", "no-row"), changed_by="admin"
    )

    assert team.budgets["default"]["max_budget"] is None
    assert team.memberships == {"unbudgeted": None}
