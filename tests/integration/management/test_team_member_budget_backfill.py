from collections.abc import Mapping
from dataclasses import dataclass

import pytest
from prisma import Json

from litellm.proxy.management_helpers.team_member_default_budget import MAX_MEMBER_BUDGETS_MOVED_PER_UPDATE
from tests.integration._support.actors import create_scratch_team

pytestmark = pytest.mark.asyncio(loop_scope="session")

_OLD_DEFAULT = 40.0
_NEW_DEFAULT = 50.0
_PERIOD = "30d"


@dataclass(frozen=True)
class _Budget:
    max_budget: float
    budget_duration: str | None = _PERIOD
    rpm_limit: int | None = None


_ROSTER: Mapping[str, _Budget | None] = {
    "below": _Budget(20.0),
    "above": _Budget(80.0),
    "equal": _Budget(_NEW_DEFAULT),
    "weekly": _Budget(10.0, budget_duration="7d"),
    "own_rpm": _Budget(20.0, rpm_limit=5),
    "on_default": None,
}


async def _seed_budget(db, budget_id: str, budget: _Budget) -> str:
    await db.db.litellm_budgettable.create(
        data={
            "budget_id": budget_id,
            "max_budget": budget.max_budget,
            "budget_duration": budget.budget_duration,
            "rpm_limit": budget.rpm_limit,
            "created_by": "backfill-test",
            "updated_by": "backfill-test",
        }
    )
    return budget_id


async def _seed_membership(db, resources, name: str, budget_id: str | None) -> None:
    key = {"user_id": resources.tag(name), "team_id": resources.prefix}
    await db.db.litellm_teammembership.upsert(
        where={"user_id_team_id": key},
        data={"create": {**key, "budget_id": budget_id}, "update": {"budget_id": budget_id}},
    )


async def _seed_team(
    db,
    resources,
    roster: Mapping[str, _Budget | None],
    *,
    unbudgeted: tuple[str, ...] = (),
    rowless: tuple[str, ...] = (),
) -> str:
    """``roster`` members hold their own budget, or the default when None. ``unbudgeted`` members have a membership
    row with no budget, ``rowless`` ones are on the team with no membership row at all."""
    default_budget_id = await _seed_budget(db, resources.tag("default"), _Budget(_OLD_DEFAULT))
    await create_scratch_team(
        db, resources.prefix, member_user_ids=[resources.tag(name) for name in (*roster, *unbudgeted, *rowless)]
    )
    await db.db.litellm_teamtable.update(
        where={"team_id": resources.prefix}, data={"metadata": Json({"team_member_budget_id": default_budget_id})}
    )
    for name, budget in roster.items():
        own = None if budget is None else await _seed_budget(db, resources.tag(f"own-{name}"), budget)
        await _seed_membership(db, resources, name, own or default_budget_id)
    for name in unbudgeted:
        await _seed_membership(db, resources, name, None)
    await db.db.litellm_teammembership.delete_many(
        where={"team_id": resources.prefix, "user_id": {"in": [resources.tag(name) for name in rowless]}}
    )
    return default_budget_id


def _update_team(db, team_id: str, body: Mapping[str, object]):
    return db.gateway.request("POST", "/team/update", {"team_id": team_id, **body})


async def _member_budget(db, resources, name: str):
    membership = await db.db.litellm_teammembership.find_unique(
        where={"user_id_team_id": {"user_id": resources.tag(name), "team_id": resources.prefix}},
        include={"litellm_budget_table": True},
    )
    assert membership is not None
    return membership


@pytest.mark.parametrize(
    "backfill,moved,skipped",
    [
        ("raise_to_default", {"below", "own_rpm"}, 1),
        ("lower_to_default", {"above"}, 0),
        ("all_to_default", {"below", "own_rpm", "above"}, 1),
    ],
)
async def test_backfill_moves_only_members_on_the_chosen_side_of_the_new_default(
    backfill: str, moved: set[str], skipped: int, db, resources
):
    default_budget_id = await _seed_team(db, resources, _ROSTER)

    response = _update_team(
        db,
        resources.prefix,
        {"team_member_budget": _NEW_DEFAULT, "team_member_budget_backfill": backfill},
    )

    assert response.status_code == 200, response.text
    assert response.json()["team_member_budget_backfill_result"] == {
        "moved_to_default": len(moved),
        "skipped_other_reset_period": skipped,
    }
    for name, budget in _ROSTER.items():
        membership = await _member_budget(db, resources, name)
        if name == "own_rpm" and name in moved:
            assert membership.budget_id == resources.tag("own-own_rpm")
            assert membership.litellm_budget_table.max_budget is None
            assert membership.litellm_budget_table.rpm_limit == 5
        elif name in moved or budget is None:
            assert membership.budget_id == default_budget_id, name
        else:
            assert membership.litellm_budget_table.max_budget == budget.max_budget, name
    default = await db.db.litellm_budgettable.find_unique(where={"budget_id": default_budget_id})
    assert default.max_budget == _NEW_DEFAULT


async def test_saving_a_new_default_without_backfill_keeps_every_custom_budget(db, resources):
    await _seed_team(db, resources, _ROSTER)

    response = _update_team(db, resources.prefix, {"team_member_budget": _NEW_DEFAULT})

    assert response.status_code == 200, response.text
    assert "team_member_budget_backfill_result" not in response.json()
    for name, budget in _ROSTER.items():
        if budget is not None:
            assert (await _member_budget(db, resources, name)).litellm_budget_table.max_budget == budget.max_budget


async def test_backfill_gives_members_sharing_a_custom_budget_their_own_copy(db, resources):
    await _seed_team(db, resources, {})
    shared_budget_id = await _seed_budget(db, resources.tag("shared"), _Budget(20.0, rpm_limit=5))
    for name in ("first", "second"):
        await _seed_membership(db, resources, name, shared_budget_id)

    response = _update_team(
        db,
        resources.prefix,
        {"team_member_budget": _NEW_DEFAULT, "team_member_budget_backfill": "raise_to_default"},
    )

    assert response.status_code == 200, response.text
    copies = [await _member_budget(db, resources, name) for name in ("first", "second")]
    assert len({copy.budget_id for copy in copies} | {shared_budget_id}) == 3
    assert [(copy.litellm_budget_table.max_budget, copy.litellm_budget_table.rpm_limit) for copy in copies] == [
        (None, 5),
        (None, 5),
    ]
    shared = await db.db.litellm_budgettable.find_unique(where={"budget_id": shared_budget_id})
    assert shared.max_budget == 20.0


async def test_backfill_links_members_sharing_a_plain_custom_budget_to_the_default(db, resources):
    default_budget_id = await _seed_team(db, resources, {})
    shared_budget_id = await _seed_budget(db, resources.tag("shared"), _Budget(20.0))
    for name in ("first", "second"):
        await _seed_membership(db, resources, name, shared_budget_id)

    response = _update_team(
        db,
        resources.prefix,
        {"team_member_budget": _NEW_DEFAULT, "team_member_budget_backfill": "raise_to_default"},
    )

    assert response.status_code == 200, response.text
    assert [(await _member_budget(db, resources, name)).budget_id for name in ("first", "second")] == [
        default_budget_id,
        default_budget_id,
    ]
    shared = await db.db.litellm_budgettable.find_unique(where={"budget_id": shared_budget_id})
    assert shared.max_budget == 20.0


async def test_backfill_over_the_member_cap_rolls_back_the_new_default(db, resources):
    default_budget_id = await _seed_team(db, resources, {})
    over_cap = MAX_MEMBER_BUDGETS_MOVED_PER_UPDATE + 1
    await db.db.execute_raw(
        """
        WITH members AS (SELECT $1 || '-bulk-' || n AS user_id FROM generate_series(1, $3::int) AS n),
        budgets AS (
            INSERT INTO "LiteLLM_BudgetTable" (budget_id, max_budget, budget_duration, created_by, updated_by)
            SELECT user_id, 20.0, $4, 'backfill-test', 'backfill-test' FROM members RETURNING budget_id
        )
        INSERT INTO "LiteLLM_TeamMembership" (user_id, team_id, budget_id)
        SELECT budget_id, $2, budget_id FROM budgets
        """,
        resources.prefix,
        resources.prefix,
        over_cap,
        _PERIOD,
    )

    response = _update_team(
        db,
        resources.prefix,
        {"team_member_budget": _NEW_DEFAULT, "team_member_budget_backfill": "raise_to_default"},
    )

    assert response.status_code == 422, response.text
    assert str(over_cap) in response.text
    default = await db.db.litellm_budgettable.find_unique(where={"budget_id": default_budget_id})
    assert default.max_budget == _OLD_DEFAULT
    sample = await db.db.litellm_budgettable.find_unique(where={"budget_id": f"{resources.prefix}-bulk-1"})
    assert sample.max_budget == 20.0


async def test_backfill_without_a_new_default_is_rejected(resources, db):
    await _seed_team(db, resources, {})

    response = _update_team(db, resources.prefix, {"team_member_budget_backfill": "raise_to_default"})

    assert response.status_code == 422, response.text
    assert "team_member_budget_backfill needs a positive team_member_budget" in response.text


async def test_clearing_a_members_only_limit_links_them_back_to_the_team_default(db, resources):
    default_budget_id = await _seed_team(db, resources, {"capped": _Budget(20.0, budget_duration=None)})

    response = db.gateway.request(
        "POST",
        f"/management/v1/teams/{resources.prefix}/members/bulk_update",
        {"members": [{"user_id": resources.tag("capped"), "max_budget_in_team": None}]},
    )

    assert response.status_code == 200, response.text
    assert (await _member_budget(db, resources, "capped")).budget_id == default_budget_id


async def test_saving_a_default_links_every_roster_member_without_a_budget_to_it(db, resources):
    default_budget_id = await _seed_team(db, resources, {}, unbudgeted=("no-budget",), rowless=("no-row",))

    response = _update_team(db, resources.prefix, {"team_member_budget": _NEW_DEFAULT})

    assert response.status_code == 200, response.text
    for name in ("no-row", "no-budget"):
        assert (await _member_budget(db, resources, name)).budget_id == default_budget_id, name
