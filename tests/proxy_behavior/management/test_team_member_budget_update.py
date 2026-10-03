import os
from collections.abc import AsyncIterator
from datetime import datetime, timedelta, timezone
from typing import Final

import httpx
import psycopg
import pytest
import pytest_asyncio
from psycopg import sql
from pydantic import JsonValue, TypeAdapter

from litellm.proxy.utils import PrismaClient
from litellm.repositories.budget_repository import BudgetRepository
from litellm.repositories.table_repositories import TeamMembershipRepository

from .conftest import MASTER_KEY, Scratch, create_scratch_team

pytestmark: Final = pytest.mark.asyncio(loop_scope="session")
_BODY: Final = TypeAdapter(dict[str, JsonValue])
_HEADERS: Final = {"Authorization": f"Bearer {MASTER_KEY}"}


@pytest_asyncio.fixture(autouse=True)
async def reclaim_generated_member_budgets(prisma: PrismaClient, scratch: Scratch) -> AsyncIterator[None]:
    yield
    memberships: Final = TeamMembershipRepository(prisma).table
    rows: Final = await memberships.find_many(where={"team_id": {"startswith": scratch.prefix}})
    budget_ids: Final = tuple(row.budget_id for row in rows if row.budget_id is not None)
    await memberships.delete_many(where={"team_id": {"startswith": scratch.prefix}})
    if budget_ids:
        await BudgetRepository(prisma).table.delete_many(where={"budget_id": {"in": budget_ids}})


async def _budget(prisma: PrismaClient, budget_id: str, amount: int | None, duration: str | None = "1h") -> str:
    await BudgetRepository(prisma).table.create(
        data={
            "budget_id": budget_id,
            "max_budget": amount,
            "budget_duration": duration,
            "rpm_limit": 9,
            "tpm_limit": 90,
            "allowed_models": ["test-model"],
            "budget_reset_at": datetime.now(timezone.utc) + timedelta(hours=1),
            "temp_budget_increase": 200,
            "temp_budget_expiry": datetime.now(timezone.utc) + timedelta(days=1),
            "created_by": "test",
            "updated_by": "test",
        }
    )
    return budget_id


async def _attach(prisma: PrismaClient, team_id: str, user_id: str, budget_id: str | None) -> None:
    await TeamMembershipRepository(prisma).table.create(
        data={
            "team_id": team_id,
            "user_id": user_id,
            "budget_id": budget_id,
            "spend": 23,
        }
    )


async def _update(client: httpx.AsyncClient, team_id: str, mode: str) -> httpx.Response:
    return await client.post(
        "/team/update",
        headers=_HEADERS,
        json={
            "team_id": team_id,
            "team_member_budget": 100,
            "team_member_budget_duration": "1h",
            "team_member_budget_update_mode": mode,
        },
    )


@pytest.mark.parametrize(("mode", "selected"), (("keep", ()), ("raise", (0, 1)), ("lower", (3,)), ("both", (0, 1, 3))))
async def test_selected_amounts_inherit_while_other_fields_and_shared_budgets_survive(
    proxy_client: httpx.AsyncClient,
    prisma: PrismaClient,
    scratch: Scratch,
    mode: str,
    selected: tuple[int, ...],
) -> None:
    default: Final = await _budget(prisma, scratch.tag("default"), 75)
    users: Final = tuple(scratch.tag(str(index)) for index in range(6))
    team: Final = await create_scratch_team(
        prisma, scratch.tag("team"), member_user_ids=users, metadata={"team_member_budget_id": default}
    )
    for index, (amount, period) in enumerate(
        ((0, "60m"), (50, "1h"), (100, "1h"), (150, "1h"), (50, "7d"), (None, "1h"))
    ):
        await _attach(prisma, team, users[index], await _budget(prisma, scratch.tag(f"budget-{index}"), amount, period))
    other_team: Final = await create_scratch_team(prisma, scratch.tag("other-team"))
    await _attach(prisma, other_team, scratch.tag("other-user"), scratch.tag("budget-1"))
    memberships: Final = TeamMembershipRepository(prisma).table
    before: Final = await memberships.find_many(where={"team_id": team}, include={"litellm_budget_table": True})
    response: Final = await _update(proxy_client, team, mode)
    assert response.status_code == 200, response.text
    assert _BODY.validate_json(response.content)["member_budgets_updated"] == len(selected)
    saved_default: Final = await BudgetRepository(prisma).table.find_unique(where={"budget_id": default})
    assert saved_default is not None and saved_default.max_budget == 100
    for original in before:
        current: Final = await memberships.find_unique(
            where={"user_id_team_id": {"user_id": original.user_id, "team_id": team}},
            include={"litellm_budget_table": True},
        )
        assert current is not None and current.spend == original.spend
        old_budget: Final = original.litellm_budget_table
        new_budget: Final = current.litellm_budget_table
        assert old_budget is not None and new_budget is not None
        assert new_budget.max_budget == (None if users.index(original.user_id) in selected else old_budget.max_budget)
        ignored: Final = {"max_budget", "budget_id", "created_at", "updated_at", "created_by", "updated_by"}
        assert new_budget.model_dump(exclude=ignored) == old_budget.model_dump(exclude=ignored)
    shared: Final = await BudgetRepository(prisma).table.find_unique(where={"budget_id": scratch.tag("budget-1")})
    assert shared is not None and shared.max_budget == 50


async def test_first_default_backfills_missing_and_unlinked_members(
    proxy_client: httpx.AsyncClient,
    prisma: PrismaClient,
    scratch: Scratch,
) -> None:
    users: Final = tuple(scratch.tag(str(index)) for index in range(3))
    team: Final = await create_scratch_team(prisma, scratch.tag("team"), member_user_ids=users)
    await _attach(prisma, team, users[0], await _budget(prisma, scratch.tag("custom"), 50))
    await _attach(prisma, team, users[1], None)
    response: Final = await _update(proxy_client, team, "raise")
    assert response.status_code == 200, response.text
    assert _BODY.validate_json(response.content)["member_budgets_updated"] == 1
    rows: Final = await TeamMembershipRepository(prisma).table.find_many(
        where={"team_id": team}, include={"litellm_budget_table": True}
    )
    assert {row.user_id for row in rows} == set(users)
    for row in rows:
        assert row.litellm_budget_table is not None
        assert row.litellm_budget_table.max_budget == (None if row.user_id == users[0] else 100)
        assert row.litellm_budget_table.budget_duration == "1h"


async def test_failed_member_write_rolls_back_default_and_member_changes(
    proxy_client: httpx.AsyncClient,
    prisma: PrismaClient,
    scratch: Scratch,
) -> None:
    default: Final = await _budget(prisma, scratch.tag("default"), 75)
    user: Final = scratch.tag("user")
    team: Final = await create_scratch_team(
        prisma, scratch.tag("team"), member_user_ids=(user,), metadata={"team_member_budget_id": default}
    )
    custom: Final = await _budget(prisma, scratch.tag("custom"), 50)
    await _attach(prisma, team, user, custom)
    budgets: Final = BudgetRepository(prisma).table
    before: Final = await budgets.find_many(where={"budget_id": {"in": (default, custom)}}, order={"budget_id": "asc"})
    constraint: Final = sql.Identifier(scratch.tag("no-clear"))
    async with await psycopg.AsyncConnection.connect(os.environ["DATABASE_URL"], autocommit=True) as connection:
        await connection.execute(
            sql.SQL(
                'ALTER TABLE "LiteLLM_BudgetTable" ADD CONSTRAINT {} CHECK (budget_id <> {} OR max_budget IS NOT NULL)'
            ).format(constraint, sql.Literal(custom))
        )
        try:
            response: Final = await _update(proxy_client, team, "raise")
            assert response.status_code >= 400, response.text
            after: Final = await budgets.find_many(
                where={"budget_id": {"in": (default, custom)}}, order={"budget_id": "asc"}
            )
            assert after == before
        finally:
            await connection.execute(sql.SQL('ALTER TABLE "LiteLLM_BudgetTable" DROP CONSTRAINT {}').format(constraint))


@pytest.mark.parametrize("body", ({"team_member_budget": None}, {"metadata": None}))
async def test_clearing_without_a_default_preserves_null_metadata(
    proxy_client: httpx.AsyncClient,
    prisma: PrismaClient,
    scratch: Scratch,
    body: dict[str, JsonValue],
) -> None:
    from litellm.repositories.team_repository import TeamRepository

    team: Final = await create_scratch_team(prisma, scratch.tag("team"))
    teams: Final = TeamRepository(prisma).table
    await teams.update(where={"team_id": team}, data={"metadata": "null"})
    response: Final = await proxy_client.post("/team/update", headers=_HEADERS, json={"team_id": team, **body})
    assert response.status_code == 200, response.text
    saved: Final = await teams.find_unique(where={"team_id": team})
    assert saved is not None and saved.metadata is None
    assert _BODY.validate_json(response.content)["member_budgets_updated"] == 0


@pytest.mark.parametrize("shared", (False, True))
async def test_clearing_amount_preserves_tpd_only_override(
    proxy_client: httpx.AsyncClient,
    prisma: PrismaClient,
    scratch: Scratch,
    shared: bool,
) -> None:
    default: Final = await _budget(prisma, scratch.tag("tpd-default"), 75, None)
    user: Final = scratch.tag("tpd-user")
    team: Final = await create_scratch_team(
        prisma, scratch.tag("tpd-team"), member_user_ids=(user,), metadata={"team_member_budget_id": default}
    )
    budget: Final = await BudgetRepository(prisma).table.create(
        data={
            "budget_id": scratch.tag("tpd-budget"),
            "max_budget": 50,
            "tpd_limit": 400,
            "created_by": "test",
            "updated_by": "test",
        }
    )
    await _attach(prisma, team, user, budget.budget_id)
    if shared:
        other: Final = await create_scratch_team(prisma, scratch.tag("tpd-other"))
        await _attach(prisma, other, scratch.tag("tpd-other-user"), budget.budget_id)
    response: Final = await proxy_client.post(
        "/team/update",
        headers=_HEADERS,
        json={
            "team_id": team,
            "team_member_budget": 100,
            "team_member_budget_duration": None,
            "team_member_budget_update_mode": "raise",
        },
    )
    assert response.status_code == 200, response.text
    assert _BODY.validate_json(response.content)["member_budgets_updated"] == 1
    current: Final = await TeamMembershipRepository(prisma).table.find_unique(
        where={"user_id_team_id": {"user_id": user, "team_id": team}}, include={"litellm_budget_table": True}
    )
    assert current is not None and current.spend == 23
    assert current.litellm_budget_table is not None
    assert current.litellm_budget_table.max_budget is None
    assert current.litellm_budget_table.tpd_limit == budget.tpd_limit
    source: Final = await BudgetRepository(prisma).table.find_unique(where={"budget_id": budget.budget_id})
    assert source is not None and source.max_budget == (50 if shared else None)


async def test_amount_only_budget_with_null_model_list_disconnects_after_reset(
    proxy_client: httpx.AsyncClient,
    prisma: PrismaClient,
    scratch: Scratch,
) -> None:
    default: Final = await _budget(prisma, scratch.tag("null-model-default"), 75, None)
    user: Final = scratch.tag("null-model-user")
    team: Final = await create_scratch_team(
        prisma, scratch.tag("null-model-team"), member_user_ids=(user,), metadata={"team_member_budget_id": default}
    )
    budget: Final = await BudgetRepository(prisma).table.create(
        data={
            "budget_id": scratch.tag("null-model-budget"),
            "max_budget": 50,
            "created_by": "test",
            "updated_by": "test",
        }
    )
    await _attach(prisma, team, user, budget.budget_id)
    await prisma.db.execute_raw(
        'UPDATE "LiteLLM_BudgetTable" SET allowed_models=NULL WHERE budget_id=$1', budget.budget_id
    )
    response: Final = await proxy_client.post(
        "/team/update",
        headers=_HEADERS,
        json={
            "team_id": team,
            "team_member_budget": 100,
            "team_member_budget_duration": None,
            "team_member_budget_update_mode": "raise",
        },
    )
    assert response.status_code == 200, response.text
    assert _BODY.validate_json(response.content)["member_budgets_updated"] == 1
    current: Final = await TeamMembershipRepository(prisma).table.find_unique(
        where={"user_id_team_id": {"user_id": user, "team_id": team}}
    )
    assert current is not None and current.budget_id is None and current.spend == 23
