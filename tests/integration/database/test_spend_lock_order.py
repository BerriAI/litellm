import asyncio
import subprocess
import sys
from collections.abc import AsyncIterator, Callable, Iterator, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Final

import psycopg
import pytest
import pytest_asyncio
from prisma import Prisma

from litellm.proxy._types import DBSpendUpdateTransactions
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.db.db_spend_update_writer import DBSpendUpdateWriter
from litellm.proxy.management_helpers.model_allowlist_rename_sync import sync_model_allowlists_for_renamed_model
from litellm.repositories import unit_of_work
from tests.integration._support.database import scratch_database

_ROOT: Final = Path(__file__).resolve().parents[3]
_KEY: Final = "2c70e12b7a0646f92279f427c7b38e7334d8e5389cff167a1dc30e73f826b683"
_CHARGES: Final = DBSpendUpdateTransactions(
    user_list_transactions={"user": 1.0},
    end_user_list_transactions={},
    key_list_transactions={_KEY: 1.0},
    team_list_transactions={"team": 1.0},
    team_member_list_transactions={"team_id::team::user_id::user": 1.0},
    org_list_transactions={"org": 1.0},
    org_member_list_transactions={"org_id::org::user_id::user": 1.0},
    project_list_transactions={"project": 1.0},
    tag_list_transactions={"tag": 1.0},
    agent_list_transactions={},
    model_access_group_list_transactions={"group": 1.0},
)
_SEED: Final = (
    "INSERT INTO \"LiteLLM_BudgetTable\" (budget_id, created_by, updated_by) VALUES ('budget', 't', 't')",
    'INSERT INTO "LiteLLM_OrganizationTable" (organization_id, organization_alias, budget_id, created_by, updated_by) '
    "VALUES ('org', 'org', 'budget', 't', 't')",
    "INSERT INTO \"LiteLLM_UserTable\" (user_id) VALUES ('user')",
    'INSERT INTO "LiteLLM_TeamTable" (team_id, members_with_roles) '
    """VALUES ('team', '[{"user_id": "user", "role": "user"}]')""",
    "INSERT INTO \"LiteLLM_TeamMembership\" (team_id, user_id, budget_id) VALUES ('team', 'user', 'budget')",
    "INSERT INTO \"LiteLLM_OrganizationMembership\" (organization_id, user_id) VALUES ('org', 'user')",
    f"INSERT INTO \"LiteLLM_VerificationToken\" (token, budget_id) VALUES ('{_KEY}', 'budget')",
    "INSERT INTO \"LiteLLM_ProjectTable\" (project_id, budget_id, created_by, updated_by) "
    "VALUES ('project', 'budget', 't', 't')",
    "INSERT INTO \"LiteLLM_TagTable\" (tag_name, budget_id) VALUES ('tag', 'budget')",
    "INSERT INTO \"LiteLLM_ModelAccessGroupBudgetTable\" (access_group_name, budget_id) VALUES ('group', 'budget')",
)


@pytest.fixture(scope="module")
def schema_sql() -> str:
    return subprocess.run(
        [sys.executable, "-I", "-m", "prisma", "migrate", "diff", "--from-empty", "--to-schema-datamodel",
         str(_ROOT / "litellm/schema.prisma"), "--script"],
        check=True,
        capture_output=True,
        text=True,
        cwd=_ROOT,
    ).stdout


@pytest.fixture
def database(schema_sql: str) -> Iterator[str]:
    with scratch_database("spend_lock_order") as url, psycopg.connect(url, autocommit=True) as connection:
        connection.execute(schema_sql)
        for statement in _SEED:
            connection.execute(statement)
        yield url


@pytest_asyncio.fixture(loop_scope="function")
async def ledger(database: str) -> AsyncIterator[Prisma]:
    client: Final = Prisma(datasource={"url": database})
    await client.connect()
    try:
        yield client
    finally:
        await client.disconnect()


async def _apply_charges(client: Prisma) -> None:
    async with client.tx() as transaction:
        await DBSpendUpdateWriter().apply_accounting_entities(client, None, transaction, _CHARGES)


async def _ledger_waits_for_a_lock(url: str) -> None:
    with psycopg.connect(url, autocommit=True) as observer:
        for _ in range(200):
            if observer.execute("SELECT 1 FROM pg_locks WHERE NOT granted").fetchone():
                return
            await asyncio.sleep(0.05)
    raise AssertionError("the ledger never waited on the other writer")


def _spend(url: str) -> Sequence[object]:
    with psycopg.connect(url) as connection:
        return connection.execute(
            'SELECT (SELECT spend FROM "LiteLLM_UserTable"), (SELECT spend FROM "LiteLLM_TeamTable"), '
            '(SELECT spend FROM "LiteLLM_TeamMembership"), (SELECT spend FROM "LiteLLM_VerificationToken"), '
            '(SELECT spend FROM "LiteLLM_OrganizationTable"), (SELECT spend FROM "LiteLLM_ProjectTable")'
        ).fetchone() or ()


_Writer = Callable[[psycopg.Connection[tuple[object, ...]]], None]


def _roster_first(connection: psycopg.Connection[tuple[object, ...]]) -> None:
    connection.execute("SELECT pg_advisory_xact_lock(hashtext('team'))")


def _roster_rest(connection: psycopg.Connection[tuple[object, ...]]) -> None:
    connection.execute("SELECT 1 FROM \"LiteLLM_UserTable\" WHERE user_id = 'user' FOR UPDATE")
    connection.execute("UPDATE \"LiteLLM_TeamTable\" SET members_with_roles = members_with_roles WHERE team_id = 'team'")


def _access_group_first(connection: psycopg.Connection[tuple[object, ...]]) -> None:
    connection.execute("UPDATE \"LiteLLM_TeamTable\" SET access_group_ids = ARRAY['ag'] WHERE team_id = 'team'")


def _access_group_rest(connection: psycopg.Connection[tuple[object, ...]]) -> None:
    connection.execute(f"UPDATE \"LiteLLM_VerificationToken\" SET access_group_ids = ARRAY['ag'] WHERE token = '{_KEY}'")


def _cascade_first(connection: psycopg.Connection[tuple[object, ...]]) -> None:
    connection.execute(unit_of_work._LOCK_BUDGET_ORGS_SQL.replace("$1", "%s"), (["budget"],))


def _cascade_rest(connection: psycopg.Connection[tuple[object, ...]]) -> None:
    for statement in (
        unit_of_work._LOCK_BUDGET_MEMBERSHIPS_SQL,
        unit_of_work._LOCK_BUDGET_KEYS_SQL,
        unit_of_work._LOCK_BUDGET_PROJECTS_SQL,
        unit_of_work._LOCK_BUDGET_TAGS_SQL,
        unit_of_work._LOCK_BUDGET_ACCESS_GROUPS_SQL,
    ):
        connection.execute(statement.replace("$1", "%s"), (["budget"],))
    connection.execute("UPDATE \"LiteLLM_TagTable\" SET spend = 0 WHERE budget_id = 'budget'")


@pytest.mark.parametrize(
    ("first", "rest"),
    [(_roster_first, _roster_rest), (_access_group_first, _access_group_rest), (_cascade_first, _cascade_rest)],
    ids=["team_roster", "access_group", "budget_cascade"],
)
@pytest.mark.asyncio(loop_scope="function")
async def test_charges_wait_for_a_concurrent_writer_instead_of_deadlocking(
    database: str, ledger: Prisma, first: _Writer, rest: _Writer
) -> None:
    """The other writer takes its first lock, the ledger starts and must block holding nothing the other writer
    needs next, so the other writer finishes and the charge lands once after it."""
    with psycopg.connect(database) as writer:
        writer.execute("SET lock_timeout = '10s'")
        first(writer)
        charges: Final = asyncio.create_task(_apply_charges(ledger))
        await _ledger_waits_for_a_lock(database)
        await asyncio.to_thread(rest, writer)
        writer.commit()
    await asyncio.wait_for(charges, timeout=20)

    assert _spend(database) == (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)


async def _rename(client: Prisma) -> None:
    await sync_model_allowlists_for_renamed_model(
        SimpleNamespace(db=client),
        model_id="deployment",
        old_name="old-model",
        new_name="new-model",
        llm_router=None,
        user_api_key_cache=UserApiKeyCache(),
    )


@pytest.mark.asyncio(loop_scope="function")
async def test_a_model_rename_waits_for_a_roster_change_instead_of_deadlocking(database: str, ledger: Prisma) -> None:
    """The roster change holds the team lock and the user row, so the rename must wait before it locks the team
    row the roster change updates next."""
    with psycopg.connect(database, autocommit=True) as seed:
        seed.execute("UPDATE \"LiteLLM_TeamTable\" SET models = ARRAY['old-model']")
        seed.execute("UPDATE \"LiteLLM_UserTable\" SET models = ARRAY['old-model']")
    with psycopg.connect(database) as writer:
        writer.execute("SET lock_timeout = '10s'")
        _roster_first(writer)
        writer.execute("SELECT 1 FROM \"LiteLLM_UserTable\" WHERE user_id = 'user' FOR UPDATE")
        rename: Final = asyncio.create_task(_rename(ledger))
        await _ledger_waits_for_a_lock(database)
        await asyncio.to_thread(_roster_rest, writer)
        writer.commit()
    await asyncio.wait_for(rename, timeout=20)

    with psycopg.connect(database) as connection:
        assert connection.execute(
            'SELECT (SELECT models FROM "LiteLLM_TeamTable"), (SELECT models FROM "LiteLLM_UserTable")'
        ).fetchone() == (["new-model"], ["new-model"])
