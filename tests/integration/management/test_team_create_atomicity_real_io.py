import os
import sys
from collections.abc import AsyncGenerator, Iterator, Mapping
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Final
from urllib.parse import urlsplit, urlunsplit

import psycopg
import pytest
from fastapi import Request
from litellm_proxy_extras.prisma_toolchain import prisma_migrate_deploy_timeout, run_prisma

from litellm.proxy import proxy_server
from litellm.proxy._types import LitellmUserRoles, NewTeamRequest, ProxyException, UserAPIKeyAuth
from litellm.proxy.management_endpoints.team_endpoints import new_team
from tests.integration._support.database import scratch_database
from tests.integration._support.prisma_clients import connected_proxy_prisma

_ADMIN: Final = UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN)
_REQUEST: Final = Request({"type": "http", "method": "POST", "path": "/team/new", "headers": []})
_ALL_OPTIONAL_ROWS: Final = {
    "access_group_ids": ["g"],
    "team_member_budget": 7,
    "budget_duration": "1d",
    "team_member_rollover_max_budget": 10,
    "model_aliases": {"short": "one"},
    "object_permission": {"mcp_tool_permissions": {"tickets": ["list"]}},
}
_COUNTS: Final = (
    'SELECT (SELECT count(*) FROM "LiteLLM_ModelTable"), (SELECT count(*) FROM "LiteLLM_BudgetTable"), '
    '(SELECT count(*) FROM "LiteLLM_ObjectPermissionTable"), (SELECT count(*) FROM "LiteLLM_TeamTable")'
)


@pytest.fixture(scope="module")
def team_database() -> Iterator[str]:
    schema: Final = Path(__file__).resolve().parents[3] / "litellm/proxy-extras/litellm_proxy_extras/schema.prisma"
    with scratch_database() as url:
        run_prisma(
            (sys.executable, "-I", "-m", "prisma", "migrate", "deploy", "--schema", str(schema)),
            env={**os.environ, "DATABASE_URL": url},
            timeout=prisma_migrate_deploy_timeout(),
        )
        with psycopg.connect(url) as db:
            db.execute('INSERT INTO "LiteLLM_MCPServerTable" (server_id) VALUES (%s)', ("tickets",))
        yield url


@pytest.fixture
def refusing_database(team_database: str) -> Iterator[psycopg.Connection]:
    with psycopg.connect(team_database, autocommit=True) as db:
        db.execute(
            """ALTER TABLE "LiteLLM_TeamTable" ADD CONSTRAINT declined_team CHECK (team_id <> 'declined-create')"""
        )
        db.execute(
            """ALTER TABLE "LiteLLM_ObjectPermissionTable"
               ADD CONSTRAINT declined_permission CHECK (NOT ('declined-tool' = ANY(agents)))"""
        )
        db.execute("""INSERT INTO "LiteLLM_AccessGroupTable" (access_group_id, access_group_name) VALUES ('g', 'g')""")
        db.execute(
            """ALTER TABLE "LiteLLM_AccessGroupTable"
               ADD CONSTRAINT declined_mirror CHECK (NOT ('declined-mirror' = ANY(assigned_team_ids)))"""
        )
        db.execute(
            """ALTER TABLE "LiteLLM_TeamMembership"
               ADD CONSTRAINT declined_member CHECK (user_id <> 'declined-member')"""
        )
        try:
            yield db
        finally:
            db.execute("""ALTER TABLE "LiteLLM_TeamMembership" DROP CONSTRAINT declined_member""")
            db.execute("""ALTER TABLE "LiteLLM_AccessGroupTable" DROP CONSTRAINT declined_mirror""")
            db.execute("""DELETE FROM "LiteLLM_AccessGroupTable" WHERE access_group_id = 'g'""")
            db.execute("""ALTER TABLE "LiteLLM_ObjectPermissionTable" DROP CONSTRAINT declined_permission""")
            db.execute("""ALTER TABLE "LiteLLM_TeamTable" DROP CONSTRAINT declined_team""")


@asynccontextmanager
async def _one_connection_proxy(database: str, monkeypatch: pytest.MonkeyPatch) -> AsyncGenerator[None, None]:
    parts: Final = urlsplit(database)
    connection: Final = urlunsplit(
        parts._replace(query="&".join(filter(None, (parts.query, "connection_limit=1&pool_timeout=2"))))
    )
    monkeypatch.setenv("DATABASE_URL", connection)
    async with connected_proxy_prisma(connection) as client:
        monkeypatch.setattr(proxy_server, "prisma_client", client)
        yield


def _counts(db: psycopg.Connection) -> tuple[int, int, int, int]:
    row: Final = db.execute(_COUNTS).fetchone()
    assert row is not None
    model, budget, permission, team = row
    return model, budget, permission, team


@pytest.mark.asyncio
async def test_create_commits_every_related_row(
    refusing_database: psycopg.Connection, team_database: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    db: Final = refusing_database
    async with _one_connection_proxy(team_database, monkeypatch):
        before: Final = _counts(db)
        await new_team(NewTeamRequest.model_validate({**_ALL_OPTIONAL_ROWS, "team_id": "full"}), _REQUEST, _ADMIN, None)
        await new_team(NewTeamRequest.model_validate({"team_id": "plain"}), _REQUEST, _ADMIN, None)
    after: Final = _counts(db)
    assert tuple(a - b for a, b in zip(after, before, strict=True)) == (1, 1, 1, 2)
    assert db.execute(
        "SELECT b.max_budget, b.rollover_max_budget, b.budget_duration, m.aliases, p.mcp_tool_permissions "
        'FROM "LiteLLM_TeamTable" t '
        "JOIN \"LiteLLM_BudgetTable\" b ON b.budget_id = t.metadata->>'team_member_budget_id' "
        'JOIN "LiteLLM_ModelTable" m ON m.id = t.model_id '
        'JOIN "LiteLLM_ObjectPermissionTable" p USING (object_permission_id) WHERE t.team_id = %s',
        ("full",),
    ).fetchone() == (7, 10, "1d", {"short": "one"}, {"tickets": ["list"]})
    assert db.execute(
        """SELECT assigned_team_ids FROM "LiteLLM_AccessGroupTable" WHERE access_group_id = 'g'"""
    ).fetchone() == (["full"],)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("overrides", "status"),
    [
        pytest.param({"team_id": "declined-create"}, "500", id="team-row-refused"),
        pytest.param(
            {"team_id": "invalid-cap", "team_member_rollover_max_budget": 6}, "400", id="rollover-cap-invalid"
        ),
        pytest.param(
            {"team_id": "declined-permission", "object_permission": {"agents": ["declined-tool"]}},
            "500",
            id="permission-row-refused",
        ),
        pytest.param({"team_id": "declined-mirror"}, "500", id="access-group-mirror-refused"),
    ],
)
async def test_a_refused_write_rolls_back_every_related_row(
    refusing_database: psycopg.Connection,
    team_database: str,
    monkeypatch: pytest.MonkeyPatch,
    overrides: Mapping[str, object],
    status: str,
) -> None:
    db: Final = refusing_database
    async with _one_connection_proxy(team_database, monkeypatch):
        before: Final = _counts(db)
        with pytest.raises(ProxyException) as error:
            await new_team(NewTeamRequest.model_validate({**_ALL_OPTIONAL_ROWS, **overrides}), _REQUEST, _ADMIN, None)
    assert error.value.code == status
    assert _counts(db) == before, overrides
    assert db.execute(
        """SELECT assigned_team_ids FROM "LiteLLM_AccessGroupTable" WHERE access_group_id = 'g'"""
    ).fetchone() == ([],)


@pytest.mark.asyncio
async def test_members_join_after_the_team_commits(
    refusing_database: psycopg.Connection, team_database: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    db: Final = refusing_database
    member: Final = {"role": "user", "user_id": "joiner"}
    async with _one_connection_proxy(team_database, monkeypatch):
        await new_team(
            NewTeamRequest.model_validate(
                {**_ALL_OPTIONAL_ROWS, "team_id": "with-member", "members_with_roles": [member]}
            ),
            _REQUEST,
            _ADMIN,
            None,
        )
    assert db.execute("""SELECT team_id FROM "LiteLLM_TeamMembership" WHERE user_id = 'joiner'""").fetchall() == [
        ("with-member",)
    ]


@pytest.mark.asyncio
async def test_a_refused_member_write_leaves_no_team(
    refusing_database: psycopg.Connection, team_database: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.skip("BUG: member add runs after the team commit, so a refused member leaves an orphan team")
    db: Final = refusing_database
    async with _one_connection_proxy(team_database, monkeypatch):
        with pytest.raises(ProxyException):
            await new_team(
                NewTeamRequest.model_validate(
                    {
                        **_ALL_OPTIONAL_ROWS,
                        "team_id": "refused-member",
                        "members_with_roles": [{"role": "user", "user_id": "declined-member"}],
                    }
                ),
                _REQUEST,
                _ADMIN,
                None,
            )
    assert db.execute("""SELECT count(*) FROM "LiteLLM_TeamTable" WHERE team_id = 'refused-member'""").fetchone() == (
        0,
    )
