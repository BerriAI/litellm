import os
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from typing import Final
from uuid import uuid4

import psycopg
import pytest
from psycopg import sql

from litellm.proxy._types import LiteLLM_VerificationToken
from litellm.proxy.management_endpoints.key_management_endpoints import (
    _prepare_key_update_object_permission,
)
from litellm.proxy.management_helpers.object_permission_utils import handle_update_object_permission_common
from tests.integration._support.client import Gateway, eventually, object_value
from tests.integration._support.database import write_rows
from tests.integration._support.prisma_clients import connected_proxy_prisma


@pytest.mark.asyncio
async def test_an_event_grant_survives_an_unrelated_permission_upsert() -> None:
    grants: Final = {"ci-server": ["build.*"]}
    async with connected_proxy_prisma(os.environ["DATABASE_URL"]) as client:
        permission_id: Final = await handle_update_object_permission_common(
            {"object_permission": {"mcp_event_permissions": grants}}, None, client
        )
        assert permission_id is not None
        table: Final = client.db.litellm_objectpermissiontable
        where: Final = {"object_permission_id": permission_id}
        try:
            created: Final = await table.find_unique(where=where)
            await handle_update_object_permission_common(
                {"object_permission": {"vector_stores": ["docs"]}}, permission_id, client
            )
            edited: Final = await table.find_unique(where=where)
            assert created is not None and edited is not None
            assert (created.mcp_event_permissions, edited.mcp_event_permissions, edited.vector_stores) == (
                grants,
                grants,
                ["docs"],
            )
        finally:
            await table.delete(where=where)


@pytest.mark.asyncio
async def test_prepared_unrelated_edit_keeps_the_last_saved_event_grant() -> None:
    async with connected_proxy_prisma(os.environ["DATABASE_URL"]) as client:
        where = {"object_permission_id": str(uuid4())}
        table = client.db.litellm_objectpermissiontable
        await table.create(data={**where, "mcp_event_permissions": '{"ci":["old"]}'})
        try:
            key = LiteLLM_VerificationToken(**where)
            prepared = await _prepare_key_update_object_permission({"vector_stores": ["docs"]}, key, client)
            await handle_update_object_permission_common(
                {"object_permission": {"mcp_event_permissions": {"ci": ["new"]}}}, key.object_permission_id, client
            )
            assert prepared is not None
            await prepared.write(client)
            row = await table.find_unique(where=where)
            assert row is not None and (row.vector_stores, row.mcp_event_permissions) == (["docs"], {"ci": ["new"]})
        finally:
            await table.delete(where=where)


def test_first_event_save_keeps_key_server_scope_until_an_explicit_edit(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        key = scenario.key()
        key_hash = sha256(key.encode()).hexdigest()
        events = {"ci": ["build.*"]}

        def save(**fields):
            updated = gateway.post("/key/update", {"key": key_hash, "object_permission": fields})
            info = object_value(gateway.get("/key/info", {"key": key_hash})["info"])
            assert updated["object_permission"] == info["object_permission"]
            return object_value(info["object_permission"])

        first = save(mcp_event_permissions=events)
        assert (first["mcp_servers"], first["mcp_event_permissions"]) == (None, events)
        assert save(vector_stores=[])["mcp_servers"] is None
        assert save(mcp_servers=[])["mcp_servers"] == []
        cleared = save(mcp_event_permissions={})
        assert (cleared["mcp_servers"], cleared["mcp_event_permissions"]) == ([], {})


def test_event_only_key_generation_preserves_absent_server_scope(gateway: Gateway) -> None:
    events = {"ci": ["build.*"]}
    with gateway.scenario() as scenario:
        key = scenario.key(object_permission={"mcp_event_permissions": events})
        info = object_value(gateway.get("/key/info", {"key": sha256(key.encode()).hexdigest()})["info"])
        permission = object_value(info["object_permission"])
        assert (permission["mcp_servers"], permission["mcp_event_permissions"]) == (None, events)


def test_rejected_team_event_alias_preserves_the_member_budget(gateway: Gateway) -> None:
    alias = str(uuid4())
    first, second = str(uuid4()), str(uuid4())
    with gateway.scenario() as scenario:
        team = scenario.team(team_member_budget=10)
        write_rows(
            'INSERT INTO "LiteLLM_MCPServerTable" (server_id, alias) VALUES (%s, %s), (%s, %s)',
            (first, alias, second, alias),
        )
        try:
            response = gateway.request(
                "POST",
                "/team/update",
                {
                    "team_id": team,
                    "team_member_budget": 24,
                    "object_permission": {"mcp_event_permissions": {alias: ["build.*"]}},
                },
            )
            assert response.status_code == 400 and "Ambiguous mcp_event_permissions key" in response.text
            rows = gateway.rows(
                'SELECT b.max_budget FROM "LiteLLM_TeamTable" t JOIN "LiteLLM_BudgetTable" b '
                "ON b.budget_id=t.metadata->>'team_member_budget_id' WHERE t.team_id=%s",
                (team,),
            )
            assert rows == [{"max_budget": 10}]
            gateway.post(
                "/team/update",
                {
                    "team_id": team,
                    "team_member_budget": 24,
                    "object_permission": {"mcp_event_permissions": {first: ["build.*"]}},
                },
            )
            stored = gateway.rows(
                'SELECT b.max_budget, p.mcp_event_permissions FROM "LiteLLM_TeamTable" t JOIN "LiteLLM_BudgetTable" b '
                "ON b.budget_id=t.metadata->>'team_member_budget_id' "
                'JOIN "LiteLLM_ObjectPermissionTable" p ON p.object_permission_id=t.object_permission_id WHERE t.team_id=%s',
                (team,),
            )
            assert stored == [{"max_budget": 24, "mcp_event_permissions": {first: ["build.*"]}}]
        finally:
            write_rows('DELETE FROM "LiteLLM_MCPServerTable" WHERE server_id IN (%s, %s)', (first, second))


def test_first_saves_on_distinct_keys_both_keep_inherited_scope(gateway: Gateway) -> None:
    with gateway.scenario() as scenario, ThreadPoolExecutor(2) as writers:
        keys = tuple(sha256(scenario.key().encode()).hexdigest() for _ in range(2))
        writes = [
            writers.submit(
                gateway.post,
                "/key/update",
                {"key": key, "object_permission": {"mcp_event_permissions": {"ci": ["build.*"]}}},
            )
            for key in keys
        ]
        assert [object_value(write.result()["object_permission"])["mcp_servers"] for write in writes] == [None, None]
        assert gateway.rows(
            'SELECT count(DISTINCT object_permission_id) AS linked FROM "LiteLLM_VerificationToken" k '
            'JOIN "LiteLLM_ObjectPermissionTable" p USING(object_permission_id) '
            "WHERE k.token IN (%s, %s) AND p.mcp_event_scope_inherited",
            keys,
        ) == [{"linked": 2}]


def test_competing_first_saves_link_one_permission_row(gateway: Gateway) -> None:
    grant = str(uuid4())
    with gateway.scenario() as scenario, ThreadPoolExecutor(2) as writers:
        token = sha256(scenario.key().encode()).hexdigest()
        with psycopg.connect(os.environ["DATABASE_URL"]) as blocker:
            blocker.execute('SELECT token FROM "LiteLLM_VerificationToken" WHERE token=%s FOR UPDATE', (token,))
            writes = [
                writers.submit(
                    gateway.request,
                    "POST",
                    "/key/update",
                    {"key": token, "object_permission": {"mcp_event_permissions": {grant: [event]}}},
                )
                for event in ("a", "b")
            ]
            # Wait for both writes to reach the same row. A tuple-lock waiter may queue behind the first waiter.
            eventually(
                lambda: gateway.rows(
                    "WITH RECURSIVE waiting(pid) AS (SELECT %s::int UNION SELECT a.pid FROM pg_stat_activity a "
                    "JOIN waiting ON waiting.pid = ANY(pg_blocking_pids(a.pid))) SELECT count(*) AS n FROM waiting",
                    (str(blocker.info.backend_pid),),
                ),
                lambda rows: rows == [{"n": 3}],
            )
        assert sorted(write.result().status_code for write in writes) == [200, 409]
        assert gateway.rows(
            'SELECT p.mcp_event_scope_inherited AS inherited, k.token AS owner FROM "LiteLLM_ObjectPermissionTable" p '
            'LEFT JOIN "LiteLLM_VerificationToken" k USING(object_permission_id) WHERE p.mcp_event_permissions ? %s',
            (grant,),
        ) == [{"inherited": True, "owner": token}]
        retry = gateway.post(
            "/key/update", {"key": token, "object_permission": {"mcp_event_permissions": {grant: ["b"]}}}
        )
        assert object_value(retry["object_permission"])["mcp_servers"] is None


@pytest.mark.parametrize("deferred", [False, True], ids=["statement", "commit"])
@pytest.mark.parametrize("owner", ["key", "team"])
def test_rejected_owner_write_rolls_back_permission_and_budget(gateway: Gateway, owner: str, deferred: bool) -> None:
    old = {"ci": ["old"]}
    with gateway.scenario() as scenario, psycopg.connect(os.environ["DATABASE_URL"], autocommit=True) as db:
        if owner == "key":
            token = scenario.key(soft_budget=10, object_permission={"mcp_event_permissions": old})
            identifier = sha256(token.encode()).hexdigest()
            owner_field, budget_field = "key", "soft_budget"
            table, alias = "LiteLLM_VerificationToken", "key_alias"
            query = (
                'SELECT b.soft_budget AS budget, p.mcp_event_permissions AS events FROM "LiteLLM_VerificationToken" k '
                'JOIN "LiteLLM_ObjectPermissionTable" p USING(object_permission_id) '
                'JOIN "LiteLLM_BudgetTable" b ON b.budget_id=k.budget_id WHERE k.token=%s'
            )
        else:
            identifier = scenario.team(team_member_budget=10, object_permission={"mcp_event_permissions": old})
            owner_field, budget_field = "team_id", "team_member_budget"
            table, alias = "LiteLLM_TeamTable", "team_alias"
            query = (
                'SELECT b.max_budget AS budget, p.mcp_event_permissions AS events FROM "LiteLLM_TeamTable" t '
                'JOIN "LiteLLM_ObjectPermissionTable" p USING(object_permission_id) '
                "JOIN \"LiteLLM_BudgetTable\" b ON b.budget_id=t.metadata->>'team_member_budget_id' WHERE t.team_id=%s"
            )
        # This constraint rejects only the scenario's unique alias after all normal input validation has passed.
        rejected = uuid4().hex
        constraint = sql.Identifier(f"reject_{rejected}")
        timing = sql.SQL("DEFERRED") if deferred else sql.SQL("IMMEDIATE")
        with db.transaction():
            db.execute(
                sql.SQL(
                    "CREATE FUNCTION {}() RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN "
                    "RAISE EXCEPTION 'scenario owner constraint'; END; $$"
                ).format(constraint)
            )
            db.execute(
                sql.SQL(
                    "CREATE CONSTRAINT TRIGGER {} AFTER UPDATE ON {} DEFERRABLE INITIALLY {} "
                    "FOR EACH ROW WHEN (NEW.{} = {}) EXECUTE PROCEDURE {}()"
                ).format(
                    constraint, sql.Identifier(table), timing, sql.Identifier(alias), sql.Literal(rejected), constraint
                )
            )
        try:
            data = {
                owner_field: identifier,
                alias: rejected,
                budget_field: 24,
                "object_permission": {"mcp_event_permissions": {"ci": ["new"]}},
            }
            response = gateway.request("POST", f"/{owner}/update", data)
            assert response.is_error, response.text
            assert gateway.rows(query, (identifier,)) == [{"budget": 10, "events": old}]
        finally:
            with db.transaction():
                db.execute(sql.SQL("DROP TRIGGER {} ON {}").format(constraint, sql.Identifier(table)))
                db.execute(sql.SQL("DROP FUNCTION {}()").format(constraint))
        gateway.post(f"/{owner}/update", data)
        assert gateway.rows(query, (identifier,)) == [{"budget": 24, "events": {"ci": ["new"]}}]
