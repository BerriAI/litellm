import os
from typing import Final
from uuid import uuid4

import pytest

from litellm.proxy._types import LiteLLM_VerificationToken
from litellm.proxy.management_endpoints.key_management_endpoints import (
    _prepare_key_update_object_permission,
    _write_prepared_key_update_object_permission,
)
from litellm.proxy.management_helpers.object_permission_utils import handle_update_object_permission_common
from tests.integration._support.client import Gateway
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
            await _write_prepared_key_update_object_permission({}, prepared, client)
            row = await table.find_unique(where=where)
            assert row is not None and (row.vector_stores, row.mcp_event_permissions) == (["docs"], {"ci": ["new"]})
        finally:
            await table.delete(where=where)


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
