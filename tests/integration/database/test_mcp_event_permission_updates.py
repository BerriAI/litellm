import os
from typing import Final

import pytest

from litellm.proxy.management_helpers.object_permission_utils import handle_update_object_permission_common
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
