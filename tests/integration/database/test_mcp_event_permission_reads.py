import os
import uuid
from typing import Final

import pytest

from litellm.litellm_core_utils.safe_json_dumps import safe_dumps
from litellm.proxy.management_helpers.object_permission_utils import prepare_object_permission_upsert
from tests.integration._support.prisma_clients import connected_proxy_prisma


@pytest.mark.asyncio
async def test_an_unrelated_key_edit_preserves_a_concurrent_event_grant() -> None:
    server_id: Final = str(uuid.uuid4())
    grants: Final = {server_id: ["build.*"]}
    revoked: Final = {server_id: []}
    async with connected_proxy_prisma(os.environ["DATABASE_URL"]) as client:
        table: Final = client.db.litellm_objectpermissiontable
        row: Final = await table.create(data={"mcp_event_permissions": safe_dumps(grants)})
        where: Final = {"object_permission_id": row.object_permission_id}
        try:
            prepared: Final = await prepare_object_permission_upsert(
                {"vector_stores": ["docs"]}, row.object_permission_id, client
            )
            await table.update(where=where, data={"mcp_event_permissions": safe_dumps(revoked)})
            saved: Final = await table.upsert(where=where, data={"create": prepared.record, "update": prepared.record})
            assert (saved.vector_stores, saved.mcp_event_permissions) == (["docs"], revoked)
        finally:
            await table.delete(where=where)
