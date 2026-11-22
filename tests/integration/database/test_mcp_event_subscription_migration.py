import os
import uuid
from typing import Final

import pytest

from tests.integration._support.prisma_clients import connected_proxy_prisma


@pytest.mark.asyncio
async def test_a_new_subscription_gets_the_database_defaults() -> None:
    row_id: Final = uuid.uuid4().hex
    async with connected_proxy_prisma(os.environ["DATABASE_URL"]) as prisma:
        try:
            row: Final = await prisma.writer_db.query_raw(
                """INSERT INTO "LiteLLM_MCPEventSubscription" (id,principal,server_id,event_name,client_url,client_secret,upstream_secret) VALUES ($1,'p','s','evt','https://example.test','sealed-a','sealed-b')
                RETURNING status, consecutive_failures, refresh_before""",
                row_id,
            )
            assert row == [{"status": "pending", "consecutive_failures": 0, "refresh_before": None}]
        finally:
            await prisma.writer_db.execute_raw('DELETE FROM "LiteLLM_MCPEventSubscription" WHERE id = $1', row_id)
