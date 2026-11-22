import os
import uuid
from typing import Final

import pytest

from tests.integration._support.prisma_clients import connected_proxy_prisma


@pytest.mark.asyncio
async def test_a_new_delivery_gets_the_database_defaults() -> None:
    row_id: Final = uuid.uuid4().hex
    async with connected_proxy_prisma(os.environ["DATABASE_URL"]) as prisma:
        try:
            row: Final = await prisma.writer_db.query_raw(
                """INSERT INTO "LiteLLM_MCPEventDelivery" (delivery_id,subscription_id,body,webhook_id) VALUES ($1,'sub-test','{}','w-test')
                RETURNING attempt, prepared, claim_id""",
                row_id,
            )
            assert row == [{"attempt": 0, "prepared": False, "claim_id": None}]
        finally:
            await prisma.writer_db.execute_raw('DELETE FROM "LiteLLM_MCPEventDelivery" WHERE delivery_id = $1', row_id)
