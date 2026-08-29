from types import SimpleNamespace
from typing import Final
from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.proxy._types import LiteLLM_VerificationToken
from litellm.proxy.db.prisma_client import PrismaWrapper
from litellm.proxy.db.routing_prisma_wrapper import RoutingPrismaWrapper
from litellm.proxy.management_helpers.access_group_key_sync import (
    sync_key_regeneration_access_group_membership,
)


@pytest.mark.asyncio
async def test_key_regeneration_access_group_write_uses_primary() -> None:
    writer_query: Final = AsyncMock(return_value=[])
    reader_query: Final = AsyncMock(side_effect=RuntimeError("read-only transaction"))
    writer: Final = PrismaWrapper(MagicMock(query_raw=writer_query))
    reader: Final = PrismaWrapper(MagicMock(query_raw=reader_query))
    prisma_client: Final = SimpleNamespace(db=RoutingPrismaWrapper(writer=writer, reader=reader))

    await sync_key_regeneration_access_group_membership(
        prisma_client=prisma_client,
        previous_key_token="old-hash",
        new_key_token="new-hash",
        data=None,
        existing_key_row=LiteLLM_VerificationToken(token="old-hash"),
    )

    writer_query.assert_awaited_once()
    reader_query.assert_not_awaited()
