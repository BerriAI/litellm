from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.proxy.db.prisma_client import PrismaWrapper
from litellm.proxy.db.routing_prisma_wrapper import RoutingPrismaWrapper
from litellm.proxy.management_helpers.access_group_key_sync import (
    sync_key_access_group_membership,
    sync_key_regeneration_access_group_membership,
)


def _routed_prisma_client():
    writer_inner = MagicMock(name="writer_prisma")
    reader_inner = MagicMock(name="reader_prisma")
    writer_inner.query_raw = AsyncMock(return_value=[])
    reader_inner.query_raw = AsyncMock(return_value=[])
    writer = PrismaWrapper(original_prisma=writer_inner, iam_token_db_auth=False)
    reader = PrismaWrapper(original_prisma=reader_inner, iam_token_db_auth=False)
    routing = RoutingPrismaWrapper(writer=writer, reader=reader)
    return SimpleNamespace(db=routing), writer_inner, reader_inner


@pytest.mark.asyncio
async def test_regeneration_repoint_update_runs_on_the_writer():
    prisma_client, writer_inner, reader_inner = _routed_prisma_client()

    await sync_key_regeneration_access_group_membership(
        prisma_client=prisma_client,
        previous_key_token="old-token",
        new_key_token="new-token",
        data=None,
        existing_key_row=MagicMock(),
    )

    writer_inner.query_raw.assert_awaited_once()
    assert writer_inner.query_raw.await_args.args[0].startswith('UPDATE "LiteLLM_AccessGroupTable"')
    reader_inner.query_raw.assert_not_awaited()


@pytest.mark.asyncio
async def test_membership_attach_and_detach_updates_run_on_the_writer():
    prisma_client, writer_inner, reader_inner = _routed_prisma_client()

    await sync_key_access_group_membership(
        prisma_client=prisma_client,
        key_token="token",
        previous_access_group_ids=["ag-old"],
        updated_access_group_ids=["ag-new"],
    )

    assert writer_inner.query_raw.await_count == 2
    assert all(
        call.args[0].startswith('UPDATE "LiteLLM_AccessGroupTable"') for call in writer_inner.query_raw.await_args_list
    )
    reader_inner.query_raw.assert_not_awaited()
