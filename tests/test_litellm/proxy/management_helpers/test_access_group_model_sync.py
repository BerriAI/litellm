from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm.proxy.db.prisma_client import PrismaWrapper
from litellm.proxy.db.routing_prisma_wrapper import RoutingPrismaWrapper
from litellm.proxy.management_helpers.access_group_model_sync import (
    sync_access_groups_for_deleted_model,
    sync_access_groups_for_renamed_model,
)

_INVALIDATE = "litellm.proxy.management_helpers.access_group_model_sync.invalidate_access_group_caches"


def _routed_prisma_client(deployment_count: int):
    async def query_raw(sql, *params):
        if sql.startswith("SELECT COUNT(*)"):
            return [{"deployment_count": deployment_count}]
        return [{"access_group_id": "ag-1"}, {"access_group_id": "ag-2"}]

    writer_inner = MagicMock(name="writer_prisma")
    reader_inner = MagicMock(name="reader_prisma")
    writer_inner.query_raw = AsyncMock(side_effect=query_raw)
    reader_inner.query_raw = AsyncMock(side_effect=query_raw)
    writer = PrismaWrapper(original_prisma=writer_inner, iam_token_db_auth=False)
    reader = PrismaWrapper(original_prisma=reader_inner, iam_token_db_auth=False)
    routing = RoutingPrismaWrapper(writer=writer, reader=reader)
    return SimpleNamespace(db=routing), writer_inner, reader_inner


def _access_group_updates(writer_inner):
    return [
        call
        for call in writer_inner.query_raw.await_args_list
        if call.args[0].startswith('UPDATE "LiteLLM_AccessGroupTable"')
    ]


@pytest.mark.asyncio
async def test_rename_replaces_the_old_name_when_no_other_deployment_carries_it():
    prisma_client, writer_inner, reader_inner = _routed_prisma_client(deployment_count=0)

    with patch(_INVALIDATE, new=AsyncMock()) as invalidate:
        await sync_access_groups_for_renamed_model(
            prisma_client, model_id="m-1", old_name="gpt-5.6", new_name="gpt-5.6-eu", llm_router=None
        )

    (update_call,) = _access_group_updates(writer_inner)
    assert "array_replace" in update_call.args[0]
    assert update_call.args[1:] == ("gpt-5.6", "gpt-5.6-eu")
    invalidate.assert_awaited_once_with(("ag-1", "ag-2"))
    reader_inner.query_raw.assert_not_awaited()


@pytest.mark.asyncio
async def test_rename_appends_the_new_name_when_a_sibling_row_keeps_the_old_one():
    prisma_client, writer_inner, _ = _routed_prisma_client(deployment_count=1)

    with patch(_INVALIDATE, new=AsyncMock()) as invalidate:
        await sync_access_groups_for_renamed_model(
            prisma_client, model_id="m-1", old_name="gpt-5.6", new_name="gpt-5.6-eu", llm_router=None
        )

    (update_call,) = _access_group_updates(writer_inner)
    assert "array_append" in update_call.args[0]
    assert "array_replace" not in update_call.args[0]
    assert update_call.args[1:] == ("gpt-5.6", "gpt-5.6-eu")
    invalidate.assert_awaited_once_with(("ag-1", "ag-2"))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "router_ids, expected_write",
    [(["m-1"], "array_replace"), (["m-1", "m-from-config"], "array_append")],
)
async def test_rename_counts_router_deployments_with_another_id_as_backing_the_old_name(router_ids, expected_write):
    prisma_client, writer_inner, _ = _routed_prisma_client(deployment_count=0)
    llm_router = MagicMock()
    llm_router.get_model_ids.return_value = router_ids

    with patch(_INVALIDATE, new=AsyncMock()):
        await sync_access_groups_for_renamed_model(
            prisma_client, model_id="m-1", old_name="gpt-5.6", new_name="gpt-5.6-eu", llm_router=llm_router
        )

    llm_router.get_model_ids.assert_called_once_with(model_name="gpt-5.6")
    (update_call,) = _access_group_updates(writer_inner)
    assert expected_write in update_call.args[0]


@pytest.mark.asyncio
async def test_rename_to_the_same_name_writes_nothing():
    prisma_client, writer_inner, _ = _routed_prisma_client(deployment_count=0)

    with patch(_INVALIDATE, new=AsyncMock()) as invalidate:
        await sync_access_groups_for_renamed_model(
            prisma_client, model_id="m-1", old_name="gpt-5.6", new_name="gpt-5.6", llm_router=None
        )

    assert _access_group_updates(writer_inner) == []
    invalidate.assert_not_awaited()


@pytest.mark.asyncio
async def test_delete_removes_the_name_when_no_row_backs_it_any_more():
    prisma_client, writer_inner, reader_inner = _routed_prisma_client(deployment_count=0)

    with patch(_INVALIDATE, new=AsyncMock()) as invalidate:
        await sync_access_groups_for_deleted_model(prisma_client, model_id="m-1", model_name="gpt-5.6", llm_router=None)

    (update_call,) = _access_group_updates(writer_inner)
    assert "array_remove" in update_call.args[0]
    assert update_call.args[1:] == ("gpt-5.6",)
    invalidate.assert_awaited_once_with(("ag-1", "ag-2"))
    reader_inner.query_raw.assert_not_awaited()


@pytest.mark.asyncio
async def test_delete_keeps_the_name_while_a_sibling_row_still_backs_it():
    prisma_client, writer_inner, _ = _routed_prisma_client(deployment_count=2)

    with patch(_INVALIDATE, new=AsyncMock()) as invalidate:
        await sync_access_groups_for_deleted_model(prisma_client, model_id="m-1", model_name="gpt-5.6", llm_router=None)

    assert _access_group_updates(writer_inner) == []
    invalidate.assert_not_awaited()
