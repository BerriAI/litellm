"""
Integration tests for BaseManagedResource listing and access control.
"""

from typing import List
from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.llms.base_llm.managed_resources.base_managed_resource import (
    BaseManagedResource,
)
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth


class _StubResource(BaseManagedResource):
    """Concrete subclass exposing the abstract surface for testing."""

    @property
    def resource_type(self) -> str:
        return "test_resource"

    @property
    def table_name(self) -> str:
        return "litellm_test_resource_table"

    def get_unified_resource_id_format(
        self, resource_object, target_model_names_list
    ) -> str:
        return "test"

    async def create_resource_for_model(
        self, llm_router, model, request_data, litellm_parent_otel_span
    ):
        return {"id": "test"}


def _make_resource(records: List = None) -> _StubResource:
    cache = MagicMock()
    cache.async_get_cache = AsyncMock(return_value=None)

    prisma = MagicMock()
    table = MagicMock()
    table.find_many = AsyncMock(return_value=records or [])
    prisma.db = MagicMock()
    setattr(prisma.db, "litellm_test_resource_table", table)

    return _StubResource(internal_usage_cache=cache, prisma_client=prisma)


@pytest.mark.asyncio
async def test_list_admin_query_is_unscoped():
    resource = _make_resource()
    admin = UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN)

    await resource.list_user_resources(user_api_key_dict=admin)

    table = resource.prisma_client.db.litellm_test_resource_table
    where = table.find_many.await_args.kwargs["where"]
    assert "created_by" not in where
    assert "team_id" not in where


@pytest.mark.asyncio
async def test_list_user_filters_by_user_id():
    resource = _make_resource()
    user = UserAPIKeyAuth(user_id="alice")

    await resource.list_user_resources(user_api_key_dict=user)

    where = resource.prisma_client.db.litellm_test_resource_table.find_many.await_args.kwargs[
        "where"
    ]
    assert where["created_by"] == "alice"
    assert "team_id" not in where


@pytest.mark.asyncio
async def test_list_service_account_filters_by_team_id():
    resource = _make_resource()
    service_account = UserAPIKeyAuth(team_id="team-eng")

    await resource.list_user_resources(user_api_key_dict=service_account)

    where = resource.prisma_client.db.litellm_test_resource_table.find_many.await_args.kwargs[
        "where"
    ]
    assert where["team_id"] == "team-eng"
    assert "created_by" not in where


@pytest.mark.asyncio
async def test_list_identity_less_caller_returns_empty_without_query():
    """A caller with no admin role and no identifying ids must NOT issue a
    query — the original bug skipped the filter and returned everything."""
    resource = _make_resource()
    nobody = UserAPIKeyAuth()

    result = await resource.list_user_resources(user_api_key_dict=nobody)

    assert result == {
        "object": "list",
        "data": [],
        "first_id": None,
        "last_id": None,
        "has_more": False,
    }
    resource.prisma_client.db.litellm_test_resource_table.find_many.assert_not_awaited()


def _make_row(index: int, created_by: str = "alice"):
    row = MagicMock()
    row.id = f"pk-{index:03d}"
    row.unified_resource_id = f"unified-resource-{index:03d}"
    row.created_at = 1_000_000 + index
    row.created_by = created_by
    row.resource_object = {
        "id": f"provider-vector-store-{index:03d}",
        "object": "vector_store",
        "name": f"store-{index:03d}",
    }
    return row


def _row_matches(row, where) -> bool:
    return all(
        isinstance(value, dict) or getattr(row, field, None) == value
        for field, value in where.items()
    )


def _make_paginating_resource(rows: List) -> _StubResource:
    async def find_many(where, take, order, cursor=None, skip=0):
        result = sorted(
            rows, key=lambda r: (r.created_at, r.unified_resource_id), reverse=True
        )
        id_filter = where.get("id")
        if isinstance(id_filter, dict) and "gt" in id_filter:
            result = [r for r in result if r.id > id_filter["gt"]]
        if cursor is not None:
            (cur_field, cur_val), = cursor.items()
            idx = next(
                (i for i, r in enumerate(result) if getattr(r, cur_field) == cur_val),
                None,
            )
            if idx is None:
                return []
            result = result[idx + skip:]
        return result[:take]

    async def find_first(where):
        return next((r for r in rows if _row_matches(r, where)), None)

    cache = MagicMock()
    cache.async_get_cache = AsyncMock(return_value=None)

    prisma = MagicMock()
    table = MagicMock()
    table.find_many = AsyncMock(side_effect=find_many)
    table.find_first = AsyncMock(side_effect=find_first)
    prisma.db = MagicMock()
    setattr(prisma.db, "litellm_test_resource_table", table)

    return _StubResource(internal_usage_cache=cache, prisma_client=prisma)


@pytest.mark.asyncio
async def test_list_resources_pagination_uses_unified_resource_id_cursor():
    """The ``after`` cursor a client sends back is a resource's
    ``unified_resource_id`` -- that is what the listing returns as each item's
    ``id`` and as ``last_id``. Paginating must use a Prisma cursor on that
    unique column, not a ``where id > after`` filter against the random-uuid
    primary key, which compares the cursor to unrelated values and so loops or
    silently drops resources.
    """
    rows = [_make_row(i) for i in range(3)]
    resource = _make_paginating_resource(rows)

    await resource.list_user_resources(
        user_api_key_dict=UserAPIKeyAuth(user_id="alice"),
        limit=2,
        after=rows[2].unified_resource_id,
    )

    table = resource.prisma_client.db.litellm_test_resource_table
    kwargs = table.find_many.await_args.kwargs
    assert kwargs["cursor"] == {"unified_resource_id": rows[2].unified_resource_id}
    assert kwargs["skip"] == 1
    assert kwargs["order"] == [
        {"created_at": "desc"},
        {"unified_resource_id": "desc"},
    ]
    assert "id" not in kwargs["where"]


@pytest.mark.asyncio
async def test_list_resources_pagination_walks_all_pages_without_loops_or_gaps():
    """Walking pages the way a client does -- feeding ``last_id`` back as
    ``after`` -- must return every resource exactly once, newest first."""
    rows = [_make_row(i) for i in range(5)]
    resource = _make_paginating_resource(rows)
    user = UserAPIKeyAuth(user_id="alice")

    seen = []
    after = None
    for _ in range(len(rows) + 5):
        page = await resource.list_user_resources(
            user_api_key_dict=user, limit=2, after=after
        )
        seen.extend(item["id"] for item in page["data"])
        if not page["has_more"]:
            break
        assert page["last_id"] is not None, "has_more was true but there is no cursor"
        assert page["last_id"] != after, "cursor did not advance (pagination loop)"
        after = page["last_id"]

    assert seen == [r.unified_resource_id for r in reversed(rows)]
    assert len(seen) == len(set(seen))


@pytest.mark.asyncio
async def test_list_resources_has_more_false_on_exactly_full_final_page():
    """``has_more`` must mean "another row exists", not "this page is full",
    so a resource count that is an exact multiple of ``limit`` does not cost
    every client an extra empty request."""
    rows = [_make_row(i) for i in range(4)]
    resource = _make_paginating_resource(rows)
    user = UserAPIKeyAuth(user_id="alice")

    first = await resource.list_user_resources(user_api_key_dict=user, limit=2)
    second = await resource.list_user_resources(
        user_api_key_dict=user, limit=2, after=first["last_id"]
    )

    assert first["has_more"] is True
    assert second["has_more"] is False
    assert [item["id"] for item in second["data"]] == [
        rows[1].unified_resource_id,
        rows[0].unified_resource_id,
    ]


@pytest.mark.asyncio
async def test_list_resources_rejects_unknown_after_cursor():
    """An ``after`` that does not resolve to a resource the caller can see is
    a client error. Returning an empty page instead is indistinguishable from
    the end of the list, so a stale cursor silently truncates the listing."""
    from fastapi import HTTPException

    resource = _make_paginating_resource([_make_row(0)])

    with pytest.raises(HTTPException) as exc_info:
        await resource.list_user_resources(
            user_api_key_dict=UserAPIKeyAuth(user_id="alice"),
            limit=2,
            after="does-not-exist-xyz",
        )

    assert exc_info.value.status_code == 400
    assert "does-not-exist-xyz" in str(exc_info.value.detail)
    resource.prisma_client.db.litellm_test_resource_table.find_many.assert_not_awaited()


@pytest.mark.asyncio
async def test_list_resources_cursor_lookup_is_scoped_to_the_caller():
    """The cursor lookup must run against the rows the caller can list. A
    Prisma cursor resolves by unique column regardless of the ``where``
    filter, so an unscoped lookup would let one user anchor their page window
    to another user's resource."""
    from fastapi import HTTPException

    resource = _make_paginating_resource([_make_row(0)])

    with pytest.raises(HTTPException):
        await resource.list_user_resources(
            user_api_key_dict=UserAPIKeyAuth(user_id="alice"),
            limit=2,
            after="unified-resource-000",
            additional_filters={"created_by": "bob"},
        )

    where = resource.prisma_client.db.litellm_test_resource_table.find_first.await_args.kwargs[
        "where"
    ]
    assert where["created_by"] == "bob"
    assert where["unified_resource_id"] == "unified-resource-000"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "caller_team_id,expected",
    [("team-eng", True), ("team-sales", False), (None, False)],
)
async def test_can_access_uses_team_id_for_service_account(caller_team_id, expected):
    cache = MagicMock()
    cache.async_get_cache = AsyncMock(
        return_value={
            "created_by": None,
            "team_id": "team-eng",
        }
    )
    prisma = MagicMock()
    resource = _StubResource(internal_usage_cache=cache, prisma_client=prisma)

    caller = (
        UserAPIKeyAuth(team_id=caller_team_id) if caller_team_id else UserAPIKeyAuth()
    )

    assert await resource.can_user_access_unified_resource_id("rid", caller) is expected
