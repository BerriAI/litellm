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
