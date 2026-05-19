import json
import os
import sys
from typing import Any, Dict, Optional

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

sys.path.insert(
    0, os.path.abspath("../../../..")
)  # Adds the parent directory to the system path

from unittest.mock import patch

import litellm
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.proxy_server import app
from litellm.types.tag_management import TagDeleteRequest, TagInfoRequest, TagNewRequest

client = TestClient(app)


@pytest.mark.asyncio
async def test_create_and_get_tag():
    """
    Test creation of a new tag and retrieving its information
    """
    from datetime import datetime
    from unittest.mock import AsyncMock, Mock

    from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

    mock_user_auth = UserAPIKeyAuth(
        user_id="test-user-123",
        user_role=LitellmUserRoles.PROXY_ADMIN,
    )
    app.dependency_overrides[user_api_key_auth] = lambda: mock_user_auth

    try:
        with (
            patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma,
            patch("litellm.proxy.proxy_server.llm_router") as mock_router,
            patch(
                "litellm.proxy.proxy_server.litellm_proxy_admin_name", "default_user_id"
            ),
            patch(
                "litellm.proxy.management_endpoints.tag_management_endpoints.get_deployments_by_model"
            ) as mock_get_deployments,
        ):
            # Setup prisma mocks
            mock_db = Mock()
            mock_prisma.db = mock_db

            # Mock find_unique to return None (tag doesn't exist)
            mock_db.litellm_tagtable.find_unique = AsyncMock(return_value=None)

            # Mock find_many for model lookup
            mock_db.litellm_proxymodeltable.find_many = AsyncMock(return_value=[])

            # Mock create to return the created tag
            created_tag = Mock()
            created_tag.tag_name = "test-tag"
            created_tag.description = "Test tag for unit testing"
            created_tag.models = ["model-1"]
            created_tag.model_info = {}
            created_tag.spend = 0.0
            created_tag.budget_id = None
            created_tag.created_at = datetime.now()
            created_tag.updated_at = datetime.now()
            created_tag.created_by = "test-user-123"
            mock_db.litellm_tagtable.create = AsyncMock(return_value=created_tag)

            # Mock get_deployments_by_model to return empty list
            mock_get_deployments.return_value = []

            # Create a new tag
            tag_data = {
                "name": "test-tag",
                "description": "Test tag for unit testing",
                "models": ["model-1"],
            }

            headers = {"Authorization": "Bearer sk-1234"}

            # Test tag creation
            response = client.post("/tag/new", json=tag_data, headers=headers)
            assert response.status_code == 200
            result = response.json()
            assert result["message"] == "Tag test-tag created successfully"
            assert result["tag"]["name"] == "test-tag"
            assert result["tag"]["description"] == "Test tag for unit testing"

            # Mock find_many for tag info retrieval
            retrieved_tag = Mock()
            retrieved_tag.tag_name = "test-tag"
            retrieved_tag.description = "Test tag for unit testing"
            retrieved_tag.models = ["model-1"]
            retrieved_tag.model_info = "{}"
            retrieved_tag.spend = 0.0
            retrieved_tag.budget_id = None
            retrieved_tag.created_at = datetime.now()
            retrieved_tag.updated_at = datetime.now()
            retrieved_tag.created_by = "test-user-123"
            retrieved_tag.litellm_budget_table = None
            mock_db.litellm_tagtable.find_many = AsyncMock(return_value=[retrieved_tag])

            # Test retrieving tag info
            info_data = {"names": ["test-tag"]}
            response = client.post("/tag/info", json=info_data, headers=headers)
            assert response.status_code == 200
            result = response.json()
            assert "test-tag" in result
            assert result["test-tag"]["description"] == "Test tag for unit testing"
    finally:
        # Clean up dependency overrides
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_update_tag():
    """
    Test updating an existing tag
    """
    from datetime import datetime
    from unittest.mock import AsyncMock, Mock

    from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

    mock_user_auth = UserAPIKeyAuth(
        user_id="test-user-123",
        user_role=LitellmUserRoles.PROXY_ADMIN,
    )
    app.dependency_overrides[user_api_key_auth] = lambda: mock_user_auth

    try:
        with (
            patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma,
            patch(
                "litellm.proxy.proxy_server.litellm_proxy_admin_name", "default_user_id"
            ),
        ):
            # Setup prisma mocks
            mock_db = Mock()
            mock_prisma.db = mock_db

            # Mock existing tag
            existing_tag = Mock()
            existing_tag.tag_name = "test-tag"
            existing_tag.description = "Original description"
            existing_tag.models = ["model-1"]
            existing_tag.budget_id = None
            existing_tag.created_at = datetime.now()
            existing_tag.updated_at = datetime.now()
            existing_tag.created_by = "user-123"

            # Mock find_unique to return existing tag
            mock_db.litellm_tagtable.find_unique = AsyncMock(return_value=existing_tag)

            # Mock find_many for model lookup
            mock_db.litellm_proxymodeltable.find_many = AsyncMock(return_value=[])

            # Mock update to return updated tag
            updated_tag = Mock()
            updated_tag.tag_name = "test-tag"
            updated_tag.description = "Updated description"
            updated_tag.models = ["model-1", "model-2"]
            updated_tag.model_info = {}
            updated_tag.spend = 0.0
            updated_tag.budget_id = None
            updated_tag.created_at = datetime.now()
            updated_tag.updated_at = datetime.now()
            updated_tag.created_by = "user-123"
            mock_db.litellm_tagtable.update = AsyncMock(return_value=updated_tag)

            # Update tag data
            update_data = {
                "name": "test-tag",
                "description": "Updated description",
                "models": ["model-1", "model-2"],
            }

            headers = {"Authorization": "Bearer sk-1234"}

            # Test tag update
            response = client.post("/tag/update", json=update_data, headers=headers)
            assert response.status_code == 200
            result = response.json()
            assert result["message"] == "Tag test-tag updated successfully"
            assert result["tag"]["description"] == "Updated description"
            assert len(result["tag"]["models"]) == 2
            assert "model-2" in result["tag"]["models"]
    finally:
        # Clean up dependency overrides
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_delete_tag():
    """
    Test deleting a tag
    """
    from datetime import datetime
    from unittest.mock import AsyncMock, Mock

    from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

    mock_user_auth = UserAPIKeyAuth(
        user_id="test-user-123",
        user_role=LitellmUserRoles.PROXY_ADMIN,
    )
    app.dependency_overrides[user_api_key_auth] = lambda: mock_user_auth

    try:
        with patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma:
            # Setup prisma mocks
            mock_db = Mock()
            mock_prisma.db = mock_db

            # Mock existing tag
            existing_tag = Mock()
            existing_tag.tag_name = "test-tag"
            existing_tag.description = "Test tag for deletion"
            existing_tag.models = ["model-1"]
            existing_tag.created_at = datetime.now()
            existing_tag.updated_at = datetime.now()
            existing_tag.created_by = "user-123"

            # Mock find_unique to return existing tag
            mock_db.litellm_tagtable.find_unique = AsyncMock(return_value=existing_tag)

            # Mock delete
            mock_db.litellm_tagtable.delete = AsyncMock(return_value=existing_tag)

            # Delete tag data
            delete_data = {"name": "test-tag"}

            headers = {"Authorization": "Bearer sk-1234"}

            # Test tag deletion
            response = client.post("/tag/delete", json=delete_data, headers=headers)
            assert response.status_code == 200
            result = response.json()
            assert result["message"] == "Tag test-tag deleted successfully"

            # Verify delete was called
            mock_db.litellm_tagtable.delete.assert_called_once()
    finally:
        # Clean up dependency overrides
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_list_tags_with_dynamic_tags():
    """
    Test that list_tags uses group_by to get distinct dynamic tags efficiently
    and merges them with stored tags, excluding duplicates.
    """
    from datetime import datetime
    from unittest.mock import AsyncMock, Mock

    from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

    mock_user_auth = UserAPIKeyAuth(
        user_id="test-user-123",
        user_role=LitellmUserRoles.PROXY_ADMIN,
    )
    app.dependency_overrides[user_api_key_auth] = lambda: mock_user_auth

    try:
        with patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma:
            mock_db = Mock()
            mock_prisma.db = mock_db

            # Setup stored tags
            stored_tag = Mock()
            stored_tag.tag_name = "stored-tag"
            stored_tag.description = "A stored tag"
            stored_tag.models = ["model-1"]
            stored_tag.model_info = {}
            stored_tag.spend = 0.0
            stored_tag.budget_id = None
            stored_tag.created_at = datetime(2025, 1, 1)
            stored_tag.updated_at = datetime(2025, 1, 1)
            stored_tag.created_by = "user-123"
            stored_tag.litellm_budget_table = None
            mock_db.litellm_tagtable.find_many = AsyncMock(return_value=[stored_tag])

            # Setup dynamic tags via group_by — includes one that overlaps with stored
            mock_db.litellm_dailytagspend.group_by = AsyncMock(
                return_value=[
                    {
                        "tag": "dynamic-tag-1",
                        "_min": {"created_at": "2025-02-01T00:00:00Z"},
                        "_max": {"updated_at": "2025-03-01T00:00:00Z"},
                    },
                    {
                        "tag": "dynamic-tag-2",
                        "_min": {"created_at": "2025-02-02T00:00:00Z"},
                        "_max": {"updated_at": "2025-03-02T00:00:00Z"},
                    },
                    {
                        "tag": "stored-tag",
                        "_min": {"created_at": "2025-01-01T00:00:00Z"},
                        "_max": {"updated_at": "2025-01-01T00:00:00Z"},
                    },  # duplicate, should be excluded
                ]
            )

            headers = {"Authorization": "Bearer sk-1234"}
            response = client.get("/tag/list", headers=headers)

            assert response.status_code == 200
            result = response.json()

            # Should have 1 stored + 2 dynamic (the duplicate excluded)
            assert len(result) == 3

            tag_names = [t["name"] for t in result]
            assert "stored-tag" in tag_names
            assert "dynamic-tag-1" in tag_names
            assert "dynamic-tag-2" in tag_names

            # Verify dynamic tags include created_at/updated_at
            dynamic_tags = {
                t["name"]: t for t in result if t["name"].startswith("dynamic-")
            }
            assert dynamic_tags["dynamic-tag-1"]["created_at"] is not None
            assert dynamic_tags["dynamic-tag-1"]["updated_at"] is not None

    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_list_tags_no_dynamic_tags():
    """
    Test list_tags when there are no dynamic tags in the spend table.
    """
    from datetime import datetime
    from unittest.mock import AsyncMock, Mock

    from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

    mock_user_auth = UserAPIKeyAuth(
        user_id="test-user-123",
        user_role=LitellmUserRoles.PROXY_ADMIN,
    )
    app.dependency_overrides[user_api_key_auth] = lambda: mock_user_auth

    try:
        with patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma:
            mock_db = Mock()
            mock_prisma.db = mock_db

            stored_tag = Mock()
            stored_tag.tag_name = "stored-tag"
            stored_tag.description = "A stored tag"
            stored_tag.models = []
            stored_tag.model_info = None
            stored_tag.spend = 0.0
            stored_tag.budget_id = None
            stored_tag.created_at = datetime(2025, 1, 1)
            stored_tag.updated_at = datetime(2025, 1, 1)
            stored_tag.created_by = "user-123"
            stored_tag.litellm_budget_table = None
            mock_db.litellm_tagtable.find_many = AsyncMock(return_value=[stored_tag])

            mock_db.litellm_dailytagspend.group_by = AsyncMock(return_value=[])

            headers = {"Authorization": "Bearer sk-1234"}
            response = client.get("/tag/list", headers=headers)

            assert response.status_code == 200
            result = response.json()
            assert len(result) == 1
            assert result[0]["name"] == "stored-tag"

    finally:
        app.dependency_overrides.clear()


async def test_internal_user_list_tags_only_returns_tags_used_by_their_keys():
    """
    Internal users can view tag usage, but the tag list must be scoped to tags
    produced by API keys owned by the caller.
    """
    from datetime import datetime
    from unittest.mock import AsyncMock, Mock

    from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

    mock_user_auth = UserAPIKeyAuth(
        api_key="current-owned-key",
        user_id="internal-user-123",
        user_role=LitellmUserRoles.INTERNAL_USER,
    )
    app.dependency_overrides[user_api_key_auth] = lambda: mock_user_auth

    try:
        with patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma:
            mock_db = Mock()
            mock_prisma.db = mock_db

            owned_key_record = Mock()
            owned_key_record.token = "owned-key"
            mock_db.litellm_verificationtoken.find_many = AsyncMock(
                return_value=[owned_key_record]
            )

            mock_db.litellm_dailytagspend.group_by = AsyncMock(
                return_value=[
                    {
                        "tag": "stored-owned-tag",
                        "_min": {"created_at": "2025-02-01T00:00:00Z"},
                        "_max": {"updated_at": "2025-03-01T00:00:00Z"},
                    },
                    {
                        "tag": "dynamic-owned-tag",
                        "_min": {"created_at": "2025-02-02T00:00:00Z"},
                        "_max": {"updated_at": "2025-03-02T00:00:00Z"},
                    },
                ]
            )

            stored_tag = Mock()
            stored_tag.tag_name = "stored-owned-tag"
            stored_tag.description = "A stored tag used by the caller"
            stored_tag.models = ["model-1"]
            stored_tag.model_info = {}
            stored_tag.spend = 0.0
            stored_tag.budget_id = None
            stored_tag.created_at = datetime(2025, 1, 1)
            stored_tag.updated_at = datetime(2025, 1, 1)
            stored_tag.created_by = "admin-user"
            stored_tag.litellm_budget_table = None
            mock_db.litellm_tagtable.find_many = AsyncMock(return_value=[stored_tag])

            response = client.get(
                "/tag/list",
                headers={"Authorization": "Bearer test-key"},
            )

            assert response.status_code == 200
            assert [tag["name"] for tag in response.json()] == [
                "stored-owned-tag",
                "dynamic-owned-tag",
            ]
            mock_db.litellm_verificationtoken.find_many.assert_awaited_once_with(
                where={"user_id": "internal-user-123"},
                select={"token": True},
            )
            mock_db.litellm_dailytagspend.group_by.assert_awaited_once_with(
                by=["tag"],
                where={
                    "tag": {"not": None},
                    "api_key": {"in": ["current-owned-key", "owned-key"]},
                },
                min={"created_at": True},
                max={"updated_at": True},
            )
            mock_db.litellm_tagtable.find_many.assert_awaited_once_with(
                where={"tag_name": {"in": ["stored-owned-tag", "dynamic-owned-tag"]}},
                include={"litellm_budget_table": True},
            )

    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_list_tags_with_date_range_filters_dynamic_tags():
    """
    /tag/list?start_date=...&end_date=... should push the date window into
    the dailytagspend group_by WHERE clause so large tables don't get scanned.
    """
    from unittest.mock import AsyncMock, Mock

    from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

    mock_user_auth = UserAPIKeyAuth(
        user_id="test-user-123",
        user_role=LitellmUserRoles.PROXY_ADMIN,
    )
    app.dependency_overrides[user_api_key_auth] = lambda: mock_user_auth

    try:
        with patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma:
            mock_db = Mock()
            mock_prisma.db = mock_db
            mock_db.litellm_tagtable.find_many = AsyncMock(return_value=[])
            group_by_mock = AsyncMock(return_value=[])
            mock_db.litellm_dailytagspend.group_by = group_by_mock

            headers = {"Authorization": "Bearer sk-1234"}
            response = client.get(
                "/tag/list?start_date=2026-04-01&end_date=2026-04-29",
                headers=headers,
            )

            assert response.status_code == 200
            group_by_mock.assert_awaited_once()
            where = group_by_mock.await_args.kwargs["where"]
            assert where["tag"] == {"not": None}
            assert where["date"] == {"gte": "2026-04-01", "lte": "2026-04-29"}

    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_internal_user_tag_daily_activity_is_scoped_to_their_keys():
    """
    Internal users must not receive proxy-wide tag spend rows when viewing tag
    usage daily activity.
    """
    from unittest.mock import AsyncMock, Mock

    from litellm.proxy.management_endpoints.tag_management_endpoints import (
        get_tag_daily_activity,
    )

    mock_user_auth = UserAPIKeyAuth(
        user_id="internal-user-123",
        user_role=LitellmUserRoles.INTERNAL_USER_VIEW_ONLY,
    )

    with (
        patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma,
        patch(
            "litellm.proxy.management_endpoints.tag_management_endpoints.get_daily_activity",
            new_callable=AsyncMock,
        ) as mock_get_daily_activity,
    ):
        mock_db = Mock()
        mock_prisma.db = mock_db

        owned_key_record = Mock()
        owned_key_record.token = "owned-key"
        mock_db.litellm_verificationtoken.find_many = AsyncMock(
            return_value=[owned_key_record]
        )
        mock_get_daily_activity.return_value = "daily-activity-response"

        result = await get_tag_daily_activity(
            start_date="2025-01-01",
            end_date="2025-01-31",
            user_api_key_dict=mock_user_auth,
        )

        assert result == "daily-activity-response"
        mock_get_daily_activity.assert_awaited_once()
        assert mock_get_daily_activity.await_args.kwargs["api_key"] == ["owned-key"]


@pytest.mark.asyncio
async def test_internal_user_tag_daily_activity_rejects_unowned_api_key_filter():
    """
    If an internal user filters tag usage by an API key they do not own, the
    endpoint should return an empty scoped filter instead of exposing that key's
    tag spend.
    """
    from unittest.mock import AsyncMock, Mock

    from litellm.proxy.management_endpoints.tag_management_endpoints import (
        get_tag_daily_activity,
    )

    mock_user_auth = UserAPIKeyAuth(
        user_id="internal-user-123",
        user_role=LitellmUserRoles.INTERNAL_USER,
    )

    with (
        patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma,
        patch(
            "litellm.proxy.management_endpoints.tag_management_endpoints.get_daily_activity",
            new_callable=AsyncMock,
        ) as mock_get_daily_activity,
    ):
        mock_db = Mock()
        mock_prisma.db = mock_db

        owned_key_record = Mock()
        owned_key_record.token = "owned-key"
        mock_db.litellm_verificationtoken.find_many = AsyncMock(
            return_value=[owned_key_record]
        )
        result = await get_tag_daily_activity(
            start_date="2025-01-01",
            end_date="2025-01-31",
            api_key="unowned-key",
            user_api_key_dict=mock_user_auth,
        )

        assert result.results == []
        assert result.metadata.total_spend == 0
        assert result.metadata.total_api_requests == 0
        mock_get_daily_activity.assert_not_awaited()


@pytest.mark.asyncio
async def test_internal_user_tag_daily_activity_scopes_to_current_key_without_user_id():
    """
    If an internal-user token has no user_id, it should still scope tag usage to
    the current request key instead of falling back to proxy-wide tag spend.
    """
    from unittest.mock import AsyncMock, Mock

    from litellm.proxy.management_endpoints.tag_management_endpoints import (
        get_tag_daily_activity,
    )

    mock_user_auth = UserAPIKeyAuth(
        api_key="current-owned-key",
        user_id=None,
        user_role=LitellmUserRoles.INTERNAL_USER,
    )

    with (
        patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma,
        patch(
            "litellm.proxy.management_endpoints.tag_management_endpoints.get_daily_activity",
            new_callable=AsyncMock,
        ) as mock_get_daily_activity,
    ):
        mock_db = Mock()
        mock_prisma.db = mock_db
        mock_db.litellm_verificationtoken.find_many = AsyncMock(return_value=[])
        mock_get_daily_activity.return_value = "daily-activity-response"

        result = await get_tag_daily_activity(
            start_date="2025-01-01",
            end_date="2025-01-31",
            user_api_key_dict=mock_user_auth,
        )

        assert result == "daily-activity-response"
        mock_db.litellm_verificationtoken.find_many.assert_not_awaited()
        mock_get_daily_activity.assert_awaited_once()
        assert mock_get_daily_activity.await_args.kwargs["api_key"] == [
            "current-owned-key"
        ]


@pytest.mark.asyncio
async def test_internal_user_tag_daily_activity_without_any_scoped_keys_returns_empty():
    """
    If an internal-user token has neither user_id nor api_key, the endpoint must
    return an empty response instead of dropping the API key filter.
    """
    from unittest.mock import AsyncMock, Mock

    from litellm.proxy.management_endpoints.tag_management_endpoints import (
        get_tag_daily_activity,
    )

    mock_user_auth = UserAPIKeyAuth(
        user_id=None,
        user_role=LitellmUserRoles.INTERNAL_USER,
    )

    with (
        patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma,
        patch(
            "litellm.proxy.management_endpoints.tag_management_endpoints.get_daily_activity",
            new_callable=AsyncMock,
        ) as mock_get_daily_activity,
    ):
        mock_db = Mock()
        mock_prisma.db = mock_db
        mock_db.litellm_verificationtoken.find_many = AsyncMock(return_value=[])

        result = await get_tag_daily_activity(
            start_date="2025-01-01",
            end_date="2025-01-31",
            user_api_key_dict=mock_user_auth,
        )

        assert result.results == []
        assert result.metadata.total_spend == 0
        assert result.metadata.total_api_requests == 0
        mock_db.litellm_verificationtoken.find_many.assert_not_awaited()
        mock_get_daily_activity.assert_not_awaited()


@pytest.mark.asyncio
async def test_get_tag_daily_activity_requires_database_connection():
    """
    Tag daily activity should fail with the same explicit DB error used by other
    tag endpoints instead of raising an AttributeError during scope resolution.
    """
    from litellm.proxy.management_endpoints.tag_management_endpoints import (
        get_tag_daily_activity,
    )

    mock_user_auth = UserAPIKeyAuth(
        user_id="internal-user-123",
        user_role=LitellmUserRoles.INTERNAL_USER,
    )

    with patch("litellm.proxy.proxy_server.prisma_client", None):
        with pytest.raises(HTTPException) as exc_info:
            await get_tag_daily_activity(
                start_date="2025-01-01",
                end_date="2025-01-31",
                user_api_key_dict=mock_user_auth,
            )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Database not connected"


@pytest.mark.asyncio
async def test_list_tags_without_date_range_omits_date_filter():
    """When no date range is passed, the WHERE clause must not carry a date key."""
    from unittest.mock import AsyncMock, Mock

    from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

    mock_user_auth = UserAPIKeyAuth(
        user_id="test-user-123",
        user_role=LitellmUserRoles.PROXY_ADMIN,
    )
    app.dependency_overrides[user_api_key_auth] = lambda: mock_user_auth

    try:
        with patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma:
            mock_db = Mock()
            mock_prisma.db = mock_db
            mock_db.litellm_tagtable.find_many = AsyncMock(return_value=[])
            group_by_mock = AsyncMock(return_value=[])
            mock_db.litellm_dailytagspend.group_by = group_by_mock

            headers = {"Authorization": "Bearer sk-1234"}
            response = client.get("/tag/list", headers=headers)

            assert response.status_code == 200
            group_by_mock.assert_awaited_once()
            where = group_by_mock.await_args.kwargs["where"]
            assert "date" not in where

    finally:
        app.dependency_overrides.clear()


@pytest.mark.parametrize(
    "query, expected_detail_fragment",
    [
        ("?start_date=2026-04-01", "must be provided together"),
        ("?end_date=2026-04-29", "must be provided together"),
        ("?start_date=2026-04-29&end_date=2026-04-01", "on or before end_date"),
        ("?start_date=not-a-date&end_date=2026-04-29", "YYYY-MM-DD"),
    ],
)
@pytest.mark.asyncio
async def test_list_tags_rejects_invalid_date_range(query, expected_detail_fragment):
    from unittest.mock import AsyncMock, Mock

    from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

    mock_user_auth = UserAPIKeyAuth(
        user_id="test-user-123",
        user_role=LitellmUserRoles.PROXY_ADMIN,
    )
    app.dependency_overrides[user_api_key_auth] = lambda: mock_user_auth

    try:
        with patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma:
            mock_db = Mock()
            mock_prisma.db = mock_db
            mock_db.litellm_tagtable.find_many = AsyncMock(return_value=[])
            mock_db.litellm_dailytagspend.group_by = AsyncMock(return_value=[])

            headers = {"Authorization": "Bearer sk-1234"}
            response = client.get(f"/tag/list{query}", headers=headers)

            assert response.status_code == 400
            assert expected_detail_fragment in response.json()["detail"]

    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_get_deployments_by_model_id():
    """
    Test get_deployments_by_model when model is found by model_id
    """
    from unittest.mock import Mock

    from litellm.proxy.management_endpoints.tag_management_endpoints import (
        get_deployments_by_model,
    )
    from litellm.types.router import Deployment, LiteLLM_Params, ModelInfo

    # Create a mock router
    mock_router = Mock()

    # Setup mock to return deployment by model_id
    mock_deployment = Deployment(
        model_name="gpt-3.5-turbo",
        litellm_params=LiteLLM_Params(model="gpt-3.5-turbo"),
        model_info=ModelInfo(),
    )
    mock_router.get_deployment.return_value = mock_deployment

    result = await get_deployments_by_model("model-123", mock_router)

    assert len(result) == 1
    assert result[0] == mock_deployment
    mock_router.get_deployment.assert_called_once_with(model_id="model-123")


@pytest.mark.asyncio
async def test_get_deployments_by_model_name():
    """
    Test get_deployments_by_model when model is found by model_name
    """
    from unittest.mock import Mock

    from litellm.proxy.management_endpoints.tag_management_endpoints import (
        get_deployments_by_model,
    )
    from litellm.types.router import Deployment

    # Create a mock router
    mock_router = Mock()

    # Setup mock to not find by model_id but find by model_name
    mock_router.get_deployment.return_value = None
    mock_router.get_model_list.return_value = [
        {
            "model_name": "gpt-3.5-turbo",
            "litellm_params": {"model": "gpt-3.5-turbo", "api_key": "test-key"},
            "model_info": {"id": "model-1", "description": "Test model"},
        }
    ]

    result = await get_deployments_by_model("gpt-3.5-turbo", mock_router)

    assert len(result) == 1
    assert result[0].model_name == "gpt-3.5-turbo"
    assert isinstance(result[0], Deployment)
    mock_router.get_deployment.assert_called_once_with(model_id="gpt-3.5-turbo")
    mock_router.get_model_list.assert_called_once_with(model_name="gpt-3.5-turbo")


@pytest.mark.asyncio
async def test_get_deployments_by_model_not_found():
    """
    Test get_deployments_by_model when model is not found
    """
    from unittest.mock import Mock

    from litellm.proxy.management_endpoints.tag_management_endpoints import (
        get_deployments_by_model,
    )

    # Create a mock router
    mock_router = Mock()

    # Setup mock to not find model by either method
    mock_router.get_deployment.return_value = None
    mock_router.get_model_list.return_value = None

    result = await get_deployments_by_model("nonexistent-model", mock_router)

    assert len(result) == 0
    assert result == []
    mock_router.get_deployment.assert_called_once_with(model_id="nonexistent-model")
    mock_router.get_model_list.assert_called_once_with(model_name="nonexistent-model")


@pytest.mark.asyncio
async def test_add_tag_to_deployment_preserves_encrypted_fields():
    """
    Test that _add_tag_to_deployment preserves encrypted fields when adding tags
    """
    from unittest.mock import AsyncMock, Mock

    from litellm.proxy.management_endpoints.tag_management_endpoints import (
        _add_tag_to_deployment,
    )
    from litellm.types.router import Deployment, LiteLLM_Params, ModelInfo

    with patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma:
        # Setup prisma mocks
        mock_db = Mock()
        mock_prisma.db = mock_db

        # Mock the database model with encrypted fields
        db_model = Mock()
        db_model.model_id = "model-123"
        db_model.litellm_params = {
            "model": "gpt-3.5-turbo",
            "api_key": "encrypted_api_key_value",  # This should be preserved
            "api_base": "https://api.openai.com",
            "other_encrypted_field": "encrypted_value",
        }

        # Mock find_unique to return the db model
        mock_db.litellm_proxymodeltable.find_unique = AsyncMock(return_value=db_model)

        # Mock update
        mock_db.litellm_proxymodeltable.update = AsyncMock(return_value=db_model)

        # Create deployment
        deployment = Deployment(
            model_name="gpt-3.5-turbo",
            litellm_params=LiteLLM_Params(model="gpt-3.5-turbo"),
            model_info=ModelInfo(id="model-123"),
        )

        # Call the function
        await _add_tag_to_deployment(deployment, "test-tag")

        # Verify find_unique was called
        mock_db.litellm_proxymodeltable.find_unique.assert_called_once_with(
            where={"model_id": "model-123"}
        )

        # Verify update was called with preserved encrypted fields
        update_call = mock_db.litellm_proxymodeltable.update.call_args
        assert update_call[1]["where"] == {"model_id": "model-123"}

        # Parse the updated litellm_params
        updated_params = json.loads(update_call[1]["data"]["litellm_params"])

        # Verify tag was added
        assert "tags" in updated_params
        assert "test-tag" in updated_params["tags"]

        # Verify encrypted fields were preserved
        assert updated_params["api_key"] == "encrypted_api_key_value"
        assert updated_params["other_encrypted_field"] == "encrypted_value"
        assert updated_params["model"] == "gpt-3.5-turbo"
        assert updated_params["api_base"] == "https://api.openai.com"


@pytest.mark.asyncio
async def test_add_tag_to_deployment_with_string_params():
    """
    Test that _add_tag_to_deployment handles string litellm_params correctly
    """
    from unittest.mock import AsyncMock, Mock

    from litellm.proxy.management_endpoints.tag_management_endpoints import (
        _add_tag_to_deployment,
    )
    from litellm.types.router import Deployment, LiteLLM_Params, ModelInfo

    with patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma:
        # Setup prisma mocks
        mock_db = Mock()
        mock_prisma.db = mock_db

        # Mock the database model with litellm_params as string
        db_model = Mock()
        db_model.model_id = "model-456"
        db_model.litellm_params = json.dumps(
            {
                "model": "claude-3",
                "api_key": "encrypted_claude_key",
            }
        )

        # Mock find_unique to return the db model
        mock_db.litellm_proxymodeltable.find_unique = AsyncMock(return_value=db_model)

        # Mock update
        mock_db.litellm_proxymodeltable.update = AsyncMock(return_value=db_model)

        # Create deployment
        deployment = Deployment(
            model_name="claude-3",
            litellm_params=LiteLLM_Params(model="claude-3"),
            model_info=ModelInfo(id="model-456"),
        )

        # Call the function
        await _add_tag_to_deployment(deployment, "test-tag-2")

        # Verify update was called
        update_call = mock_db.litellm_proxymodeltable.update.call_args
        updated_params = json.loads(update_call[1]["data"]["litellm_params"])

        # Verify tag was added and encrypted field preserved
        assert "tags" in updated_params
        assert "test-tag-2" in updated_params["tags"]
        assert updated_params["api_key"] == "encrypted_claude_key"


@pytest.mark.asyncio
async def test_add_tag_to_deployment_no_duplicate_tags():
    """
    Test that _add_tag_to_deployment doesn't add duplicate tags
    """
    from unittest.mock import AsyncMock, Mock

    from litellm.proxy.management_endpoints.tag_management_endpoints import (
        _add_tag_to_deployment,
    )
    from litellm.types.router import Deployment, LiteLLM_Params, ModelInfo

    with patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma:
        # Setup prisma mocks
        mock_db = Mock()
        mock_prisma.db = mock_db

        # Mock the database model with existing tags
        db_model = Mock()
        db_model.model_id = "model-789"
        db_model.litellm_params = {
            "model": "gpt-4",
            "api_key": "encrypted_key",
            "tags": ["existing-tag", "another-tag"],
        }

        # Mock find_unique to return the db model
        mock_db.litellm_proxymodeltable.find_unique = AsyncMock(return_value=db_model)

        # Mock update
        mock_db.litellm_proxymodeltable.update = AsyncMock(return_value=db_model)

        # Create deployment
        deployment = Deployment(
            model_name="gpt-4",
            litellm_params=LiteLLM_Params(model="gpt-4"),
            model_info=ModelInfo(id="model-789"),
        )

        # Try to add an existing tag
        await _add_tag_to_deployment(deployment, "existing-tag")

        # Verify update was called
        update_call = mock_db.litellm_proxymodeltable.update.call_args
        updated_params = json.loads(update_call[1]["data"]["litellm_params"])

        # Verify no duplicate tags
        assert updated_params["tags"].count("existing-tag") == 1
        assert len(updated_params["tags"]) == 2
        assert "another-tag" in updated_params["tags"]


@pytest.mark.asyncio
async def test_add_tag_to_deployment_model_not_found():
    """
    Test that _add_tag_to_deployment raises HTTPException when model not found
    """
    from unittest.mock import AsyncMock, Mock

    from litellm.proxy.management_endpoints.tag_management_endpoints import (
        _add_tag_to_deployment,
    )
    from litellm.types.router import Deployment, LiteLLM_Params, ModelInfo

    with patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma:
        # Setup prisma mocks
        mock_db = Mock()
        mock_prisma.db = mock_db

        # Mock find_unique to return None (model not found)
        mock_db.litellm_proxymodeltable.find_unique = AsyncMock(return_value=None)

        # Create deployment
        deployment = Deployment(
            model_name="nonexistent-model",
            litellm_params=LiteLLM_Params(model="nonexistent-model"),
            model_info=ModelInfo(id="model-999"),
        )

        # Call should raise HTTPException (wrapped as 500 by the exception handler)
        with pytest.raises(HTTPException) as exc_info:
            await _add_tag_to_deployment(deployment, "test-tag")

        assert exc_info.value.status_code == 500
        assert "not found in database" in str(exc_info.value.detail)
