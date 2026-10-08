import inspect
import json
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime
from types import MappingProxyType, SimpleNamespace
from typing import Final, cast
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from prisma.actions import LiteLLM_VerificationTokenActions

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.proxy_server import app
from litellm.proxy.utils import PrismaClient
from litellm.types.tag_management import TagNewRequest

client = TestClient(app)


class _BudgetState:
    def __init__(self, values: Mapping[str, object]) -> None:
        self._values: Mapping[str, object] = MappingProxyType(dict(values))

    def store(self, values: Mapping[str, object]) -> None:
        self._values = MappingProxyType({**self._values, **values})

    def get(self, field: str) -> object:
        return self._values[field]

    def row(self) -> SimpleNamespace:
        return SimpleNamespace(**self._values)


class FakeVerificationTokenTable:
    """Stand-in for ``prisma_client.db.litellm_verificationtoken``.

    ``AsyncMock`` swallows any keyword argument, so a plain mock cannot catch a
    call that the generated prisma client would reject at runtime. This double
    binds every call against the real ``find_many`` signature, so passing an
    unsupported kwarg (e.g. ``select``) raises the same ``TypeError`` the proxy
    surfaces as an HTTP 500.
    """

    def __init__(self, records: Sequence[Mock]):
        self._records = tuple(records)
        self.calls: list[dict[str, object]] = []

    async def find_many(self, **kwargs: object) -> tuple[Mock, ...]:
        inspect.signature(LiteLLM_VerificationTokenActions.find_many).bind(
            self, **kwargs
        )
        self.calls.append(kwargs)
        return self._records


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
            patch("litellm.proxy.proxy_server.llm_router"),
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

            headers = {"Authorization": "Bearer sk-9876"}

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

            headers = {"Authorization": "Bearer sk-9876"}

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
async def test_new_tag_persists_a_budget():
    from datetime import datetime

    from litellm.proxy.management_endpoints.tag_management_endpoints import new_tag

    budget_state = _BudgetState({"budget_id": "budget-1", "max_budget": None})
    created_tag = SimpleNamespace(
        tag_name="budget-tag",
        description=None,
        models=[],
        created_at=datetime(2024, 1, 1),
        updated_at=datetime(2024, 1, 1),
        created_by="admin",
    )
    mock_db = Mock()
    mock_prisma = SimpleNamespace(db=mock_db, jsonify_object=lambda data: dict(data))
    mock_db.litellm_tagtable.find_unique = AsyncMock(return_value=None)
    mock_db.litellm_proxymodeltable.find_many = AsyncMock(return_value=[])

    async def create_budget(data, **_):
        budget_state.store(data)
        return budget_state.row()

    async def create_tag(data, **_):
        created_tag.budget_id = data["budget_id"]
        return created_tag

    mock_db.litellm_budgettable.create = create_budget
    mock_db.litellm_tagtable.create = create_tag
    with (
        patch(  # test-quality-ok: endpoint resolves the fake database through proxy_server
            "litellm.proxy.proxy_server.prisma_client", mock_prisma
        ),
        patch(  # test-quality-ok: endpoint reads the audit actor from proxy_server
            "litellm.proxy.proxy_server.litellm_proxy_admin_name", "admin"
        ),
        patch(  # test-quality-ok: endpoint requires a router before the budget write
            "litellm.proxy.proxy_server.llm_router", object()
        ),
        patch(  # test-quality-ok: cache invalidation is outside this budget contract
            "litellm.proxy.management_endpoints.tag_management_endpoints._evict_tag_cache_keys", new=AsyncMock()
        ),
    ):
        await new_tag(
            tag=TagNewRequest(name="budget-tag", max_budget=25.0),
            user_api_key_dict=UserAPIKeyAuth(user_id="admin", user_role=LitellmUserRoles.PROXY_ADMIN),
        )

    assert budget_state.get("max_budget") == 25.0
    assert created_tag.budget_id == "budget-1"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("budget_fields", "should_update", "expected_max_budget"),
    [
        ({"max_budget": None}, True, None),
        ({}, False, None),
        ({"max_budget": 0}, True, 0.0),
    ],
)
async def test_update_tag_clears_or_sets_only_provided_budget_fields(
    budget_fields: Mapping[str, object],
    should_update: bool,
    expected_max_budget: float | None,
) -> None:
    from datetime import datetime

    from litellm.proxy.management_endpoints.tag_management_endpoints import update_tag
    from litellm.types.tag_management import TagUpdateRequest

    existing_tag: Final = SimpleNamespace(budget_id="budget-1")
    updated_tag: Final = SimpleNamespace(
        tag_name="budget-tag",
        description=None,
        models=[],
        created_at=datetime(2024, 1, 1),
        updated_at=datetime(2024, 1, 1),
        created_by="admin",
    )
    find_tag: Final = AsyncMock(return_value=existing_tag)
    find_models: Final = AsyncMock(return_value=[])
    update_tag_row: Final = AsyncMock(return_value=updated_tag)
    update_budget: Final = AsyncMock()
    mock_prisma: Final = cast(
        PrismaClient,
        SimpleNamespace(
            db=SimpleNamespace(
                litellm_tagtable=SimpleNamespace(find_unique=find_tag, update=update_tag_row),
                litellm_proxymodeltable=SimpleNamespace(find_many=find_models),
                litellm_budgettable=SimpleNamespace(update=update_budget),
            )
        ),
    )

    with (
        patch(  # test-quality-ok: endpoint resolves the fake database through proxy_server
            "litellm.proxy.proxy_server.prisma_client", mock_prisma
        ),
        patch(  # test-quality-ok: endpoint reads the audit actor from proxy_server
            "litellm.proxy.proxy_server.litellm_proxy_admin_name", "admin"
        ),
        patch(  # test-quality-ok: cache invalidation is outside this budget contract
            "litellm.proxy.management_endpoints.tag_management_endpoints._evict_tag_cache_keys", new=AsyncMock()
        ),
    ):
        await update_tag(
            tag=TagUpdateRequest.model_validate({"name": "budget-tag", **budget_fields}),
            user_api_key_dict=UserAPIKeyAuth(user_id="admin", user_role=LitellmUserRoles.PROXY_ADMIN),
        )

    if not should_update:
        update_budget.assert_not_awaited()
        return

    update_args: Final = update_budget.await_args
    assert update_args is not None
    budget_data: Final = cast(Mapping[str, object], update_args.kwargs["data"])
    assert "max_budget" in budget_data
    assert budget_data["max_budget"] == expected_max_budget
    assert not budget_data.keys() & {
        "soft_budget",
        "max_parallel_requests",
        "tpm_limit",
        "rpm_limit",
        "model_max_budget",
        "budget_duration",
    }


@pytest.mark.asyncio
async def test_update_tag_explicit_null_clears_budget_duration():
    from datetime import datetime

    from litellm.proxy.management_endpoints.tag_management_endpoints import update_tag
    from litellm.types.tag_management import TagUpdateRequest

    budget_state = _BudgetState({"budget_id": "budget-1", "budget_duration": "30d"})
    existing_tag = SimpleNamespace(budget_id="budget-1")
    updated_tag = SimpleNamespace(
        tag_name="budget-tag",
        description=None,
        models=[],
        created_at=datetime(2024, 1, 1),
        updated_at=datetime(2024, 1, 1),
        created_by="admin",
    )
    mock_db = Mock()
    mock_prisma = SimpleNamespace(db=mock_db)
    mock_db.litellm_tagtable.find_unique = AsyncMock(return_value=existing_tag)
    mock_db.litellm_proxymodeltable.find_many = AsyncMock(return_value=[])
    mock_db.litellm_tagtable.update = AsyncMock(return_value=updated_tag)

    async def update_budget(where, data, **_):
        budget_state.store(data)
        return budget_state.row()

    mock_db.litellm_budgettable.update = update_budget
    with (
        patch(  # test-quality-ok: endpoint resolves the fake database through proxy_server
            "litellm.proxy.proxy_server.prisma_client", mock_prisma
        ),
        patch(  # test-quality-ok: endpoint reads the audit actor from proxy_server
            "litellm.proxy.proxy_server.litellm_proxy_admin_name", "admin"
        ),
        patch(  # test-quality-ok: cache invalidation is outside this budget contract
            "litellm.proxy.management_endpoints.tag_management_endpoints._evict_tag_cache_keys", new=AsyncMock()
        ),
    ):
        await update_tag(
            tag=TagUpdateRequest(name="budget-tag", budget_duration=None),
            user_api_key_dict=UserAPIKeyAuth(user_id="admin", user_role=LitellmUserRoles.PROXY_ADMIN),
        )

    assert budget_state.get("budget_duration") is None


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

            headers = {"Authorization": "Bearer sk-9876"}

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


class _RecordingAuthCache:
    """Captures the keys an endpoint evicts, so tests assert on cache keys not mock plumbing."""

    def __init__(self):
        self.deleted: list[str] = []

    async def async_delete_cache(self, key: str) -> None:
        self.deleted.append(key)


@contextmanager
def _tag_cache_doubles():
    """Swaps in the auth cache and the cross-worker publisher a tag mutation is expected to hit."""
    recording_cache = _RecordingAuthCache()
    mock_publish = AsyncMock()
    with (
        patch("litellm.proxy.proxy_server.user_api_key_cache", recording_cache),
        patch(
            "litellm.proxy.common_utils.auth_cache_invalidation_pubsub.publish_auth_cache_invalidation",
            mock_publish,
        ),
    ):
        yield recording_cache, mock_publish


def _published_keys(mock_publish) -> list[str]:
    return [call.kwargs["cache_key"] for call in mock_publish.call_args_list]


@pytest.mark.asyncio
async def test_new_tag_invalidates_tag_and_registry_caches():
    """
    A tag created on one worker must be visible to every worker's auth path immediately.

    Auth serves tags cache-first, and the cached tag-name registry is what decides whether a
    request tag is looked up at all, so a create that leaves both entries stale means the new
    tag's budget goes unenforced until the TTL expires.
    """
    from datetime import datetime
    from unittest.mock import AsyncMock, Mock

    from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

    app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_id="test-user-123",
        user_role=LitellmUserRoles.PROXY_ADMIN,
    )

    try:
        with (
            _tag_cache_doubles() as (recording_cache, mock_publish),
            patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma,
            patch("litellm.proxy.proxy_server.llm_router"),
            patch(
                "litellm.proxy.proxy_server.litellm_proxy_admin_name", "default_user_id"
            ),
            patch(
                "litellm.proxy.management_endpoints.tag_management_endpoints.get_deployments_by_model"
            ) as mock_get_deployments,
        ):
            mock_db = Mock()
            mock_prisma.db = mock_db
            mock_db.litellm_tagtable.find_unique = AsyncMock(return_value=None)
            mock_db.litellm_proxymodeltable.find_many = AsyncMock(return_value=[])
            mock_get_deployments.return_value = []

            created_tag = Mock()
            created_tag.tag_name = "cache-tag"
            created_tag.description = None
            created_tag.models = []
            created_tag.model_info = {}
            created_tag.spend = 0.0
            created_tag.budget_id = None
            created_tag.created_at = datetime.now()
            created_tag.updated_at = datetime.now()
            created_tag.created_by = "test-user-123"
            mock_db.litellm_tagtable.create = AsyncMock(return_value=created_tag)

            response = client.post(
                "/tag/new",
                json={"name": "cache-tag"},
                headers={"Authorization": "Bearer sk-9876"},
            )
            assert response.status_code == 200

            assert recording_cache.deleted == ["tag:cache-tag", "tag_registry"]
            assert _published_keys(mock_publish) == ["tag:cache-tag", "tag_registry"]
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_update_tag_invalidates_only_the_tag_cache():
    """An update can change the tag's budget but never the set of names, so the registry stands."""
    from datetime import datetime
    from unittest.mock import AsyncMock, Mock

    from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

    app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_id="test-user-123",
        user_role=LitellmUserRoles.PROXY_ADMIN,
    )

    try:
        with (
            _tag_cache_doubles() as (recording_cache, mock_publish),
            patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma,
            patch(
                "litellm.proxy.proxy_server.litellm_proxy_admin_name", "default_user_id"
            ),
        ):
            mock_db = Mock()
            mock_prisma.db = mock_db

            existing_tag = Mock()
            existing_tag.tag_name = "cache-tag"
            existing_tag.budget_id = None
            mock_db.litellm_tagtable.find_unique = AsyncMock(return_value=existing_tag)
            mock_db.litellm_proxymodeltable.find_many = AsyncMock(return_value=[])

            updated_tag = Mock()
            updated_tag.tag_name = "cache-tag"
            updated_tag.description = "updated"
            updated_tag.models = []
            updated_tag.model_info = {}
            updated_tag.spend = 0.0
            updated_tag.budget_id = None
            updated_tag.created_at = datetime.now()
            updated_tag.updated_at = datetime.now()
            updated_tag.created_by = "test-user-123"
            mock_db.litellm_tagtable.update = AsyncMock(return_value=updated_tag)

            response = client.post(
                "/tag/update",
                json={"name": "cache-tag", "description": "updated"},
                headers={"Authorization": "Bearer sk-9876"},
            )
            assert response.status_code == 200

            assert recording_cache.deleted == ["tag:cache-tag"]
            assert _published_keys(mock_publish) == ["tag:cache-tag"]
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_delete_tag_invalidates_tag_and_registry_caches():
    """Without this a deleted tag keeps its cached budget enforced until the TTL expires."""
    from unittest.mock import AsyncMock, Mock

    from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

    app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_id="test-user-123",
        user_role=LitellmUserRoles.PROXY_ADMIN,
    )

    try:
        with (
            _tag_cache_doubles() as (recording_cache, mock_publish),
            patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma,
        ):
            mock_db = Mock()
            mock_prisma.db = mock_db

            existing_tag = Mock()
            existing_tag.tag_name = "cache-tag"
            mock_db.litellm_tagtable.find_unique = AsyncMock(return_value=existing_tag)
            mock_db.litellm_tagtable.delete = AsyncMock(return_value=existing_tag)

            response = client.post(
                "/tag/delete",
                json={"name": "cache-tag"},
                headers={"Authorization": "Bearer sk-9876"},
            )
            assert response.status_code == 200

            assert recording_cache.deleted == ["tag:cache-tag", "tag_registry"]
            assert _published_keys(mock_publish) == ["tag:cache-tag", "tag_registry"]
    finally:
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

            headers = {"Authorization": "Bearer sk-9876"}
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

            headers = {"Authorization": "Bearer sk-9876"}
            response = client.get("/tag/list", headers=headers)

            assert response.status_code == 200
            result = response.json()
            assert len(result) == 1
            assert result[0]["name"] == "stored-tag"

    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
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
            fake_token_table = FakeVerificationTokenTable([owned_key_record])
            mock_db.litellm_verificationtoken = fake_token_table

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
            assert fake_token_table.calls == [
                {"where": {"user_id": "internal-user-123"}}
            ]
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
async def test_internal_user_list_tags_does_not_500_on_unsupported_prisma_kwarg():
    """
    Regression: /tag/list returned 500 for every internal user because the
    non-admin branch looked up the caller's keys with
    ``find_many(select={"token": True})``, and the generated prisma client has no
    ``select`` kwarg. This reproduces the reported case exactly: a freshly created
    internal user with no tag spend yet, which must get an empty 200 rather than
    "LiteLLM_VerificationTokenActions.find_many() got an unexpected keyword
    argument 'select'".
    """
    from unittest.mock import AsyncMock, Mock

    from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

    mock_user_auth = UserAPIKeyAuth(
        api_key="new-user-key",
        user_id="brand-new-internal-user",
        user_role=LitellmUserRoles.INTERNAL_USER,
    )
    app.dependency_overrides[user_api_key_auth] = lambda: mock_user_auth

    try:
        with patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma:
            mock_db = Mock()
            mock_prisma.db = mock_db

            key_record = Mock()
            key_record.token = "new-user-key"
            fake_token_table = FakeVerificationTokenTable([key_record])
            mock_db.litellm_verificationtoken = fake_token_table

            mock_db.litellm_dailytagspend.group_by = AsyncMock(return_value=[])
            mock_db.litellm_tagtable.find_many = AsyncMock(return_value=[])

            response = client.get(
                "/tag/list", headers={"Authorization": "Bearer new-user-key"}
            )

            assert response.status_code == 200, response.text
            assert response.json() == []
            assert fake_token_table.calls == [
                {"where": {"user_id": "brand-new-internal-user"}}
            ]
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

            headers = {"Authorization": "Bearer sk-9876"}
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


class _FakeDailySpendTable:
    """Records the kwargs prisma would receive and replays canned rows."""

    def __init__(self, rows: Sequence[object] = ()):
        self._rows = list(rows)
        self.calls: list[dict[str, object]] = []

    async def count(self, **kwargs: object) -> int:
        self.calls.append({"method": "count", **kwargs})
        return len(self._rows)

    async def find_many(self, **kwargs: object) -> list[object]:
        self.calls.append({"method": "find_many", **kwargs})
        return list(self._rows)

    def where_clauses(self) -> list[Mapping[str, object]]:
        return [call["where"] for call in self.calls if "where" in call]


def _daily_tag_spend_row(
    *, spend: float, tag: str | None, team_id: str, api_key: str = ""
) -> SimpleNamespace:
    return SimpleNamespace(
        date="2026-06-01",
        api_key=api_key,
        model="model-a",
        model_group=None,
        custom_llm_provider="provider-a",
        mcp_namespaced_tool_name=None,
        endpoint=None,
        prompt_tokens=1,
        completion_tokens=1,
        spend=spend,
        cache_read_input_tokens=0,
        cache_creation_input_tokens=0,
        compression_saved_tokens=0,
        compression_savings_spend=0.0,
        prompt_caching_savings_spend=0.0,
        gateway_injected_caching_savings_spend=0.0,
        autorouter_savings_spend=0.0,
        api_requests=1,
        successful_requests=1,
        failed_requests=0,
        total_response_time_ms=0,
        timed_requests=0,
        ptu_flat_cost=0.0,
        request_id=None,
        tag=tag,
        team_id=team_id,
    )


def _tag_activity_prisma(
    *, tag_table: _FakeDailySpendTable, token_records: Sequence[Mock] = (), team_rows: Sequence[object] = ()
) -> SimpleNamespace:
    return SimpleNamespace(
        db=SimpleNamespace(
            litellm_dailytagspend=tag_table,
            litellm_verificationtoken=FakeVerificationTokenTable(token_records),
            litellm_teamtable=SimpleNamespace(find_many=AsyncMock(return_value=list(team_rows))),
        )
    )


@pytest.mark.asyncio
async def test_internal_user_tag_daily_activity_is_scoped_to_their_keys():
    """
    Internal users must not receive proxy-wide tag spend rows when viewing tag
    usage daily activity.
    """
    from unittest.mock import Mock

    from litellm.proxy.management_endpoints.tag_management_endpoints import (
        get_tag_daily_activity,
    )

    mock_user_auth = UserAPIKeyAuth(
        user_id="internal-user-123",
        user_role=LitellmUserRoles.INTERNAL_USER_VIEW_ONLY,
    )

    owned_key_record = Mock()
    owned_key_record.token = "owned-key"
    tag_table = _FakeDailySpendTable()
    mock_prisma = _tag_activity_prisma(tag_table=tag_table, token_records=[owned_key_record])

    with patch("litellm.proxy.proxy_server.prisma_client", mock_prisma):
        result = await get_tag_daily_activity(
            start_date="2025-01-01",
            end_date="2025-01-31",
            user_api_key_dict=mock_user_auth,
        )

    assert result.results == []
    where = tag_table.where_clauses()[0]
    assert where["api_key"] == {"in": ["owned-key"]}


@pytest.mark.asyncio
async def test_internal_user_tag_daily_activity_rejects_unowned_api_key_filter():
    """
    If an internal user filters tag usage by an API key they do not own, the
    endpoint should return an empty scoped filter instead of exposing that key's
    tag spend.
    """
    from unittest.mock import Mock

    from litellm.proxy.management_endpoints.tag_management_endpoints import (
        get_tag_daily_activity,
    )

    mock_user_auth = UserAPIKeyAuth(
        user_id="internal-user-123",
        user_role=LitellmUserRoles.INTERNAL_USER,
    )

    owned_key_record = Mock()
    owned_key_record.token = "owned-key"
    tag_table = _FakeDailySpendTable()
    mock_prisma = _tag_activity_prisma(tag_table=tag_table, token_records=[owned_key_record])

    with patch("litellm.proxy.proxy_server.prisma_client", mock_prisma):
        result = await get_tag_daily_activity(
            start_date="2025-01-01",
            end_date="2025-01-31",
            api_key="unowned-key",
            user_api_key_dict=mock_user_auth,
        )

    assert result.results == []
    assert result.metadata.total_spend == 0
    assert result.metadata.total_api_requests == 0
    assert tag_table.calls == []


@pytest.mark.asyncio
async def test_internal_user_tag_daily_activity_scopes_to_current_key_without_user_id():
    """
    If an internal-user token has no user_id, it should still scope tag usage to
    the current request key instead of falling back to proxy-wide tag spend.
    """
    from litellm.proxy.management_endpoints.tag_management_endpoints import (
        get_tag_daily_activity,
    )

    mock_user_auth = UserAPIKeyAuth(
        api_key="current-owned-key",
        user_id=None,
        user_role=LitellmUserRoles.INTERNAL_USER,
    )

    tag_table = _FakeDailySpendTable()
    mock_prisma = _tag_activity_prisma(tag_table=tag_table)

    with patch("litellm.proxy.proxy_server.prisma_client", mock_prisma):
        result = await get_tag_daily_activity(
            start_date="2025-01-01",
            end_date="2025-01-31",
            user_api_key_dict=mock_user_auth,
        )

    assert result.results == []
    where = tag_table.where_clauses()[0]
    assert where["api_key"] == {"in": ["current-owned-key"]}


@pytest.mark.asyncio
async def test_internal_user_tag_daily_activity_without_any_scoped_keys_returns_empty():
    """
    If an internal-user token has neither user_id nor api_key, the endpoint must
    return an empty response instead of dropping the API key filter.
    """
    from litellm.proxy.management_endpoints.tag_management_endpoints import (
        get_tag_daily_activity,
    )

    mock_user_auth = UserAPIKeyAuth(
        user_id=None,
        user_role=LitellmUserRoles.INTERNAL_USER,
    )

    tag_table = _FakeDailySpendTable()
    mock_prisma = _tag_activity_prisma(tag_table=tag_table)

    with patch("litellm.proxy.proxy_server.prisma_client", mock_prisma):
        result = await get_tag_daily_activity(
            start_date="2025-01-01",
            end_date="2025-01-31",
            user_api_key_dict=mock_user_auth,
        )

    assert result.results == []
    assert result.metadata.total_spend == 0
    assert result.metadata.total_api_requests == 0
    assert tag_table.calls == []


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

            headers = {"Authorization": "Bearer sk-9876"}
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

            headers = {"Authorization": "Bearer sk-9876"}
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


class _StoredTagTable:
    def __init__(self, model_info: object) -> None:
        self.model_info = model_info

    async def find_many(self, where: object = None, include: object = None) -> list[SimpleNamespace]:
        return [
            SimpleNamespace(
                tag_name="routed-tag",
                description="Routes to one model",
                models=["model-1"],
                model_info=self.model_info,
                budget_id=None,
                created_at=datetime(2025, 1, 1),
                updated_at=datetime(2025, 1, 2),
                created_by="user-123",
                litellm_budget_table=None,
            )
        ]


class _NoDynamicTagSpend:
    async def group_by(self, by: object, where: object, min: object, max: object) -> list[object]:
        return []


@pytest.mark.parametrize(
    ("stored_model_info", "returned_model_info"),
    [
        ('{"model-1": "gpt-4o"}', {"model-1": "gpt-4o"}),
        ({"model-1": "gpt-4o"}, {"model-1": "gpt-4o"}),
        (None, {}),
    ],
)
def test_tag_info_and_tag_list_return_the_stored_model_info_decoded(
    monkeypatch, stored_model_info, returned_model_info
):
    from litellm.proxy import proxy_server
    from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

    monkeypatch.setattr(
        proxy_server,
        "prisma_client",
        SimpleNamespace(
            db=SimpleNamespace(
                litellm_tagtable=_StoredTagTable(stored_model_info),
                litellm_dailytagspend=_NoDynamicTagSpend(),
            )
        ),
    )
    app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_id="test-user-123", user_role=LitellmUserRoles.PROXY_ADMIN
    )
    try:
        info_response = client.post("/tag/info", json={"names": ["routed-tag"]})
        list_response = client.get("/tag/list")
    finally:
        app.dependency_overrides.clear()

    assert info_response.status_code == 200
    assert info_response.json()["routed-tag"]["model_info"] == returned_model_info
    assert list_response.status_code == 200
    assert [tag["model_info"] for tag in list_response.json()] == [returned_model_info]


def _team_row_without_member_view(team_id: str, user_id: str):
    """A team the user belongs to without the /team/daily/activity permission."""
    team = Mock(spec=["team_id", "team_alias", "members_with_roles", "team_member_permissions", "model_dump"])
    team.team_id = team_id
    team.team_alias = f"Alias {team_id}"
    team.model_dump.return_value = {
        "team_id": team_id,
        "team_alias": f"Alias {team_id}",
        "members_with_roles": [{"user_id": user_id, "role": "user"}],
        "team_member_permissions": [],
    }
    return team


@pytest.mark.asyncio
async def test_tag_daily_activity_team_grouping_filters_team_and_tag():
    """
    /tag/daily/activity?team_ids=...&tags=...&group_by=team must read the tag
    spend table filtered to both the permitted teams and the requested tags,
    with results bucketed by team_id carrying the team alias metadata.
    """
    from litellm.proxy.management_endpoints.tag_management_endpoints import (
        get_tag_daily_activity,
    )

    tag_table = _FakeDailySpendTable([_daily_tag_spend_row(spend=5.0, tag="shared", team_id="team-a")])
    team_row = SimpleNamespace(team_id="team-a", team_alias="Team A")
    mock_prisma = _tag_activity_prisma(tag_table=tag_table, team_rows=[team_row])

    with patch("litellm.proxy.proxy_server.prisma_client", mock_prisma):
        result = await get_tag_daily_activity(
            tags="shared",
            team_ids="team-a",
            group_by="team",
            start_date="2026-06-01",
            end_date="2026-06-02",
            user_api_key_dict=UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN),
        )

    assert len(tag_table.where_clauses()) == 2
    where = tag_table.where_clauses()[0]
    assert {"team_id": {"in": ["team-a"]}} in where["AND"]
    assert {"tag": {"in": ["shared"]}} in where["AND"]
    entity = result.results[0].breakdown.entities["team-a"]
    assert entity.metadata == {"team_alias": "Team A"}
    assert entity.metrics.spend == 5.0


@pytest.mark.asyncio
async def test_tag_daily_activity_team_ids_denied_for_non_member():
    """
    An internal user asking for a team they are not a member of must get a 404,
    the same contract /team/daily/activity enforces.
    """
    from litellm.proxy._types import LiteLLM_UserTable
    from litellm.proxy.management_endpoints.tag_management_endpoints import (
        get_tag_daily_activity,
    )

    tag_table = _FakeDailySpendTable()
    mock_prisma = _tag_activity_prisma(tag_table=tag_table)
    member = LiteLLM_UserTable(
        user_id="internal-user-1",
        teams=["team-b"],
        max_budget=None,
        spend=0.0,
        user_email=None,
        user_role="internal_user",
    )

    with (
        patch("litellm.proxy.proxy_server.prisma_client", mock_prisma),
        patch(
            "litellm.proxy.management_endpoints.team_endpoints.get_user_object",
            new=AsyncMock(return_value=member),
        ),
        pytest.raises(HTTPException) as exc_info,
    ):
        await get_tag_daily_activity(
            team_ids="team-a",
            start_date="2026-06-01",
            end_date="2026-06-02",
            user_api_key_dict=UserAPIKeyAuth(
                user_id="internal-user-1", user_role=LitellmUserRoles.INTERNAL_USER
            ),
        )

    assert exc_info.value.status_code == 404
    assert tag_table.calls == []


@pytest.mark.asyncio
async def test_tag_daily_activity_member_without_team_view_scoped_to_own_keys():
    """
    A member of a team who lacks the /team/daily/activity permission and is not
    a team admin only sees usage produced by their own API keys.
    """
    from litellm.proxy._types import LiteLLM_UserTable
    from litellm.proxy.management_endpoints.tag_management_endpoints import (
        get_tag_daily_activity,
    )

    owned_key_record = Mock()
    owned_key_record.token = "key-owned-2"
    tag_table = _FakeDailySpendTable()
    mock_prisma = _tag_activity_prisma(
        tag_table=tag_table,
        token_records=[owned_key_record],
        team_rows=[_team_row_without_member_view("team-a", "member-1")],
    )
    member = LiteLLM_UserTable(
        user_id="member-1",
        teams=["team-a"],
        max_budget=None,
        spend=0.0,
        user_email=None,
        user_role="internal_user",
    )

    with (
        patch("litellm.proxy.proxy_server.prisma_client", mock_prisma),
        patch(
            "litellm.proxy.management_endpoints.team_endpoints.get_user_object",
            new=AsyncMock(return_value=member),
        ),
    ):
        await get_tag_daily_activity(
            tags="shared",
            team_ids="team-a",
            group_by="team",
            start_date="2026-06-01",
            end_date="2026-06-02",
            user_api_key_dict=UserAPIKeyAuth(
                api_key="key-owned-1", user_id="member-1", user_role=LitellmUserRoles.INTERNAL_USER
            ),
        )

    where = tag_table.where_clauses()[0]
    assert where["api_key"] == {"in": ["key-owned-2"]}
    assert {"team_id": {"in": ["team-a"]}} in where["AND"]


@pytest.mark.asyncio
async def test_admin_tag_list_team_ids_scopes_dynamic_tags_to_team():
    """
    /tag/list?team_ids=team-a returns only tags used under that team and pushes
    the team filter into the dynamic-tag group_by query.
    """
    from unittest.mock import AsyncMock

    from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

    app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_id="admin-1", user_role=LitellmUserRoles.PROXY_ADMIN
    )
    group_by_mock = AsyncMock(
        return_value=[
            {
                "tag": "team-tag",
                "_min": {"created_at": "2026-06-01T00:00:00Z"},
                "_max": {"updated_at": "2026-06-02T00:00:00Z"},
            }
        ]
    )
    mock_prisma = SimpleNamespace(
        db=SimpleNamespace(
            litellm_dailytagspend=SimpleNamespace(group_by=group_by_mock),
            litellm_tagtable=SimpleNamespace(find_many=AsyncMock(return_value=[])),
            litellm_teamtable=SimpleNamespace(
                find_many=AsyncMock(return_value=[SimpleNamespace(team_id="team-a", team_alias="Team A")])
            ),
        )
    )

    try:
        with patch("litellm.proxy.proxy_server.prisma_client", mock_prisma):
            response = client.get("/tag/list?team_ids=team-a", headers={"Authorization": "Bearer sk"})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200, response.text
    assert [tag["name"] for tag in response.json()] == ["team-tag"]
    where = group_by_mock.await_args.kwargs["where"]
    assert where["team_id"] == {"in": ["team-a"]}
    assert "api_key" not in where


@pytest.mark.asyncio
async def test_internal_member_tag_list_team_ids_scoped_to_own_keys():
    """
    A team member without the daily-activity permission sees only tags produced
    by their own keys inside the requested team.
    """
    from unittest.mock import AsyncMock

    from litellm.proxy._types import LiteLLM_UserTable
    from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

    app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(
        api_key="key-m1", user_id="member-1", user_role=LitellmUserRoles.INTERNAL_USER
    )
    group_by_mock = AsyncMock(
        return_value=[
            {
                "tag": "member-tag",
                "_min": {"created_at": "2026-06-01T00:00:00Z"},
                "_max": {"updated_at": "2026-06-02T00:00:00Z"},
            }
        ]
    )
    token_record = Mock()
    token_record.token = "key-m2"
    mock_prisma = SimpleNamespace(
        db=SimpleNamespace(
            litellm_dailytagspend=SimpleNamespace(group_by=group_by_mock),
            litellm_tagtable=SimpleNamespace(find_many=AsyncMock(return_value=[])),
            litellm_teamtable=SimpleNamespace(
                find_many=AsyncMock(return_value=[_team_row_without_member_view("team-a", "member-1")])
            ),
            litellm_verificationtoken=FakeVerificationTokenTable([token_record]),
        )
    )
    member = LiteLLM_UserTable(
        user_id="member-1",
        teams=["team-a"],
        max_budget=None,
        spend=0.0,
        user_email=None,
        user_role="internal_user",
    )

    try:
        with (
            patch("litellm.proxy.proxy_server.prisma_client", mock_prisma),
            patch(
                "litellm.proxy.management_endpoints.team_endpoints.get_user_object",
                new=AsyncMock(return_value=member),
            ),
        ):
            response = client.get("/tag/list?team_ids=team-a", headers={"Authorization": "Bearer sk"})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200, response.text
    assert [tag["name"] for tag in response.json()] == ["member-tag"]
    where = group_by_mock.await_args.kwargs["where"]
    assert where["team_id"] == {"in": ["team-a"]}
    assert where["api_key"] == {"in": ["key-m2"]}


@pytest.mark.asyncio
async def test_tag_list_team_ids_non_member_gets_404():
    """
    /tag/list?team_ids=... applies the team daily-activity permission model:
    a user who is not in the team gets 404, not an unfiltered list.
    """
    from unittest.mock import AsyncMock

    from litellm.proxy._types import LiteLLM_UserTable
    from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

    app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_id="outsider-1", user_role=LitellmUserRoles.INTERNAL_USER
    )
    group_by_mock = AsyncMock(return_value=[])
    mock_prisma = SimpleNamespace(
        db=SimpleNamespace(
            litellm_dailytagspend=SimpleNamespace(group_by=group_by_mock),
            litellm_tagtable=SimpleNamespace(find_many=AsyncMock(return_value=[])),
            litellm_teamtable=SimpleNamespace(find_many=AsyncMock(return_value=[])),
            litellm_verificationtoken=FakeVerificationTokenTable([]),
        )
    )
    outsider = LiteLLM_UserTable(
        user_id="outsider-1",
        teams=["team-b"],
        max_budget=None,
        spend=0.0,
        user_email=None,
        user_role="internal_user",
    )

    try:
        with (
            patch("litellm.proxy.proxy_server.prisma_client", mock_prisma),
            patch(
                "litellm.proxy.management_endpoints.team_endpoints.get_user_object",
                new=AsyncMock(return_value=outsider),
            ),
        ):
            response = client.get("/tag/list?team_ids=team-a", headers={"Authorization": "Bearer sk"})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 404, response.text
    group_by_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_tag_list_team_ids_member_without_keys_returns_empty_without_querying():
    """
    A permitted member whose team resolver returns an empty key filter gets an
    empty list and the spend table is never queried.
    """
    from unittest.mock import AsyncMock

    from litellm.proxy._types import LiteLLM_UserTable
    from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

    app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_id="member-1", user_role=LitellmUserRoles.INTERNAL_USER
    )
    group_by_mock = AsyncMock(return_value=[])
    mock_prisma = SimpleNamespace(
        db=SimpleNamespace(
            litellm_dailytagspend=SimpleNamespace(group_by=group_by_mock),
            litellm_tagtable=SimpleNamespace(find_many=AsyncMock(return_value=[])),
            litellm_teamtable=SimpleNamespace(
                find_many=AsyncMock(return_value=[_team_row_without_member_view("team-a", "member-1")])
            ),
            litellm_verificationtoken=FakeVerificationTokenTable([]),
        )
    )
    member = LiteLLM_UserTable(
        user_id="member-1",
        teams=["team-a"],
        max_budget=None,
        spend=0.0,
        user_email=None,
        user_role="internal_user",
    )

    try:
        with (
            patch("litellm.proxy.proxy_server.prisma_client", mock_prisma),
            patch(
                "litellm.proxy.management_endpoints.team_endpoints.get_user_object",
                new=AsyncMock(return_value=member),
            ),
        ):
            response = client.get("/tag/list?team_ids=team-a", headers={"Authorization": "Bearer sk"})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200, response.text
    assert response.json() == []
    group_by_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_tag_list_usage_only_drops_stored_tags_without_usage():
    """
    usage_only=true drops stored tags that have no daily tag spend rows while
    keeping the full stored config for the ones that do.
    """
    from datetime import datetime
    from unittest.mock import AsyncMock

    from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

    app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_id="admin-1", user_role=LitellmUserRoles.PROXY_ADMIN
    )
    used_stored_tag = Mock()
    used_stored_tag.tag_name = "used-stored"
    used_stored_tag.description = "stored config kept"
    used_stored_tag.models = ["model-1"]
    used_stored_tag.model_info = {}
    used_stored_tag.spend = 0.0
    used_stored_tag.budget_id = None
    used_stored_tag.created_at = datetime(2026, 1, 1)
    used_stored_tag.updated_at = datetime(2026, 1, 2)
    used_stored_tag.created_by = "admin-1"
    used_stored_tag.litellm_budget_table = None
    tag_find_many = AsyncMock(return_value=[used_stored_tag])
    group_by_mock = AsyncMock(
        return_value=[
            {
                "tag": "used-stored",
                "_min": {"created_at": "2026-06-01T00:00:00Z"},
                "_max": {"updated_at": "2026-06-02T00:00:00Z"},
            },
            {
                "tag": "dynamic-only",
                "_min": {"created_at": "2026-06-03T00:00:00Z"},
                "_max": {"updated_at": "2026-06-04T00:00:00Z"},
            },
        ]
    )
    mock_prisma = SimpleNamespace(
        db=SimpleNamespace(
            litellm_dailytagspend=SimpleNamespace(group_by=group_by_mock),
            litellm_tagtable=SimpleNamespace(find_many=tag_find_many),
        )
    )

    try:
        with patch("litellm.proxy.proxy_server.prisma_client", mock_prisma):
            response = client.get("/tag/list?usage_only=true", headers={"Authorization": "Bearer sk"})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200, response.text
    tag_find_many.assert_awaited_once_with(
        where={"tag_name": {"in": ["used-stored", "dynamic-only"]}},
        include={"litellm_budget_table": True},
    )
    by_name = {tag["name"]: tag for tag in response.json()}
    assert set(by_name) == {"used-stored", "dynamic-only"}
    assert by_name["used-stored"]["description"] == "stored config kept"
    assert by_name["used-stored"]["models"] == ["model-1"]
