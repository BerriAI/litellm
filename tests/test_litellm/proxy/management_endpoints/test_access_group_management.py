"""
Test access group management endpoints
"""

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


from litellm import Router
from litellm.proxy.management_endpoints.model_management_endpoints import (
    ReconcileOutcome,
)


@pytest.mark.asyncio
async def test_create_duplicate_access_group_fails():
    """
    Test that creating an access group with a name that already exists returns 409 error.

    Scenario: User creates "production-models" access group, then tries to create it again.
    Should fail with 409 Conflict.
    """
    from fastapi import HTTPException

    from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
    from litellm.proxy.management_endpoints.model_access_group_management_endpoints import (
        create_model_group,
    )
    from litellm.types.proxy.management_endpoints.model_management_endpoints import (
        NewModelGroupRequest,
    )

    # Mock dependencies - use exact model name (not wildcard)
    mock_router = Router(
        model_list=[
            {
                "model_name": "gpt-4",  # Exact model name
                "litellm_params": {
                    "model": "gpt-4",
                    "api_key": "fake-key",
                },
            }
        ]
    )

    mock_prisma = MagicMock()
    mock_prisma.db.litellm_proxymodeltable.find_many = AsyncMock(
        return_value=[
            MagicMock(
                model_id="1",
                model_name="gpt-4",
                model_info={"access_groups": ["production-models"]},  # Already exists
            )
        ]
    )

    mock_user = UserAPIKeyAuth(
        user_id="test_admin",
        user_role=LitellmUserRoles.PROXY_ADMIN,
    )

    request_data = NewModelGroupRequest(
        access_group="production-models",
        model_names=["gpt-4"],
    )

    # Mock the imported dependencies from proxy_server (where they're actually imported from)
    with (
        patch("litellm.proxy.proxy_server.llm_router", mock_router),
        patch("litellm.proxy.proxy_server.prisma_client", mock_prisma),
    ):

        # Should raise 409 Conflict
        with pytest.raises(HTTPException) as exc_info:
            await create_model_group(data=request_data, user_api_key_dict=mock_user)

        assert exc_info.value.status_code == 409
        assert "already exists" in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_create_access_group_with_model_ids_tags_only_specific_deployments():
    """
    Test that using model_ids only tags the specific deployments, not all
    deployments sharing the same model_name.

    Fixes: https://github.com/BerriAI/litellm/issues/21544
    """
    from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
    from litellm.proxy.management_endpoints.model_access_group_management_endpoints import (
        create_model_group,
    )
    from litellm.types.proxy.management_endpoints.model_management_endpoints import (
        NewModelGroupRequest,
    )

    deploy_a = MagicMock(model_id="deploy-A", model_name="gpt-4o", model_info={})

    mock_prisma = MagicMock()
    mock_prisma.db.litellm_proxymodeltable.find_many = AsyncMock(return_value=[])
    mock_prisma.db.litellm_proxymodeltable.find_unique = AsyncMock(
        return_value=deploy_a
    )
    mock_prisma.db.litellm_proxymodeltable.update = AsyncMock()

    mock_user = UserAPIKeyAuth(
        user_id="test_admin",
        user_role=LitellmUserRoles.PROXY_ADMIN,
    )

    request_data = NewModelGroupRequest(
        access_group="production-models",
        model_ids=["deploy-A"],
    )

    with (
        patch("litellm.proxy.proxy_server.llm_router", MagicMock()),
        patch("litellm.proxy.proxy_server.prisma_client", mock_prisma),
        patch(
            "litellm.proxy.management_endpoints.model_access_group_management_endpoints.clear_cache",
            new=AsyncMock(return_value=ReconcileOutcome(still_desired=None, live_after=None)),
        ),
    ):
        response = await create_model_group(
            data=request_data, user_api_key_dict=mock_user
        )

    assert response.models_updated == 1
    assert response.model_ids == ["deploy-A"]
    mock_prisma.db.litellm_proxymodeltable.find_unique.assert_called_once_with(
        where={"model_id": "deploy-A"}
    )
    assert mock_prisma.db.litellm_proxymodeltable.update.call_count == 1
    update_call = mock_prisma.db.litellm_proxymodeltable.update.call_args
    assert update_call.kwargs["where"] == {"model_id": "deploy-A"}


@pytest.mark.asyncio
async def test_create_access_group_with_model_names_tags_all_deployments():
    """
    Test backward compat: model_names still tags ALL deployments sharing that model_name.
    """
    from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
    from litellm.proxy.management_endpoints.model_access_group_management_endpoints import (
        create_model_group,
    )
    from litellm.types.proxy.management_endpoints.model_management_endpoints import (
        NewModelGroupRequest,
    )

    deploy_a = MagicMock(model_id="deploy-A", model_name="gpt-4o", model_info={})
    deploy_b = MagicMock(model_id="deploy-B", model_name="gpt-4o", model_info={})
    deploy_c = MagicMock(model_id="deploy-C", model_name="gpt-4o", model_info={})

    mock_router = Router(
        model_list=[
            {
                "model_name": "gpt-4o",
                "litellm_params": {"model": "gpt-4o", "api_key": "fake-key"},
                "model_info": {"id": deployment_id, "db_model": True},
            }
            for deployment_id in ("deploy-A", "deploy-B", "deploy-C")
        ]
    )

    mock_prisma = MagicMock()
    mock_prisma.db.litellm_proxymodeltable.find_many = AsyncMock(
        side_effect=[[], [deploy_a, deploy_b, deploy_c]]
    )
    mock_prisma.db.litellm_proxymodeltable.update = AsyncMock()

    mock_user = UserAPIKeyAuth(
        user_id="test_admin",
        user_role=LitellmUserRoles.PROXY_ADMIN,
    )

    request_data = NewModelGroupRequest(
        access_group="production-models", model_names=["gpt-4o"]
    )

    with (
        patch("litellm.proxy.proxy_server.llm_router", mock_router),
        patch("litellm.proxy.proxy_server.prisma_client", mock_prisma),
        patch(
            "litellm.proxy.management_endpoints.model_access_group_management_endpoints.clear_cache",
            new=AsyncMock(return_value=ReconcileOutcome(still_desired=None, live_after=None)),
        ),
    ):
        response = await create_model_group(
            data=request_data, user_api_key_dict=mock_user
        )

    assert response.models_updated == 3
    assert response.model_names == ["gpt-4o"]
    assert mock_prisma.db.litellm_proxymodeltable.update.call_count == 3


@pytest.mark.asyncio
async def test_create_access_group_model_ids_takes_priority_over_model_names():
    """
    Test that when both model_ids and model_names are provided, model_ids is used.
    """
    from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
    from litellm.proxy.management_endpoints.model_access_group_management_endpoints import (
        create_model_group,
    )
    from litellm.types.proxy.management_endpoints.model_management_endpoints import (
        NewModelGroupRequest,
    )

    deploy_a = MagicMock(model_id="deploy-A", model_name="gpt-4o", model_info={})

    mock_prisma = MagicMock()
    mock_prisma.db.litellm_proxymodeltable.find_many = AsyncMock(return_value=[])
    mock_prisma.db.litellm_proxymodeltable.find_unique = AsyncMock(
        return_value=deploy_a
    )
    mock_prisma.db.litellm_proxymodeltable.update = AsyncMock()

    mock_user = UserAPIKeyAuth(
        user_id="test_admin",
        user_role=LitellmUserRoles.PROXY_ADMIN,
    )

    request_data = NewModelGroupRequest(
        access_group="production-models",
        model_names=["gpt-4o"],
        model_ids=["deploy-A"],
    )

    with (
        patch("litellm.proxy.proxy_server.llm_router", MagicMock()),
        patch("litellm.proxy.proxy_server.prisma_client", mock_prisma),
        patch(
            "litellm.proxy.management_endpoints.model_access_group_management_endpoints.clear_cache",
            new=AsyncMock(return_value=ReconcileOutcome(still_desired=None, live_after=None)),
        ),
    ):
        response = await create_model_group(
            data=request_data, user_api_key_dict=mock_user
        )

    assert response.models_updated == 1
    mock_prisma.db.litellm_proxymodeltable.find_unique.assert_called_once_with(
        where={"model_id": "deploy-A"}
    )


@pytest.mark.asyncio
async def test_create_access_group_requires_model_names_or_model_ids():
    """
    Test that creating an access group without model_names or model_ids fails.
    """
    from fastapi import HTTPException
    from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
    from litellm.proxy.management_endpoints.model_access_group_management_endpoints import (
        create_model_group,
    )
    from litellm.types.proxy.management_endpoints.model_management_endpoints import (
        NewModelGroupRequest,
    )

    mock_user = UserAPIKeyAuth(
        user_id="test_admin",
        user_role=LitellmUserRoles.PROXY_ADMIN,
    )

    request_data = NewModelGroupRequest(access_group="production-models")

    with (
        patch("litellm.proxy.proxy_server.llm_router", MagicMock()),
        patch("litellm.proxy.proxy_server.prisma_client", MagicMock()),
    ):
        with pytest.raises(HTTPException) as exc_info:
            await create_model_group(data=request_data, user_api_key_dict=mock_user)
        assert exc_info.value.status_code == 400
        assert "model_names or model_ids" in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_create_access_group_invalid_model_id_returns_400():
    """
    Test that passing a non-existent model_id returns 400 error.
    """
    from fastapi import HTTPException
    from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
    from litellm.proxy.management_endpoints.model_access_group_management_endpoints import (
        create_model_group,
    )
    from litellm.types.proxy.management_endpoints.model_management_endpoints import (
        NewModelGroupRequest,
    )

    mock_prisma = MagicMock()
    mock_prisma.db.litellm_proxymodeltable.find_many = AsyncMock(return_value=[])
    mock_prisma.db.litellm_proxymodeltable.find_unique = AsyncMock(return_value=None)

    mock_user = UserAPIKeyAuth(
        user_id="test_admin",
        user_role=LitellmUserRoles.PROXY_ADMIN,
    )

    request_data = NewModelGroupRequest(
        access_group="production-models",
        model_ids=["non-existent-id"],
    )

    with (
        patch("litellm.proxy.proxy_server.llm_router", MagicMock()),
        patch("litellm.proxy.proxy_server.prisma_client", mock_prisma),
        patch(
            "litellm.proxy.management_endpoints.model_access_group_management_endpoints.clear_cache",
            new=AsyncMock(return_value=ReconcileOutcome(still_desired=None, live_after=None)),
        ),
    ):
        with pytest.raises(HTTPException) as exc_info:
            await create_model_group(data=request_data, user_api_key_dict=mock_user)
        assert exc_info.value.status_code == 400
        assert "non-existent-id" in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_create_access_group_surfaces_dropped_models():
    """An access-group write whose reload does not leave the tagged models live on this
    pod must report the drop through this file's HTTPException contract, not a 200."""
    from fastapi import HTTPException

    from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
    from litellm.proxy.management_endpoints.model_access_group_management_endpoints import (
        create_model_group,
    )
    from litellm.types.proxy.management_endpoints.model_management_endpoints import (
        NewModelGroupRequest,
    )

    deploy_a = MagicMock(model_id="deploy-A", model_name="gpt-4o", model_info={})

    mock_prisma = MagicMock()
    mock_prisma.db.litellm_proxymodeltable.find_many = AsyncMock(return_value=[])
    mock_prisma.db.litellm_proxymodeltable.find_unique = AsyncMock(return_value=deploy_a)
    mock_prisma.db.litellm_proxymodeltable.update = AsyncMock()

    mock_user = UserAPIKeyAuth(user_id="test_admin", user_role=LitellmUserRoles.PROXY_ADMIN)

    wiped_router = MagicMock()
    wiped_router.get_model_ids.side_effect = [["deploy-A"], []]
    with (
        patch("litellm.proxy.proxy_server.llm_router", wiped_router),
        patch("litellm.proxy.proxy_server.prisma_client", mock_prisma),
        patch(
            "litellm.proxy.management_endpoints.model_access_group_management_endpoints.clear_cache",
            new=AsyncMock(return_value=ReconcileOutcome(still_desired=None, live_after=None)),
        ),
    ):
        with pytest.raises(HTTPException) as exc_info:
            await create_model_group(
                data=NewModelGroupRequest(access_group="production-models", model_ids=["deploy-A"]),
                user_api_key_dict=mock_user,
            )

    assert exc_info.value.status_code == 500
    assert "deploy-A" in str(exc_info.value.detail)



@pytest.mark.asyncio
async def test_create_access_group_trusts_reload_snapshot_over_post_lock_fresh_read():
    """A concurrent reconcile sampled after the lock is released must not make this
    write's reload look like it dropped the tagged model: the verdict has to judge from
    the ReconcileOutcome the reload captured under the lock, not a fresh router read."""
    from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
    from litellm.proxy.management_endpoints.model_access_group_management_endpoints import (
        create_model_group,
    )
    from litellm.types.proxy.management_endpoints.model_management_endpoints import (
        NewModelGroupRequest,
    )

    deploy_a = MagicMock(model_id="deploy-A", model_name="gpt-4o", model_info={})

    mock_prisma = MagicMock()
    mock_prisma.db.litellm_proxymodeltable.find_many = AsyncMock(return_value=[])
    mock_prisma.db.litellm_proxymodeltable.find_unique = AsyncMock(return_value=deploy_a)
    mock_prisma.db.litellm_proxymodeltable.update = AsyncMock()

    concurrently_wiped_router = MagicMock()
    concurrently_wiped_router.get_model_ids.side_effect = [["deploy-A"], []]
    with (
        patch("litellm.proxy.proxy_server.llm_router", concurrently_wiped_router),
        patch("litellm.proxy.proxy_server.prisma_client", mock_prisma),
        patch(
            "litellm.proxy.management_endpoints.model_access_group_management_endpoints.clear_cache",
            new=AsyncMock(
                return_value=ReconcileOutcome(
                    still_desired=frozenset({"deploy-A"}), live_after=frozenset({"deploy-A"})
                )
            ),
        ),
    ):
        response = await create_model_group(
            data=NewModelGroupRequest(access_group="production-models", model_ids=["deploy-A"]),
            user_api_key_dict=UserAPIKeyAuth(user_id="test_admin", user_role=LitellmUserRoles.PROXY_ADMIN),
        )

    assert response.models_updated == 1
    assert concurrently_wiped_router.get_model_ids.call_count == 1


@pytest.mark.asyncio
async def test_tag_deployment_parses_string_model_info_and_refuses_corrupt():
    """The model_info column can arrive as its JSON string; tagging must parse it rather
    than crash, and must refuse to rewrite a present-but-unreadable value."""
    from litellm.proxy.management_endpoints.model_access_group_management_endpoints import (
        _tag_deployment_with_access_group,
    )

    mock_prisma = MagicMock()
    mock_prisma.db.litellm_proxymodeltable.update = AsyncMock()

    pair = await _tag_deployment_with_access_group(
        model_id="deploy-str",
        model_info='{"access_groups": ["existing"]}',
        access_group="new-group",
        prisma_client=mock_prisma,
    )
    assert pair is not None
    assert pair[0] == "deploy-str"
    assert pair[1]["access_groups"] == ["existing", "new-group"]

    with pytest.raises(ValueError, match="deploy-corrupt"):
        await _tag_deployment_with_access_group(
            model_id="deploy-corrupt",
            model_info="{not json",
            access_group="new-group",
            prisma_client=mock_prisma,
        )


@pytest.mark.asyncio
async def test_delete_access_group_ignores_models_that_were_already_dead():
    """A metadata-only strip over a model this pod never served must not fail the write;
    the model's deadness predates the request, and blaming it here would make a broken
    model block every access-group fix that touches it."""
    deploy_broken = MagicMock(
        model_id="deploy-broken", model_name="broken-model", model_info={"access_groups": ["doomed-group"]}
    )

    mock_prisma = MagicMock()
    mock_prisma.db.litellm_proxymodeltable.find_many = AsyncMock(return_value=[deploy_broken])
    mock_prisma.db.litellm_proxymodeltable.update = AsyncMock()
    mock_prisma.db.litellm_modelaccessgroupbudgettable.delete = AsyncMock(return_value=None)

    from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
    from litellm.proxy.management_endpoints.model_access_group_management_endpoints import (
        delete_access_group,
    )

    never_served_router = MagicMock()
    never_served_router.get_model_ids.return_value = []
    with (
        patch("litellm.proxy.proxy_server.llm_router", never_served_router),
        patch("litellm.proxy.proxy_server.prisma_client", mock_prisma),
        patch(
            "litellm.proxy.management_endpoints.model_access_group_management_endpoints.clear_cache",
            new=AsyncMock(return_value=ReconcileOutcome(still_desired=None, live_after=None)),
        ),
    ):
        response = await delete_access_group(
            access_group="doomed-group",
            user_api_key_dict=UserAPIKeyAuth(user_id="admin", user_role=LitellmUserRoles.PROXY_ADMIN),
            auth_cache=_FakeAuthCache(),
        )

    assert response.models_updated == 1
    mock_prisma.db.litellm_proxymodeltable.update.assert_awaited_once()


@pytest.mark.asyncio
async def test_create_access_group_read_through_recovers_model_created_on_sibling_replica():
    """Regression: an access group referencing a model that another replica just wrote
    to the DB must be created instead of 400ing until the periodic config reload."""
    from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
    from litellm.proxy.management_endpoints.model_access_group_management_endpoints import (
        create_model_group,
    )
    from litellm.types.proxy.management_endpoints.model_management_endpoints import (
        NewModelGroupRequest,
    )

    from types import SimpleNamespace

    model_name = "e2e-ag-sibling-replica-model"
    db_row = SimpleNamespace(
        model_id=f"{model_name}-id",
        model_name=model_name,
        litellm_params={"model": "openai/gpt-4o", "api_key": "fake", "mock_response": "hi"},
        model_info={},
        blocked=False,
    )

    mock_router = Router(
        model_list=[
            {
                "model_name": "some-other-model",
                "litellm_params": {"model": "openai/gpt-4o", "api_key": "fake"},
            }
        ]
    )

    mock_prisma = MagicMock()
    mock_prisma.db.litellm_proxymodeltable.find_many = AsyncMock(side_effect=[[db_row], [], [db_row]])
    mock_prisma.db.litellm_proxymodeltable.update = AsyncMock()

    with (
        patch("litellm.proxy.proxy_server.llm_router", mock_router),
        patch("litellm.proxy.proxy_server.prisma_client", mock_prisma),
        patch("litellm.proxy.proxy_server.store_model_in_db", True),
        patch(
            "litellm.proxy.management_endpoints.model_access_group_management_endpoints.clear_cache",
            new=AsyncMock(return_value=ReconcileOutcome(still_desired=None, live_after=None)),
        ),
    ):
        response = await create_model_group(
            data=NewModelGroupRequest(access_group="replica-lag-group", model_names=[model_name]),
            user_api_key_dict=UserAPIKeyAuth(user_id="admin", user_role=LitellmUserRoles.PROXY_ADMIN),
        )

    assert response.models_updated == 1
    assert response.model_names == [model_name]
    assert mock_prisma.db.litellm_proxymodeltable.find_many.await_args_list[0].kwargs["where"] == {
        "model_name": model_name
    }


@pytest.mark.asyncio
async def test_create_access_group_model_missing_everywhere_still_400s():
    from fastapi import HTTPException

    from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
    from litellm.proxy.management_endpoints.model_access_group_management_endpoints import (
        create_model_group,
    )
    from litellm.types.proxy.management_endpoints.model_management_endpoints import (
        NewModelGroupRequest,
    )

    model_name = "e2e-ag-model-nobody-created"
    mock_router = Router(
        model_list=[
            {
                "model_name": "some-other-model",
                "litellm_params": {"model": "openai/gpt-4o", "api_key": "fake"},
            }
        ]
    )
    mock_prisma = MagicMock()
    mock_prisma.db.litellm_proxymodeltable.find_many = AsyncMock(return_value=[])

    with (
        patch("litellm.proxy.proxy_server.llm_router", mock_router),
        patch("litellm.proxy.proxy_server.prisma_client", mock_prisma),
        patch("litellm.proxy.proxy_server.store_model_in_db", True),
    ):
        with pytest.raises(HTTPException) as exc_info:
            await create_model_group(
                data=NewModelGroupRequest(access_group="ghost-group", model_names=[model_name]),
                user_api_key_dict=UserAPIKeyAuth(user_id="admin", user_role=LitellmUserRoles.PROXY_ADMIN),
            )

    assert exc_info.value.status_code == 400
    assert model_name in str(exc_info.value.detail)

@dataclass
class _FakeBudgetRow:
    budget_id: str
    max_budget: float | None = None
    soft_budget: float | None = None
    budget_duration: str | None = None
    budget_reset_at: datetime | None = None


@dataclass
class _FakeAccessGroupBudgetRow:
    access_group_name: str
    budget_id: str | None = None
    spend: float = 0.0
    litellm_budget_table: _FakeBudgetRow | None = None


@dataclass
class _FakeDeployment:
    model_id: str
    model_name: str
    model_info: dict


class _FakeBudgetTable:
    """Stands in for litellm_budgettable so a test can see whether a budget row was created,
    updated in place, or left orphaned."""

    def __init__(self, journal: list[str]) -> None:
        self.journal = journal
        self.rows: dict[str, _FakeBudgetRow] = {}
        self.create_calls: list[dict] = []
        self.update_calls: list[tuple[str, dict]] = []
        self.deleted_ids: list[str] = []
        self._sequence = 0

    async def create(self, data, include=None):
        self._sequence += 1
        budget_id = str(data.get("budget_id") or f"budget-{self._sequence}")
        row = _FakeBudgetRow(
            budget_id=budget_id,
            max_budget=data.get("max_budget"),
            soft_budget=data.get("soft_budget"),
            budget_duration=data.get("budget_duration"),
        )
        self.rows[budget_id] = row
        self.create_calls.append(dict(data))
        self.journal.append(f"budget_table.create:{budget_id}")
        return row

    async def update(self, where, data, include=None):
        budget_id = where["budget_id"]
        self.update_calls.append((budget_id, dict(data)))
        self.journal.append(f"budget_table.update:{budget_id}")
        row = self.rows.get(budget_id)
        if row is None:
            return None
        for field_name in ("max_budget", "soft_budget", "budget_duration"):
            if data.get(field_name) is not None:
                setattr(row, field_name, data[field_name])
        return row

    async def delete(self, where, include=None):
        budget_id = where["budget_id"]
        self.journal.append(f"budget_table.delete:{budget_id}")
        self.deleted_ids.append(budget_id)
        return self.rows.pop(budget_id, None)


class _FakeAccessGroupBudgetTable:
    """Stands in for litellm_modelaccessgroupbudgettable, resolving `include` against the fake
    budget table the way prisma resolves the relation."""

    def __init__(self, journal: list[str], budget_table: _FakeBudgetTable) -> None:
        self.journal = journal
        self.budget_table = budget_table
        self.rows: dict[str, _FakeAccessGroupBudgetRow] = {}
        self.upsert_calls: list[dict] = []

    def _resolve(self, row, include):
        if row is None:
            return None
        row.litellm_budget_table = (
            self.budget_table.rows.get(row.budget_id) if include and row.budget_id is not None else None
        )
        return row

    async def find_unique(self, where, include=None):
        return self._resolve(self.rows.get(where["access_group_name"]), include)

    async def upsert(self, where, data, include=None):
        access_group_name = where["access_group_name"]
        self.upsert_calls.append(dict(data))
        self.journal.append(f"access_group_budget.upsert:{access_group_name}")
        existing = self.rows.get(access_group_name)
        payload = data["update"] if existing is not None else data["create"]
        row = existing or _FakeAccessGroupBudgetRow(access_group_name=access_group_name)
        row.budget_id = payload.get("budget_id")
        self.rows[access_group_name] = row
        return self._resolve(row, include)

    async def delete(self, where, include=None):
        access_group_name = where["access_group_name"]
        self.journal.append(f"access_group_budget.delete:{access_group_name}")
        return self.rows.pop(access_group_name, None)


class _FakeModelTable:
    def __init__(self, journal: list[str], deployments) -> None:
        self.journal = journal
        self.deployments = list(deployments)
        self.updates: list[tuple[dict, dict]] = []

    async def find_many(self, where=None, **kwargs):
        return list(self.deployments)

    async def find_unique(self, where, include=None):
        return next((d for d in self.deployments if d.model_id == where["model_id"]), None)

    async def update(self, where, data, include=None):
        self.journal.append(f"model_table.update:{where['model_id']}")
        self.updates.append((dict(where), dict(data)))
        return None


class _FakePrismaClient:
    def __init__(self, journal: list[str], deployments=()) -> None:
        self.budget_table = _FakeBudgetTable(journal)
        self.access_group_budget_table = _FakeAccessGroupBudgetTable(journal, self.budget_table)
        self.model_table = _FakeModelTable(journal, deployments)
        self.db = SimpleNamespace(
            litellm_budgettable=self.budget_table,
            litellm_modelaccessgroupbudgettable=self.access_group_budget_table,
            litellm_proxymodeltable=self.model_table,
        )

    def jsonify_object(self, data):
        return dict(data)


class _FakeAuthCache:
    """Spy for the auth cache the endpoints evict through. Injected into the endpoint rather than
    patched over the proxy_server global, so dropping the eviction call fails a test."""

    def __init__(self, journal: list[str] | None = None) -> None:
        self.journal = journal if journal is not None else []
        self.deleted_keys: list[str] = []

    async def async_delete_cache(self, key):
        self.deleted_keys.append(key)
        self.journal.append(f"auth_cache.delete:{key}")


def _admin():
    from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth

    return UserAPIKeyAuth(user_id="admin", user_role=LitellmUserRoles.PROXY_ADMIN)


def _deployment(model_id="deploy-1", model_name="gpt-4o", access_groups=("prod-models",)):
    return _FakeDeployment(
        model_id=model_id,
        model_name=model_name,
        model_info={"access_groups": list(access_groups)},
    )


def _seed_budget(prisma, access_group, spend=0.0, budget_id="budget-seed", **budget_fields):
    prisma.budget_table.rows[budget_id] = _FakeBudgetRow(budget_id=budget_id, **budget_fields)
    prisma.access_group_budget_table.rows[access_group] = _FakeAccessGroupBudgetRow(
        access_group_name=access_group,
        budget_id=budget_id,
        spend=spend,
    )


@contextmanager
def _proxy(prisma):
    with patch("litellm.proxy.proxy_server.prisma_client", prisma):
        yield


def _eviction_journal(access_group):
    """Both auth cache keys, in the order a write path has to evict them."""
    from litellm.proxy.common_utils.user_api_key_cache import (
        model_access_group_cache_key,
        model_access_group_registry_cache_key,
    )

    return [
        f"auth_cache.delete:{model_access_group_cache_key(access_group)}",
        f"auth_cache.delete:{model_access_group_registry_cache_key()}",
    ]


def _assert_evicted_after_write(journal, access_group, write_entry):
    """Exactly the two keys, in order, after the DB write. Deliberately not a tail slice: what
    has to hold is that the eviction follows the write, not that nothing follows the eviction."""
    evictions = [entry for entry in journal if entry.startswith("auth_cache.delete:")]
    assert evictions == _eviction_journal(access_group)
    assert journal.index(write_entry) < journal.index(evictions[0])


@pytest.mark.asyncio
async def test_put_access_group_budget_creates_the_row_and_its_budget():
    """First PUT has to create both halves: the budget row it links, and the access group row
    that carries the link and the shared spend."""
    from litellm.proxy.management_endpoints.model_access_group_management_endpoints import (
        set_access_group_budget,
    )
    from litellm.types.proxy.management_endpoints.model_management_endpoints import (
        AccessGroupBudgetRequest,
    )

    prisma = _FakePrismaClient([], deployments=[_deployment()])
    cache = _FakeAuthCache()

    with _proxy(prisma):
        response = await set_access_group_budget(
            access_group="prod-models",
            data=AccessGroupBudgetRequest(max_budget=100.0, soft_budget=80.0, budget_duration="30d"),
            user_api_key_dict=_admin(),
            auth_cache=cache,
        )

    assert response.access_group == "prod-models"
    assert response.spend == 0.0
    assert response.budget is not None
    assert response.budget.max_budget == 100.0
    assert response.budget.soft_budget == 80.0
    assert response.budget.budget_duration == "30d"
    assert len(prisma.budget_table.create_calls) == 1
    assert prisma.access_group_budget_table.rows["prod-models"].budget_id == response.budget.budget_id


@pytest.mark.asyncio
async def test_second_put_replaces_the_budget_instead_of_creating_another():
    """PUT is idempotent: a second call must update the budget already linked to the group,
    not leave a second budget row (and a second group row) behind."""
    from litellm.proxy.management_endpoints.model_access_group_management_endpoints import (
        set_access_group_budget,
    )
    from litellm.types.proxy.management_endpoints.model_management_endpoints import (
        AccessGroupBudgetRequest,
    )

    prisma = _FakePrismaClient([], deployments=[_deployment()])
    cache = _FakeAuthCache()

    with _proxy(prisma):
        first = await set_access_group_budget(
            access_group="prod-models",
            data=AccessGroupBudgetRequest(max_budget=100.0),
            user_api_key_dict=_admin(),
            auth_cache=cache,
        )
        second = await set_access_group_budget(
            access_group="prod-models",
            data=AccessGroupBudgetRequest(max_budget=250.0),
            user_api_key_dict=_admin(),
            auth_cache=cache,
        )

    assert first.budget is not None and second.budget is not None
    assert second.budget.budget_id == first.budget.budget_id
    assert second.budget.max_budget == 250.0
    assert len(prisma.budget_table.create_calls) == 1
    assert len(prisma.budget_table.rows) == 1
    assert len(prisma.access_group_budget_table.rows) == 1
    assert prisma.budget_table.update_calls[-1][0] == first.budget.budget_id


@pytest.mark.asyncio
async def test_put_access_group_budget_links_an_existing_budget_without_creating_one():
    from litellm.proxy.management_endpoints.model_access_group_management_endpoints import (
        set_access_group_budget,
    )
    from litellm.types.proxy.management_endpoints.model_management_endpoints import (
        AccessGroupBudgetRequest,
    )

    prisma = _FakePrismaClient([], deployments=[_deployment()])
    prisma.budget_table.rows["shared-budget"] = _FakeBudgetRow(budget_id="shared-budget", max_budget=7.0)
    cache = _FakeAuthCache()

    with _proxy(prisma):
        response = await set_access_group_budget(
            access_group="prod-models",
            data=AccessGroupBudgetRequest(budget_id="shared-budget"),
            user_api_key_dict=_admin(),
            auth_cache=cache,
        )

    assert prisma.budget_table.create_calls == []
    assert response.budget is not None
    assert response.budget.budget_id == "shared-budget"
    assert response.budget.max_budget == 7.0
    assert prisma.access_group_budget_table.rows["prod-models"].budget_id == "shared-budget"


@pytest.mark.asyncio
async def test_put_access_group_budget_rejects_an_empty_body():
    """An empty PUT would register the group as budgeted while enforcing nothing."""
    from fastapi import HTTPException

    from litellm.proxy.management_endpoints.model_access_group_management_endpoints import (
        set_access_group_budget,
    )
    from litellm.types.proxy.management_endpoints.model_management_endpoints import (
        AccessGroupBudgetRequest,
    )

    prisma = _FakePrismaClient([], deployments=[_deployment()])
    cache = _FakeAuthCache()

    with _proxy(prisma), pytest.raises(HTTPException) as exc_info:
        await set_access_group_budget(
            access_group="prod-models",
            data=AccessGroupBudgetRequest(),
            user_api_key_dict=_admin(),
            auth_cache=cache,
        )

    assert exc_info.value.status_code == 400
    assert prisma.access_group_budget_table.rows == {}
    assert cache.deleted_keys == []


@pytest.mark.asyncio
async def test_put_access_group_budget_rejects_an_unparseable_duration():
    """An unparseable duration can only be discovered by the reset job, long after the write."""
    from fastapi import HTTPException

    from litellm.proxy.management_endpoints.model_access_group_management_endpoints import (
        set_access_group_budget,
    )
    from litellm.types.proxy.management_endpoints.model_management_endpoints import (
        AccessGroupBudgetRequest,
    )

    prisma = _FakePrismaClient([], deployments=[_deployment()])
    cache = _FakeAuthCache()

    with _proxy(prisma), pytest.raises(HTTPException) as exc_info:
        await set_access_group_budget(
            access_group="prod-models",
            data=AccessGroupBudgetRequest(max_budget=10.0, budget_duration="every other tuesday"),
            user_api_key_dict=_admin(),
            auth_cache=cache,
        )

    assert exc_info.value.status_code == 400
    assert prisma.budget_table.create_calls == []
    assert prisma.access_group_budget_table.rows == {}


def test_access_group_budget_request_rejects_rate_limit_fields():
    """tpm/rpm/max_parallel_requests are not enforced per access group, so accepting them would
    promise rate limiting that never happens."""
    from pydantic import ValidationError

    from litellm.types.proxy.management_endpoints.model_management_endpoints import (
        AccessGroupBudgetRequest,
    )

    for unsupported in ({"tpm_limit": 10}, {"rpm_limit": 10}, {"max_parallel_requests": 10}):
        with pytest.raises(ValidationError):
            AccessGroupBudgetRequest(max_budget=1.0, **unsupported)


@pytest.mark.asyncio
async def test_get_access_group_budget_returns_the_budget_and_the_shared_spend():
    from litellm.proxy.management_endpoints.model_access_group_management_endpoints import (
        get_access_group_budget,
    )

    prisma = _FakePrismaClient([], deployments=[_deployment()])
    _seed_budget(prisma, "prod-models", spend=42.5, max_budget=100.0, budget_duration="30d")

    with _proxy(prisma):
        response = await get_access_group_budget(access_group="prod-models", user_api_key_dict=_admin())

    assert response.access_group == "prod-models"
    assert response.spend == 42.5
    assert response.budget is not None
    assert response.budget.max_budget == 100.0
    assert response.budget.budget_duration == "30d"


@pytest.mark.asyncio
async def test_get_access_group_budget_on_a_budgetless_group_is_200_not_404():
    """A real group that simply has no budget is not an error; only an unknown group is."""
    from litellm.proxy.management_endpoints.model_access_group_management_endpoints import (
        get_access_group_budget,
    )

    prisma = _FakePrismaClient([], deployments=[_deployment()])

    with _proxy(prisma):
        response = await get_access_group_budget(access_group="prod-models", user_api_key_dict=_admin())

    assert response.spend == 0.0
    assert response.budget is None


@pytest.mark.asyncio
async def test_access_group_budget_routes_404_on_an_unknown_group():
    from fastapi import HTTPException

    from litellm.proxy.management_endpoints.model_access_group_management_endpoints import (
        delete_access_group_budget,
        get_access_group_budget,
        set_access_group_budget,
    )
    from litellm.types.proxy.management_endpoints.model_management_endpoints import (
        AccessGroupBudgetRequest,
    )

    prisma = _FakePrismaClient([], deployments=[_deployment()])
    cache = _FakeAuthCache()
    admin = _admin()

    calls = (
        lambda: get_access_group_budget(access_group="ghost-group", user_api_key_dict=admin),
        lambda: set_access_group_budget(
            access_group="ghost-group",
            data=AccessGroupBudgetRequest(max_budget=1.0),
            user_api_key_dict=admin,
            auth_cache=cache,
        ),
        lambda: delete_access_group_budget(
            access_group="ghost-group", user_api_key_dict=admin, auth_cache=cache
        ),
    )

    with _proxy(prisma):
        for make_call in calls:
            with pytest.raises(HTTPException) as exc_info:
                await make_call()
            assert exc_info.value.status_code == 404

    assert prisma.budget_table.create_calls == []
    assert prisma.access_group_budget_table.rows == {}


@pytest.mark.asyncio
async def test_delete_access_group_budget_drops_the_row_and_spares_the_shared_budget():
    """The group row goes; the LiteLLM_BudgetTable row it linked survives, as /tag/delete leaves a
    tag's. That row can be shared, so deleting it would be data loss for whatever else points at it."""
    from litellm.proxy.management_endpoints.model_access_group_management_endpoints import (
        delete_access_group_budget,
    )

    prisma = _FakePrismaClient([], deployments=[_deployment()])
    _seed_budget(prisma, "prod-models", spend=12.0, max_budget=100.0)
    cache = _FakeAuthCache()

    with _proxy(prisma):
        response = await delete_access_group_budget(
            access_group="prod-models", user_api_key_dict=_admin(), auth_cache=cache
        )

    assert response.budget_deleted is True
    assert prisma.access_group_budget_table.rows == {}
    assert prisma.budget_table.deleted_ids == []
    assert prisma.budget_table.rows["budget-seed"].max_budget == 100.0


@pytest.mark.asyncio
async def test_delete_access_group_budget_on_a_budgetless_group_still_evicts():
    """budget_deleted is False, but the group can still be sitting in the cached registry of
    budgeted groups, so the eviction has to run whether or not a row was there to drop."""
    from litellm.proxy.management_endpoints.model_access_group_management_endpoints import (
        delete_access_group_budget,
    )

    journal: list[str] = []
    prisma = _FakePrismaClient(journal, deployments=[_deployment()])
    cache = _FakeAuthCache(journal)

    with _proxy(prisma):
        response = await delete_access_group_budget(
            access_group="prod-models", user_api_key_dict=_admin(), auth_cache=cache
        )

    assert response.budget_deleted is False
    assert prisma.budget_table.deleted_ids == []
    _assert_evicted_after_write(journal, "prod-models", "access_group_budget.delete:prod-models")


@pytest.mark.asyncio
async def test_deleting_the_access_group_strips_deployments_before_dropping_the_budget():
    """Ordering is the point: stripping first means a failure leaves an unreachable budget row,
    while the reverse leaves a live group whose enforcement silently vanished. The shared
    LiteLLM_BudgetTable row survives here too."""
    from litellm.proxy.management_endpoints.model_access_group_management_endpoints import (
        delete_access_group,
    )

    journal: list[str] = []
    prisma = _FakePrismaClient(journal, deployments=[_deployment()])
    _seed_budget(prisma, "prod-models", spend=3.0, max_budget=100.0)
    cache = _FakeAuthCache()

    never_served_router = MagicMock()
    never_served_router.get_model_ids.return_value = []
    with (
        _proxy(prisma),
        patch("litellm.proxy.proxy_server.llm_router", never_served_router),
        patch(
            "litellm.proxy.management_endpoints.model_access_group_management_endpoints.clear_cache",
            new=AsyncMock(return_value=ReconcileOutcome(still_desired=None, live_after=None)),
        ),
    ):
        response = await delete_access_group(
            access_group="prod-models", user_api_key_dict=_admin(), auth_cache=cache
        )

    assert response.models_updated == 1
    assert prisma.access_group_budget_table.rows == {}
    assert prisma.budget_table.deleted_ids == []
    assert prisma.budget_table.rows["budget-seed"].max_budget == 100.0
    assert journal.index("model_table.update:deploy-1") < journal.index("access_group_budget.delete:prod-models")


@pytest.mark.asyncio
async def test_access_group_info_surfaces_the_budget_and_spend():
    from litellm.proxy.management_endpoints.model_access_group_management_endpoints import (
        get_access_group_info,
    )

    prisma = _FakePrismaClient([], deployments=[_deployment()])
    _seed_budget(prisma, "prod-models", spend=9.5, max_budget=100.0, soft_budget=50.0)

    with _proxy(prisma):
        info = await get_access_group_info(access_group="prod-models", user_api_key_dict=_admin())

    assert info.model_names == ["gpt-4o"]
    assert info.spend == 9.5
    assert info.budget is not None
    assert info.budget.max_budget == 100.0
    assert info.budget.soft_budget == 50.0


@pytest.mark.asyncio
async def test_put_access_group_budget_evicts_both_auth_cache_keys():
    """Auth reads the per-group row and the registry of budgeted groups cache-first with no
    freshness check, so a PUT that skips either eviction returns 200 and enforces nothing until
    the TTL expires. Both keys, after the write."""
    from litellm.proxy.management_endpoints.model_access_group_management_endpoints import (
        set_access_group_budget,
    )
    from litellm.types.proxy.management_endpoints.model_management_endpoints import (
        AccessGroupBudgetRequest,
    )

    journal: list[str] = []
    prisma = _FakePrismaClient(journal, deployments=[_deployment()])
    cache = _FakeAuthCache(journal)

    with _proxy(prisma):
        await set_access_group_budget(
            access_group="prod-models",
            data=AccessGroupBudgetRequest(max_budget=100.0),
            user_api_key_dict=_admin(),
            auth_cache=cache,
        )

    _assert_evicted_after_write(journal, "prod-models", "access_group_budget.upsert:prod-models")


@pytest.mark.asyncio
async def test_delete_access_group_budget_evicts_both_auth_cache_keys():
    """Clearing a budget has the same window as setting one: until both keys are dropped, auth
    keeps enforcing the budget that is already gone."""
    from litellm.proxy.management_endpoints.model_access_group_management_endpoints import (
        delete_access_group_budget,
    )

    journal: list[str] = []
    prisma = _FakePrismaClient(journal, deployments=[_deployment()])
    _seed_budget(prisma, "prod-models", spend=12.0, max_budget=100.0)
    cache = _FakeAuthCache(journal)

    with _proxy(prisma):
        await delete_access_group_budget(
            access_group="prod-models", user_api_key_dict=_admin(), auth_cache=cache
        )

    _assert_evicted_after_write(journal, "prod-models", "access_group_budget.delete:prod-models")


@pytest.mark.asyncio
async def test_deleting_the_access_group_evicts_both_auth_cache_keys():
    """The group-delete cascade drops the budget row too, so it owes the same two evictions."""
    from litellm.proxy.management_endpoints.model_access_group_management_endpoints import (
        delete_access_group,
    )

    journal: list[str] = []
    prisma = _FakePrismaClient(journal, deployments=[_deployment()])
    _seed_budget(prisma, "prod-models", spend=3.0, max_budget=100.0)
    cache = _FakeAuthCache(journal)

    never_served_router = MagicMock()
    never_served_router.get_model_ids.return_value = []
    with (
        _proxy(prisma),
        patch("litellm.proxy.proxy_server.llm_router", never_served_router),
        patch(
            "litellm.proxy.management_endpoints.model_access_group_management_endpoints.clear_cache",
            new=AsyncMock(return_value=ReconcileOutcome(still_desired=None, live_after=None)),
        ),
    ):
        await delete_access_group(access_group="prod-models", user_api_key_dict=_admin(), auth_cache=cache)

    _assert_evicted_after_write(journal, "prod-models", "access_group_budget.delete:prod-models")


@pytest.mark.asyncio
async def test_deleting_an_access_group_that_never_had_a_budget_still_evicts():
    """The cascade's delete finds no row and reports nothing dropped, but the group can still be
    sitting in the cached registry of budgeted groups, so both keys have to go regardless."""
    from litellm.proxy.management_endpoints.model_access_group_management_endpoints import (
        delete_access_group,
    )

    journal: list[str] = []
    prisma = _FakePrismaClient(journal, deployments=[_deployment()])
    cache = _FakeAuthCache(journal)

    never_served_router = MagicMock()
    never_served_router.get_model_ids.return_value = []
    with (
        _proxy(prisma),
        patch("litellm.proxy.proxy_server.llm_router", never_served_router),
        patch(
            "litellm.proxy.management_endpoints.model_access_group_management_endpoints.clear_cache",
            new=AsyncMock(return_value=ReconcileOutcome(still_desired=None, live_after=None)),
        ),
    ):
        response = await delete_access_group(
            access_group="prod-models", user_api_key_dict=_admin(), auth_cache=cache
        )

    assert response.models_updated == 1
    assert prisma.access_group_budget_table.rows == {}
    _assert_evicted_after_write(journal, "prod-models", "access_group_budget.delete:prod-models")
