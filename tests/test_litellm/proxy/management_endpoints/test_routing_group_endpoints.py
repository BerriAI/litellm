"""
Unit tests for routing group management endpoints.
"""

import os
import sys
import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import litellm.proxy.proxy_server as ps
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.proxy_server import app
from litellm.types.router import (
    RoutingGroupConfig,
    RoutingGroupDeployment,
)

sys.path.insert(0, os.path.abspath("../../../"))


def _make_routing_group_record(
    routing_group_id: str = "rg-123",
    routing_group_name: str = "test-group",
    routing_strategy: str = "simple-shuffle",
    deployments: list | None = None,
    description: str | None = None,
    fallback_config: dict | None = None,
    retry_config: dict | None = None,
    cooldown_config: dict | None = None,
    settings: dict | None = None,
    assigned_team_ids: list | None = None,
    assigned_key_ids: list | None = None,
    is_active: bool = True,
):
    data = {
        "routing_group_id": routing_group_id,
        "routing_group_name": routing_group_name,
        "routing_strategy": routing_strategy,
        "deployments": deployments or [],
        "description": description,
        "fallback_config": fallback_config or {},
        "retry_config": retry_config or {},
        "cooldown_config": cooldown_config or {},
        "settings": settings or {},
        "assigned_team_ids": assigned_team_ids or [],
        "assigned_key_ids": assigned_key_ids or [],
        "is_active": is_active,
    }
    record = MagicMock()
    for k, v in data.items():
        setattr(record, k, v)
    return record


@pytest.fixture
def client_and_mocks(monkeypatch):
    """Setup mock prisma and auth for routing group endpoints."""
    mock_routing_group_table = MagicMock()
    mock_prisma = MagicMock()

    def _create_side_effect(*, data):
        return _make_routing_group_record(
            routing_group_id=data.get("routing_group_id", "rg-new"),
            routing_group_name=data.get("routing_group_name", "new-group"),
            routing_strategy=data.get("routing_strategy", "simple-shuffle"),
            deployments=data.get("deployments", []),
            description=data.get("description"),
            assigned_team_ids=data.get("assigned_team_ids", []),
            assigned_key_ids=data.get("assigned_key_ids", []),
            is_active=data.get("is_active", True),
        )

    mock_routing_group_table.create = AsyncMock(side_effect=_create_side_effect)
    mock_routing_group_table.find_unique = AsyncMock(return_value=None)
    mock_routing_group_table.find_many = AsyncMock(return_value=[])
    mock_routing_group_table.count = AsyncMock(return_value=0)
    mock_routing_group_table.update = AsyncMock(
        side_effect=lambda *, where, data: _make_routing_group_record(
            routing_group_id=where.get("routing_group_id", "rg-123"),
            routing_group_name=data.get("routing_group_name", "updated-group"),
            routing_strategy=data.get("routing_strategy", "simple-shuffle"),
            deployments=data.get("deployments", []),
            description=data.get("description"),
            assigned_team_ids=data.get("assigned_team_ids", []),
            assigned_key_ids=data.get("assigned_key_ids", []),
            is_active=data.get("is_active", True),
        )
    )
    mock_routing_group_table.delete = AsyncMock(return_value=None)

    mock_db = types.SimpleNamespace(
        litellm_routinggrouptable=mock_routing_group_table,
    )
    mock_prisma.db = mock_db

    monkeypatch.setattr(ps, "prisma_client", mock_prisma)

    user = UserAPIKeyAuth(user_id="test-user")
    app.dependency_overrides[ps.user_api_key_auth] = lambda: user

    # Mock the router sync so tests don't need a live LLM router
    mock_llm_router = MagicMock()
    mock_llm_router.fallbacks = []
    mock_llm_router.add_deployment = MagicMock()
    monkeypatch.setattr(ps, "llm_router", mock_llm_router)

    client = TestClient(app)

    yield client, mock_prisma, mock_routing_group_table, mock_llm_router

    app.dependency_overrides.clear()
    monkeypatch.setattr(ps, "prisma_client", ps.prisma_client)


# ---------------------------------------------------------------------------
# CREATE
# ---------------------------------------------------------------------------


def test_create_routing_group_success(client_and_mocks):
    client, _, mock_table, _ = client_and_mocks

    payload = {
        "routing_group_name": "my-group",
        "routing_strategy": "simple-shuffle",
        "deployments": [
            {
                "model_id": "id-1",
                "model_name": "gpt-4o",
                "provider": "openai",
            }
        ],
    }
    resp = client.post("/v1/routing_group", json=payload)
    assert resp.status_code == 201
    body = resp.json()
    assert body["routing_group_name"] == "my-group"
    mock_table.create.assert_awaited_once()


def test_create_routing_group_invalid_strategy(client_and_mocks):
    client, _, _, _ = client_and_mocks

    payload = {
        "routing_group_name": "bad-group",
        "routing_strategy": "not-a-real-strategy",
        "deployments": [
            {
                "model_id": "id-1",
                "model_name": "gpt-4o",
                "provider": "openai",
            }
        ],
    }
    resp = client.post("/v1/routing_group", json=payload)
    assert resp.status_code == 400
    assert "Invalid routing_strategy" in resp.json()["detail"]


def test_create_routing_group_empty_deployments(client_and_mocks):
    client, _, _, _ = client_and_mocks

    payload = {
        "routing_group_name": "empty-group",
        "routing_strategy": "simple-shuffle",
        "deployments": [],
    }
    resp = client.post("/v1/routing_group", json=payload)
    assert resp.status_code == 400
    assert "deployments" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# LIST
# ---------------------------------------------------------------------------


def test_list_routing_groups_empty(client_and_mocks):
    client, _, _, _ = client_and_mocks

    resp = client.get("/v1/routing_group")
    assert resp.status_code == 200
    body = resp.json()
    assert body["routing_groups"] == []
    assert body["total"] == 0


def test_list_routing_groups_with_records(client_and_mocks):
    client, _, mock_table, _ = client_and_mocks

    records = [
        _make_routing_group_record(
            routing_group_id="rg-1",
            routing_group_name="group-1",
            deployments=[
                {"model_id": "id-1", "model_name": "gpt-4o", "provider": "openai"}
            ],
        ),
        _make_routing_group_record(
            routing_group_id="rg-2",
            routing_group_name="group-2",
            deployments=[
                {
                    "model_id": "id-2",
                    "model_name": "claude-3-5-sonnet-20241022",
                    "provider": "anthropic",
                }
            ],
        ),
    ]
    mock_table.find_many = AsyncMock(return_value=records)
    mock_table.count = AsyncMock(return_value=2)

    resp = client.get("/v1/routing_group")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 2
    assert len(body["routing_groups"]) == 2


# ---------------------------------------------------------------------------
# GET
# ---------------------------------------------------------------------------


def test_get_routing_group_not_found(client_and_mocks):
    client, _, mock_table, _ = client_and_mocks
    mock_table.find_unique = AsyncMock(return_value=None)

    resp = client.get("/v1/routing_group/nonexistent-id")
    assert resp.status_code == 404


def test_get_routing_group_success(client_and_mocks):
    client, _, mock_table, _ = client_and_mocks

    record = _make_routing_group_record(
        routing_group_id="rg-abc",
        routing_group_name="found-group",
        deployments=[
            {"model_id": "id-1", "model_name": "gpt-4o", "provider": "openai"}
        ],
    )
    mock_table.find_unique = AsyncMock(return_value=record)

    resp = client.get("/v1/routing_group/rg-abc")
    assert resp.status_code == 200
    body = resp.json()
    assert body["routing_group_id"] == "rg-abc"
    assert body["routing_group_name"] == "found-group"


# ---------------------------------------------------------------------------
# UPDATE
# ---------------------------------------------------------------------------


def test_update_routing_group_success(client_and_mocks):
    client, _, mock_table, _ = client_and_mocks

    payload = {
        "routing_group_name": "updated-group",
        "routing_strategy": "latency-based-routing",
        "deployments": [
            {"model_id": "id-1", "model_name": "gpt-4o", "provider": "openai"}
        ],
    }
    resp = client.put("/v1/routing_group/rg-123", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["routing_group_name"] == "updated-group"
    mock_table.update.assert_awaited_once()


def test_update_routing_group_db_error_returns_404(client_and_mocks):
    client, _, mock_table, _ = client_and_mocks

    mock_table.update = AsyncMock(side_effect=Exception("Record not found"))

    payload = {
        "routing_group_name": "updated-group",
        "routing_strategy": "simple-shuffle",
        "deployments": [
            {"model_id": "id-1", "model_name": "gpt-4o", "provider": "openai"}
        ],
    }
    resp = client.put("/v1/routing_group/bad-id", json=payload)
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# DELETE
# ---------------------------------------------------------------------------


def test_delete_routing_group_success(client_and_mocks):
    client, _, mock_table, _ = client_and_mocks

    resp = client.delete("/v1/routing_group/rg-123")
    assert resp.status_code == 200
    body = resp.json()
    assert body["deleted"] is True
    assert body["routing_group_id"] == "rg-123"
    mock_table.delete.assert_awaited_once()


def test_delete_routing_group_not_found(client_and_mocks):
    client, _, mock_table, _ = client_and_mocks
    mock_table.delete = AsyncMock(side_effect=Exception("Record does not exist"))

    resp = client.delete("/v1/routing_group/missing-id")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# _sync_routing_group_to_router unit tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sync_priority_failover_wires_fallbacks():
    """priority-failover strategy should create tiered groups and wire fallback chain."""
    from litellm.proxy.management_endpoints.routing_group_endpoints import (
        _sync_routing_group_to_router,
    )

    mock_router = MagicMock()
    mock_router.fallbacks = []
    mock_router.add_deployment = MagicMock()

    config = RoutingGroupConfig(
        routing_group_name="my-failover-group",
        routing_strategy="priority-failover",
        deployments=[
            RoutingGroupDeployment(
                model_id="id-1",
                model_name="gpt-4o",
                provider="openai",
                priority=1,
            ),
            RoutingGroupDeployment(
                model_id="id-2",
                model_name="claude-3-5-sonnet-20241022",
                provider="anthropic",
                priority=2,
            ),
        ],
    )

    with patch(
        "litellm.proxy.management_endpoints.routing_group_endpoints._get_llm_router",
        return_value=mock_router,
    ):
        await _sync_routing_group_to_router(config)

    # add_deployment should be called twice (once per deployment)
    assert mock_router.add_deployment.call_count == 2

    # Fallback chain should be wired: primary -> [fallback_p2]
    assert len(mock_router.fallbacks) == 1
    fallback_entry = mock_router.fallbacks[0]
    assert "my-failover-group" in fallback_entry
    assert fallback_entry["my-failover-group"] == ["my-failover-group__fallback_p2"]


@pytest.mark.asyncio
async def test_sync_weighted_includes_weight_in_litellm_params():
    """weighted strategy should pass weight into litellm_params."""
    from litellm.proxy.management_endpoints.routing_group_endpoints import (
        _sync_routing_group_to_router,
    )

    captured_deployments = []

    mock_router = MagicMock()
    mock_router.fallbacks = []

    def capture_add_deployment(deployment):
        captured_deployments.append(deployment)

    mock_router.add_deployment = MagicMock(side_effect=capture_add_deployment)

    config = RoutingGroupConfig(
        routing_group_name="weighted-group",
        routing_strategy="weighted",
        deployments=[
            RoutingGroupDeployment(
                model_id="id-1",
                model_name="gpt-4o",
                provider="openai",
                weight=80,
            ),
            RoutingGroupDeployment(
                model_id="id-2",
                model_name="claude-3-5-sonnet-20241022",
                provider="anthropic",
                weight=20,
            ),
        ],
    )

    with patch(
        "litellm.proxy.management_endpoints.routing_group_endpoints._get_llm_router",
        return_value=mock_router,
    ):
        await _sync_routing_group_to_router(config)

    assert mock_router.add_deployment.call_count == 2

    # Each captured deployment should have weight in its litellm_params
    for dep in captured_deployments:
        assert hasattr(dep, "litellm_params")
        params = dep.litellm_params
        # litellm_params is a pydantic model; access as dict or attribute
        params_dict = (
            params.model_dump() if hasattr(params, "model_dump") else dict(params)
        )
        assert "weight" in params_dict
        assert params_dict["weight"] in (80, 20)


@pytest.mark.asyncio
async def test_sync_no_router_logs_warning():
    """When llm_router is None, _sync should log a warning and return gracefully."""
    from litellm.proxy.management_endpoints.routing_group_endpoints import (
        _sync_routing_group_to_router,
    )

    config = RoutingGroupConfig(
        routing_group_name="no-router-group",
        routing_strategy="simple-shuffle",
        deployments=[
            RoutingGroupDeployment(
                model_id="id-1",
                model_name="gpt-4o",
                provider="openai",
            )
        ],
    )

    with patch(
        "litellm.proxy.management_endpoints.routing_group_endpoints._get_llm_router",
        return_value=None,
    ):
        # Should not raise
        await _sync_routing_group_to_router(config)


@pytest.mark.asyncio
async def test_sync_simple_shuffle_no_fallbacks():
    """Non-priority-failover strategy should not modify router.fallbacks."""
    from litellm.proxy.management_endpoints.routing_group_endpoints import (
        _sync_routing_group_to_router,
    )

    mock_router = MagicMock()
    mock_router.fallbacks = []
    mock_router.add_deployment = MagicMock()

    config = RoutingGroupConfig(
        routing_group_name="shuffle-group",
        routing_strategy="simple-shuffle",
        deployments=[
            RoutingGroupDeployment(
                model_id="id-1",
                model_name="gpt-4o",
                provider="openai",
            ),
            RoutingGroupDeployment(
                model_id="id-2",
                model_name="gpt-4o-mini",
                provider="openai",
            ),
        ],
    )

    with patch(
        "litellm.proxy.management_endpoints.routing_group_endpoints._get_llm_router",
        return_value=mock_router,
    ):
        await _sync_routing_group_to_router(config)

    assert mock_router.add_deployment.call_count == 2
    # fallbacks should remain unchanged (empty)
    assert mock_router.fallbacks == []
