"""
Test that models stored in the database but not loaded into the router
still appear in the /v2/model/info endpoint response.

This addresses a bug where models added via the UI would be saved to the DB
but fail to sync into the router (e.g. due to decryption errors or invalid
params with ignore_invalid_deployments=True), making them invisible in the
model list.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from litellm.proxy.proxy_server import (
    _apply_search_filter_to_models,
    _get_db_only_models,
)


def _make_router_model(model_name: str, model_id: str, db_model: bool = True):
    """Helper to create a router-format model dict."""
    return {
        "model_name": model_name,
        "litellm_params": {"model": model_name},
        "model_info": {
            "id": model_id,
            "db_model": db_model,
        },
    }


def _make_db_record(model_name: str, model_id: str):
    """Helper to create a mock DB record."""
    record = MagicMock()
    record.model_id = model_id
    record.model_name = model_name
    record.litellm_params = {"model": model_name}
    record.model_info = {"id": model_id}
    record.created_by = "default_user_id"
    return record


class TestGetDbOnlyModels:
    """Test _get_db_only_models fetches models not in router."""

    @pytest.mark.asyncio
    async def test_no_db_only_models(self):
        """When all DB models are in the router, return empty list."""
        router_models = [
            _make_router_model("gpt-4", "id-1", db_model=True),
            _make_router_model("gpt-5", "id-2", db_model=True),
        ]
        prisma_client = MagicMock()
        prisma_client.db.litellm_proxymodeltable.count = AsyncMock(return_value=0)
        proxy_config = MagicMock()

        db_models, count = await _get_db_only_models(
            all_models=router_models,
            prisma_client=prisma_client,
            proxy_config=proxy_config,
        )

        assert db_models == []
        assert count == 0
        # Should have excluded router model IDs in the query
        call_args = prisma_client.db.litellm_proxymodeltable.count.call_args
        where = call_args.kwargs.get("where", {})
        assert "model_id" in where
        assert set(where["model_id"]["not"]["in"]) == {"id-1", "id-2"}

    @pytest.mark.asyncio
    async def test_db_only_models_returned(self):
        """Models in DB but not in router should be fetched and returned."""
        router_models = [
            _make_router_model("gpt-4", "id-1", db_model=True),
        ]
        db_record = _make_db_record("gpt-5.1", "id-missing")

        prisma_client = MagicMock()
        prisma_client.db.litellm_proxymodeltable.count = AsyncMock(return_value=1)
        prisma_client.db.litellm_proxymodeltable.find_many = AsyncMock(
            return_value=[db_record]
        )

        decrypted_model = _make_router_model("gpt-5.1", "id-missing", db_model=True)
        proxy_config = MagicMock()
        proxy_config.decrypt_model_list_from_db = MagicMock(
            return_value=[decrypted_model]
        )

        db_models, count = await _get_db_only_models(
            all_models=router_models,
            prisma_client=prisma_client,
            proxy_config=proxy_config,
        )

        assert count == 1
        assert len(db_models) == 1
        assert db_models[0]["model_name"] == "gpt-5.1"

    @pytest.mark.asyncio
    async def test_search_filter_applied(self):
        """When search is provided, DB query should filter by model name."""
        router_models = []
        prisma_client = MagicMock()
        prisma_client.db.litellm_proxymodeltable.count = AsyncMock(return_value=0)
        proxy_config = MagicMock()

        await _get_db_only_models(
            all_models=router_models,
            prisma_client=prisma_client,
            proxy_config=proxy_config,
            search="gpt-5",
        )

        call_args = prisma_client.db.litellm_proxymodeltable.count.call_args
        where = call_args.kwargs.get("where", {})
        assert "model_name" in where
        assert where["model_name"]["contains"] == "gpt-5"

    @pytest.mark.asyncio
    async def test_config_models_not_excluded(self):
        """Config models (db_model=False) should not be added to the exclusion set."""
        router_models = [
            _make_router_model("gpt-4", "config-id-1", db_model=False),
            _make_router_model("gpt-5", "db-id-1", db_model=True),
        ]
        prisma_client = MagicMock()
        prisma_client.db.litellm_proxymodeltable.count = AsyncMock(return_value=0)
        proxy_config = MagicMock()

        await _get_db_only_models(
            all_models=router_models,
            prisma_client=prisma_client,
            proxy_config=proxy_config,
        )

        call_args = prisma_client.db.litellm_proxymodeltable.count.call_args
        where = call_args.kwargs.get("where", {})
        # Only db_model=True IDs should be excluded
        assert "db-id-1" in where["model_id"]["not"]["in"]
        assert "config-id-1" not in where["model_id"]["not"]["in"]


class TestApplySearchFilterIncludesDbOnlyModels:
    """Test that _apply_search_filter_to_models includes DB-only models."""

    @pytest.mark.asyncio
    async def test_no_search_includes_db_only_models(self):
        """Even without search, DB-only models should be included in results."""
        router_models = [
            _make_router_model("gpt-4", "id-1", db_model=True),
        ]
        db_record = _make_db_record("gpt-5.1", "id-missing")

        prisma_client = MagicMock()
        prisma_client.db.litellm_proxymodeltable.count = AsyncMock(return_value=1)
        prisma_client.db.litellm_proxymodeltable.find_many = AsyncMock(
            return_value=[db_record]
        )

        decrypted_model = _make_router_model("gpt-5.1", "id-missing", db_model=True)
        proxy_config = MagicMock()
        proxy_config.decrypt_model_list_from_db = MagicMock(
            return_value=[decrypted_model]
        )

        result, total_count = await _apply_search_filter_to_models(
            all_models=router_models,
            search="",
            page=1,
            size=50,
            prisma_client=prisma_client,
            proxy_config=proxy_config,
        )

        # Should include both router model and DB-only model
        model_names = [m["model_name"] for m in result]
        assert "gpt-4" in model_names
        assert "gpt-5.1" in model_names
        assert total_count == 2

    @pytest.mark.asyncio
    async def test_no_search_no_db_only_models_returns_none_count(self):
        """When no search and no DB-only models, total_count should be None (original behavior)."""
        router_models = [
            _make_router_model("gpt-4", "id-1", db_model=True),
        ]
        prisma_client = MagicMock()
        prisma_client.db.litellm_proxymodeltable.count = AsyncMock(return_value=0)
        proxy_config = MagicMock()

        result, total_count = await _apply_search_filter_to_models(
            all_models=router_models,
            search="",
            page=1,
            size=50,
            prisma_client=prisma_client,
            proxy_config=proxy_config,
        )

        assert result == router_models
        assert total_count is None  # Preserve original behavior

    @pytest.mark.asyncio
    async def test_search_with_db_only_models(self):
        """Search should filter router models AND include matching DB-only models."""
        router_models = [
            _make_router_model("gpt-4", "id-1", db_model=True),
            _make_router_model("gpt-5", "id-2", db_model=True),
        ]
        db_record = _make_db_record("gpt-5.1", "id-missing")

        prisma_client = MagicMock()
        prisma_client.db.litellm_proxymodeltable.count = AsyncMock(return_value=1)
        prisma_client.db.litellm_proxymodeltable.find_many = AsyncMock(
            return_value=[db_record]
        )

        decrypted_model = _make_router_model("gpt-5.1", "id-missing", db_model=True)
        proxy_config = MagicMock()
        proxy_config.decrypt_model_list_from_db = MagicMock(
            return_value=[decrypted_model]
        )

        result, total_count = await _apply_search_filter_to_models(
            all_models=router_models,
            search="gpt-5",
            page=1,
            size=50,
            prisma_client=prisma_client,
            proxy_config=proxy_config,
        )

        model_names = [m["model_name"] for m in result]
        # gpt-4 should be filtered out by search
        assert "gpt-4" not in model_names
        # gpt-5 (from router) and gpt-5.1 (from DB) should be included
        assert "gpt-5" in model_names
        assert "gpt-5.1" in model_names
        assert total_count == 2

    @pytest.mark.asyncio
    async def test_no_prisma_client_returns_router_only(self):
        """Without prisma_client, should return only router models."""
        router_models = [
            _make_router_model("gpt-4", "id-1", db_model=True),
        ]

        result, total_count = await _apply_search_filter_to_models(
            all_models=router_models,
            search="",
            page=1,
            size=50,
            prisma_client=None,
            proxy_config=MagicMock(),
        )

        assert result == router_models
        assert total_count is None

    @pytest.mark.asyncio
    async def test_search_pagination_uses_filtered_count(self):
        """When searching, pagination should use the filtered router count, not the full count.

        Regression test: if the router has 50 models but only 1 matches search,
        DB-only models should still be fetched (take should be based on 1, not 50).
        """
        # 50 router models, but only "gpt-5" matches the search "gpt-5"
        router_models = [_make_router_model(f"model-{i}", f"id-{i}", db_model=True) for i in range(49)]
        router_models.append(_make_router_model("gpt-5", "id-49", db_model=True))

        db_record = _make_db_record("gpt-5.1", "id-missing")

        prisma_client = MagicMock()
        prisma_client.db.litellm_proxymodeltable.count = AsyncMock(return_value=1)
        prisma_client.db.litellm_proxymodeltable.find_many = AsyncMock(
            return_value=[db_record]
        )

        decrypted_model = _make_router_model("gpt-5.1", "id-missing", db_model=True)
        proxy_config = MagicMock()
        proxy_config.decrypt_model_list_from_db = MagicMock(
            return_value=[decrypted_model]
        )

        result, total_count = await _apply_search_filter_to_models(
            all_models=router_models,
            search="gpt-5",
            page=1,
            size=50,
            prisma_client=prisma_client,
            proxy_config=proxy_config,
        )

        model_names = [m["model_name"] for m in result]
        # Only gpt-5 matches search from router, gpt-5.1 from DB
        assert "gpt-5" in model_names
        assert "gpt-5.1" in model_names
        assert len(result) == 2
        assert total_count == 2
        # Verify find_many was actually called (not skipped due to bad take calculation)
        prisma_client.db.litellm_proxymodeltable.find_many.assert_called_once()
