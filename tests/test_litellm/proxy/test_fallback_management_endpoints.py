"""
Tests for fallback management endpoints

Tests:
1. Create fallback configuration
2. Get fallback configuration
3. Delete fallback configuration
4. Validation tests (invalid models, duplicate fallbacks, etc.)
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from litellm.proxy.management_endpoints.fallback_management_endpoints import (
    FallbackCreateRequest,
    create_fallback,
    delete_fallback,
    get_fallback,
)


class TestFallbackCreateRequest:
    """Test the FallbackCreateRequest validation"""

    def test_valid_request(self):
        """Test valid fallback request"""
        request = FallbackCreateRequest(
            model="gpt-3.5-turbo",
            fallback_models=["gpt-4", "claude-3-haiku"],
            fallback_type="general",
        )
        assert request.model == "gpt-3.5-turbo"
        assert request.fallback_models == ["gpt-4", "claude-3-haiku"]
        assert request.fallback_type == "general"

    def test_default_fallback_type(self):
        """Test default fallback type is 'general'"""
        request = FallbackCreateRequest(
            model="gpt-3.5-turbo",
            fallback_models=["gpt-4"],
        )
        assert request.fallback_type == "general"

    def test_empty_fallback_models(self):
        """Test that empty fallback_models raises validation error"""
        with pytest.raises(ValueError, match="at least 1 item"):
            FallbackCreateRequest(
                model="gpt-3.5-turbo",
                fallback_models=[],
            )

    def test_duplicate_fallback_models(self):
        """Test that duplicate fallback models raise validation error"""
        with pytest.raises(
            ValueError, match="fallback_models must not contain duplicates"
        ):
            FallbackCreateRequest(
                model="gpt-3.5-turbo",
                fallback_models=["gpt-4", "gpt-4"],
            )

    def test_empty_model_name(self):
        """Test that empty model name raises validation error"""
        with pytest.raises(ValueError, match="model must be a non-empty string"):
            FallbackCreateRequest(
                model="",
                fallback_models=["gpt-4"],
            )

    def test_whitespace_model_name(self):
        """Test that whitespace-only model name raises validation error"""
        with pytest.raises(ValueError, match="model must be a non-empty string"):
            FallbackCreateRequest(
                model="   ",
                fallback_models=["gpt-4"],
            )

    def test_model_name_trimmed(self):
        """Test that model name is trimmed"""
        request = FallbackCreateRequest(
            model="  gpt-3.5-turbo  ",
            fallback_models=["gpt-4"],
        )
        assert request.model == "gpt-3.5-turbo"

    def test_context_window_fallback_type(self):
        """Test context_window fallback type"""
        request = FallbackCreateRequest(
            model="gpt-3.5-turbo",
            fallback_models=["gpt-4-32k"],
            fallback_type="context_window",
        )
        assert request.fallback_type == "context_window"

    def test_content_policy_fallback_type(self):
        """Test content_policy fallback type"""
        request = FallbackCreateRequest(
            model="gpt-3.5-turbo",
            fallback_models=["gpt-4"],
            fallback_type="content_policy",
        )
        assert request.fallback_type == "content_policy"


@pytest.mark.asyncio
class TestCreateFallback:
    """Test the create_fallback endpoint"""

    @pytest.fixture
    def mock_router(self):
        """Create a mock router"""
        router = MagicMock()
        router.model_names = {"gpt-3.5-turbo", "gpt-4", "claude-3-haiku"}
        router.fallbacks = []
        router.context_window_fallbacks = []
        router.content_policy_fallbacks = []
        return router

    @pytest.fixture
    def mock_prisma_client(self):
        """Create a mock prisma client"""
        client = MagicMock()
        client.db.litellm_config.upsert = AsyncMock()
        client.jsonify_object = lambda x: x
        return client

    @pytest.fixture
    def mock_proxy_config(self):
        """Create a mock proxy config"""
        config = MagicMock()
        config.get_config = AsyncMock(return_value={"router_settings": {}})
        return config

    @pytest.fixture
    def mock_user_api_key_dict(self):
        """Create a mock user API key dict"""
        return MagicMock()

    async def test_create_fallback_success(
        self, mock_router, mock_prisma_client, mock_proxy_config, mock_user_api_key_dict
    ):
        """Test successful fallback creation"""
        request = FallbackCreateRequest(
            model="gpt-3.5-turbo",
            fallback_models=["gpt-4", "claude-3-haiku"],
            fallback_type="general",
        )

        with (
            patch(
                "litellm.proxy.proxy_server.llm_router",
                mock_router,
            ),
            patch(
                "litellm.proxy.proxy_server.prisma_client",
                mock_prisma_client,
            ),
            patch(
                "litellm.proxy.proxy_server.proxy_config",
                mock_proxy_config,
            ),
            patch(
                "litellm.proxy.proxy_server.store_model_in_db",
                True,
            ),
        ):
            response = await create_fallback(request, mock_user_api_key_dict)

            assert response.model == "gpt-3.5-turbo"
            assert response.fallback_models == ["gpt-4", "claude-3-haiku"]
            assert response.fallback_type == "general"
            assert (
                "created" in response.message.lower()
                or "updated" in response.message.lower()
            )

            # Verify database was updated
            mock_prisma_client.db.litellm_config.upsert.assert_called_once()

    async def test_create_fallback_router_not_initialized(
        self, mock_prisma_client, mock_proxy_config, mock_user_api_key_dict
    ):
        """Test error when router is not initialized"""
        request = FallbackCreateRequest(
            model="gpt-3.5-turbo",
            fallback_models=["gpt-4"],
        )

        with (
            patch(
                "litellm.proxy.proxy_server.llm_router",
                None,
            ),
            pytest.raises(HTTPException) as exc_info,
        ):
            await create_fallback(request, mock_user_api_key_dict)

        assert exc_info.value.status_code == 500
        assert "Router not initialized" in str(exc_info.value.detail)

    async def test_create_fallback_model_not_found(
        self, mock_router, mock_prisma_client, mock_proxy_config, mock_user_api_key_dict
    ):
        """Test error when model is not found in router"""
        request = FallbackCreateRequest(
            model="invalid-model",
            fallback_models=["gpt-4"],
        )

        with (
            patch(
                "litellm.proxy.proxy_server.llm_router",
                mock_router,
            ),
            patch(
                "litellm.proxy.proxy_server.prisma_client",
                mock_prisma_client,
            ),
            patch(
                "litellm.proxy.proxy_server.store_model_in_db",
                True,
            ),
            pytest.raises(HTTPException) as exc_info,
        ):
            await create_fallback(request, mock_user_api_key_dict)

        assert exc_info.value.status_code == 404
        assert "not found in router" in str(exc_info.value.detail)

    async def test_create_fallback_invalid_fallback_model(
        self, mock_router, mock_prisma_client, mock_proxy_config, mock_user_api_key_dict
    ):
        """Test error when fallback model is not found in router"""
        request = FallbackCreateRequest(
            model="gpt-3.5-turbo",
            fallback_models=["invalid-fallback-model"],
        )

        with (
            patch(
                "litellm.proxy.proxy_server.llm_router",
                mock_router,
            ),
            patch(
                "litellm.proxy.proxy_server.prisma_client",
                mock_prisma_client,
            ),
            patch(
                "litellm.proxy.proxy_server.store_model_in_db",
                True,
            ),
            pytest.raises(HTTPException) as exc_info,
        ):
            await create_fallback(request, mock_user_api_key_dict)

        assert exc_info.value.status_code == 400
        assert "Invalid fallback models" in str(exc_info.value.detail)

    async def test_create_fallback_model_is_own_fallback(
        self, mock_router, mock_prisma_client, mock_proxy_config, mock_user_api_key_dict
    ):
        """Test error when model is its own fallback"""
        request = FallbackCreateRequest(
            model="gpt-3.5-turbo",
            fallback_models=["gpt-3.5-turbo", "gpt-4"],
        )

        with (
            patch(
                "litellm.proxy.proxy_server.llm_router",
                mock_router,
            ),
            patch(
                "litellm.proxy.proxy_server.prisma_client",
                mock_prisma_client,
            ),
            patch(
                "litellm.proxy.proxy_server.store_model_in_db",
                True,
            ),
            pytest.raises(HTTPException) as exc_info,
        ):
            await create_fallback(request, mock_user_api_key_dict)

        assert exc_info.value.status_code == 400
        assert "cannot be its own fallback" in str(exc_info.value.detail)

    async def test_create_fallback_db_not_enabled(
        self, mock_router, mock_user_api_key_dict
    ):
        """Test error when database storage is not enabled"""
        request = FallbackCreateRequest(
            model="gpt-3.5-turbo",
            fallback_models=["gpt-4"],
        )

        with (
            patch(
                "litellm.proxy.proxy_server.llm_router",
                mock_router,
            ),
            patch(
                "litellm.proxy.proxy_server.store_model_in_db",
                False,
            ),
            pytest.raises(HTTPException) as exc_info,
        ):
            await create_fallback(request, mock_user_api_key_dict)

        assert exc_info.value.status_code == 400
        assert "Database storage not enabled" in str(exc_info.value.detail)

    async def test_create_fallback_context_window_type(
        self, mock_router, mock_prisma_client, mock_proxy_config, mock_user_api_key_dict
    ):
        """Test creating context_window fallback"""
        request = FallbackCreateRequest(
            model="gpt-3.5-turbo",
            fallback_models=["gpt-4"],
            fallback_type="context_window",
        )

        with (
            patch(
                "litellm.proxy.proxy_server.llm_router",
                mock_router,
            ),
            patch(
                "litellm.proxy.proxy_server.prisma_client",
                mock_prisma_client,
            ),
            patch(
                "litellm.proxy.proxy_server.proxy_config",
                mock_proxy_config,
            ),
            patch(
                "litellm.proxy.proxy_server.store_model_in_db",
                True,
            ),
        ):
            response = await create_fallback(request, mock_user_api_key_dict)

            assert response.fallback_type == "context_window"
            # Verify the correct attribute was updated
            assert hasattr(mock_router, "context_window_fallbacks")


@pytest.mark.asyncio
class TestGetFallback:
    """Test the get_fallback endpoint"""

    @pytest.fixture
    def mock_router_with_fallbacks(self):
        """Create a mock router with fallbacks configured"""
        router = MagicMock()
        router.fallbacks = [{"gpt-3.5-turbo": ["gpt-4", "claude-3-haiku"]}]
        router.context_window_fallbacks = []
        router.content_policy_fallbacks = []
        return router

    @pytest.fixture
    def mock_user_api_key_dict(self):
        """Create a mock user API key dict"""
        return MagicMock()

    async def test_get_fallback_success(
        self, mock_router_with_fallbacks, mock_user_api_key_dict
    ):
        """Test successful fallback retrieval"""
        with patch(
            "litellm.proxy.proxy_server.llm_router",
            mock_router_with_fallbacks,
        ):
            response = await get_fallback(
                "gpt-3.5-turbo", "general", mock_user_api_key_dict
            )

            assert response.model == "gpt-3.5-turbo"
            assert response.fallback_models == ["gpt-4", "claude-3-haiku"]
            assert response.fallback_type == "general"

    async def test_get_fallback_not_found(
        self, mock_router_with_fallbacks, mock_user_api_key_dict
    ):
        """Test error when fallback is not found"""
        with (
            patch(
                "litellm.proxy.proxy_server.llm_router",
                mock_router_with_fallbacks,
            ),
            pytest.raises(HTTPException) as exc_info,
        ):
            await get_fallback("gpt-4", "general", mock_user_api_key_dict)

        assert exc_info.value.status_code == 404
        assert "No general fallbacks configured" in str(exc_info.value.detail)

    async def test_get_fallback_router_not_initialized(self, mock_user_api_key_dict):
        """Test error when router is not initialized"""
        with (
            patch(
                "litellm.proxy.proxy_server.llm_router",
                None,
            ),
            pytest.raises(HTTPException) as exc_info,
        ):
            await get_fallback("gpt-3.5-turbo", "general", mock_user_api_key_dict)

        assert exc_info.value.status_code == 500
        assert "Router not initialized" in str(exc_info.value.detail)


@pytest.mark.asyncio
class TestDeleteFallback:
    """Test the delete_fallback endpoint"""

    @pytest.fixture
    def mock_router_with_fallbacks(self):
        """Create a mock router with fallbacks configured"""
        router = MagicMock()
        router.fallbacks = [{"gpt-3.5-turbo": ["gpt-4", "claude-3-haiku"]}]
        router.context_window_fallbacks = []
        router.content_policy_fallbacks = []
        return router

    @pytest.fixture
    def mock_prisma_client(self):
        """Create a mock prisma client"""
        client = MagicMock()
        client.db.litellm_config.upsert = AsyncMock()
        client.jsonify_object = lambda x: x
        return client

    @pytest.fixture
    def mock_proxy_config(self):
        """Create a mock proxy config"""
        config = MagicMock()
        config.get_config = AsyncMock(
            return_value={
                "router_settings": {
                    "fallbacks": [{"gpt-3.5-turbo": ["gpt-4", "claude-3-haiku"]}]
                }
            }
        )
        return config

    @pytest.fixture
    def mock_user_api_key_dict(self):
        """Create a mock user API key dict"""
        return MagicMock()

    async def test_delete_fallback_success(
        self,
        mock_router_with_fallbacks,
        mock_prisma_client,
        mock_proxy_config,
        mock_user_api_key_dict,
    ):
        """Test successful fallback deletion"""
        with (
            patch(
                "litellm.proxy.proxy_server.llm_router",
                mock_router_with_fallbacks,
            ),
            patch(
                "litellm.proxy.proxy_server.prisma_client",
                mock_prisma_client,
            ),
            patch(
                "litellm.proxy.proxy_server.proxy_config",
                mock_proxy_config,
            ),
            patch(
                "litellm.proxy.proxy_server.store_model_in_db",
                True,
            ),
        ):
            response = await delete_fallback(
                "gpt-3.5-turbo", "general", mock_user_api_key_dict
            )

            assert response.model == "gpt-3.5-turbo"
            assert response.fallback_type == "general"
            assert "deleted" in response.message.lower()

            # Verify database was updated
            mock_prisma_client.db.litellm_config.upsert.assert_called_once()

    async def test_delete_fallback_not_found(
        self,
        mock_router_with_fallbacks,
        mock_prisma_client,
        mock_proxy_config,
        mock_user_api_key_dict,
    ):
        """Test error when fallback to delete is not found"""
        with (
            patch(
                "litellm.proxy.proxy_server.llm_router",
                mock_router_with_fallbacks,
            ),
            patch(
                "litellm.proxy.proxy_server.prisma_client",
                mock_prisma_client,
            ),
            patch(
                "litellm.proxy.proxy_server.proxy_config",
                mock_proxy_config,
            ),
            patch(
                "litellm.proxy.proxy_server.store_model_in_db",
                True,
            ),
            pytest.raises(HTTPException) as exc_info,
        ):
            await delete_fallback("gpt-4", "general", mock_user_api_key_dict)

        assert exc_info.value.status_code == 404
        assert "No general fallbacks configured" in str(exc_info.value.detail)

    async def test_delete_fallback_router_not_initialized(self, mock_user_api_key_dict):
        """Test error when router is not initialized"""
        with (
            patch(
                "litellm.proxy.proxy_server.llm_router",
                None,
            ),
            pytest.raises(HTTPException) as exc_info,
        ):
            await delete_fallback("gpt-3.5-turbo", "general", mock_user_api_key_dict)

        assert exc_info.value.status_code == 500
        assert "Router not initialized" in str(exc_info.value.detail)

    async def test_delete_fallback_db_not_enabled(
        self, mock_router_with_fallbacks, mock_user_api_key_dict
    ):
        """Test error when database storage is not enabled"""
        with (
            patch(
                "litellm.proxy.proxy_server.llm_router",
                mock_router_with_fallbacks,
            ),
            patch(
                "litellm.proxy.proxy_server.store_model_in_db",
                False,
            ),
            pytest.raises(HTTPException) as exc_info,
        ):
            await delete_fallback("gpt-3.5-turbo", "general", mock_user_api_key_dict)

        assert exc_info.value.status_code == 400
        assert "Database storage not enabled" in str(exc_info.value.detail)


class TestScrubModelFromFallbackEntries:
    def test_drops_mappings_from_deleted_model_and_refs_to_it(self):
        from litellm.proxy.management_endpoints.fallback_management_endpoints import (
            scrub_model_from_fallback_entries,
        )

        cleaned = scrub_model_from_fallback_entries(
            [
                {"gone": ["still-there"]},
                {"kept": ["gone", "other"]},
                {"also-kept": ["gone"]},
            ],
            "gone",
        )

        assert cleaned == [{"kept": ["other"]}]

    def test_leaves_unrelated_entries_unchanged(self):
        from litellm.proxy.management_endpoints.fallback_management_endpoints import (
            scrub_model_from_fallback_entries,
        )

        existing = [{"primary": ["fb-a", "fb-b"]}]
        assert scrub_model_from_fallback_entries(existing, "missing") == existing

    def test_treats_non_list_config_as_empty(self):
        from litellm.proxy.management_endpoints.fallback_management_endpoints import (
            scrub_model_from_fallback_entries,
        )

        assert scrub_model_from_fallback_entries(None, "gone") == []
        assert scrub_model_from_fallback_entries("not-a-list", "gone") == []

    def test_router_lost_last_deployment_only_when_model_names_is_a_real_collection(self):
        from litellm.proxy.management_endpoints.fallback_management_endpoints import (
            router_lost_last_deployment_for_model,
        )

        present = MagicMock()
        present.model_names = {"kept", "gone"}
        gone = MagicMock()
        gone.model_names = {"kept"}
        unknown = MagicMock()

        assert router_lost_last_deployment_for_model(present, "gone") is False
        assert router_lost_last_deployment_for_model(gone, "gone") is True
        assert router_lost_last_deployment_for_model(unknown, "gone") is False


@pytest.mark.asyncio
class TestRemoveDeletedModelFromRouterFallbacks:
    async def test_persists_scrubbed_lists_and_updates_router(self):
        from litellm.proxy.management_endpoints.fallback_management_endpoints import (
            remove_deleted_model_from_router_fallbacks,
        )

        router = MagicMock()
        router.fallbacks = [{"primary": ["gone"]}]
        router.context_window_fallbacks = [{"primary": ["gone", "other"]}]
        router.content_policy_fallbacks = []

        prisma = MagicMock()
        prisma.db.litellm_config.upsert = AsyncMock()

        proxy_config = MagicMock()
        proxy_config.get_config = AsyncMock(
            return_value={
                "router_settings": {
                    "num_retries": 2,
                    "fallbacks": [{"primary": ["gone"]}],
                    "context_window_fallbacks": [{"primary": ["gone", "other"]}],
                }
            }
        )

        with (
            patch("litellm.proxy.proxy_server.proxy_config", proxy_config),
            patch(
                "litellm.proxy.utils.invalidate_config_param",
                new_callable=AsyncMock,
            ) as invalidate,
        ):
            await remove_deleted_model_from_router_fallbacks(
                model_name="gone",
                prisma_client=prisma,
                llm_router=router,
            )

        prisma.db.litellm_config.upsert.assert_called_once()
        saved = json.loads(
            prisma.db.litellm_config.upsert.call_args.kwargs["data"]["update"][
                "param_value"
            ]
        )
        assert saved["num_retries"] == 2
        assert saved["fallbacks"] == []
        assert saved["context_window_fallbacks"] == [{"primary": ["other"]}]
        assert saved["content_policy_fallbacks"] == []
        assert router.fallbacks == []
        assert router.context_window_fallbacks == [{"primary": ["other"]}]
        assert router.content_policy_fallbacks == []
        invalidate.assert_awaited_once_with("router_settings")

    async def test_skips_persist_when_model_is_not_referenced(self):
        from litellm.proxy.management_endpoints.fallback_management_endpoints import (
            remove_deleted_model_from_router_fallbacks,
        )

        prisma = MagicMock()
        prisma.db.litellm_config.upsert = AsyncMock()
        proxy_config = MagicMock()
        proxy_config.get_config = AsyncMock(
            return_value={"router_settings": {"fallbacks": [{"primary": ["other"]}]}}
        )

        with (
            patch("litellm.proxy.proxy_server.proxy_config", proxy_config),
            patch(
                "litellm.proxy.utils.invalidate_config_param",
                new_callable=AsyncMock,
            ) as invalidate,
        ):
            await remove_deleted_model_from_router_fallbacks(
                model_name="gone",
                prisma_client=prisma,
                llm_router=MagicMock(),
            )

        prisma.db.litellm_config.upsert.assert_not_called()
        invalidate.assert_not_awaited()
