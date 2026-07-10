"""
Tests for fallback management endpoints

Tests:
1. Create fallback configuration
2. Get fallback configuration
3. Delete fallback configuration
4. Validation tests (invalid models, duplicate fallbacks, etc.)
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
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


class TestFallbackRoutingWithSlashInModelName:
    """
    Regression tests for GET/DELETE /fallback/{model:path} route matching.

    Model names containing "/" (e.g. "openrouter/gpt-4") previously returned
    Starlette's default 404 because {model} only matches a single path segment.
    These tests go through actual HTTP routing, unlike the direct handler
    tests above.
    """

    @pytest.fixture
    def mock_router_with_slash_fallbacks(self):
        router = MagicMock()
        router.fallbacks = [{"openrouter/gpt-4": ["gpt-4", "claude-3-haiku"]}]
        router.context_window_fallbacks = []
        router.content_policy_fallbacks = []
        return router

    @pytest.fixture
    def client(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from litellm.proxy.management_endpoints.fallback_management_endpoints import (
            router as fallback_router,
        )

        app = FastAPI()
        app.include_router(fallback_router)
        app.dependency_overrides[user_api_key_auth] = lambda: MagicMock()
        return TestClient(app)

    @pytest.mark.parametrize("path", ["/fallback/openrouter%2Fgpt-4", "/fallback/openrouter/gpt-4"])
    def test_get_fallback_model_with_slash(self, client, mock_router_with_slash_fallbacks, path):
        with patch(
            "litellm.proxy.proxy_server.llm_router",
            mock_router_with_slash_fallbacks,
        ):
            response = client.get(path)

        assert response.status_code == 200
        assert response.json()["model"] == "openrouter/gpt-4"
        assert response.json()["fallback_models"] == ["gpt-4", "claude-3-haiku"]

    def test_delete_fallback_model_with_slash(self, client, mock_router_with_slash_fallbacks):
        mock_prisma_client = MagicMock()
        mock_prisma_client.db.litellm_config.upsert = AsyncMock()
        mock_proxy_config = MagicMock()
        mock_proxy_config.get_config = AsyncMock(
            return_value={
                "router_settings": {
                    "fallbacks": [{"openrouter/gpt-4": ["gpt-4", "claude-3-haiku"]}]
                }
            }
        )

        with (
            patch(
                "litellm.proxy.proxy_server.llm_router",
                mock_router_with_slash_fallbacks,
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
            response = client.delete("/fallback/openrouter%2Fgpt-4")

        assert response.status_code == 200
        assert response.json()["model"] == "openrouter/gpt-4"
        mock_prisma_client.db.litellm_config.upsert.assert_called_once()

    def test_get_fallback_model_without_slash_still_works(self, client, mock_router_with_slash_fallbacks):
        mock_router_with_slash_fallbacks.fallbacks = [{"gpt-3.5-turbo": ["gpt-4"]}]
        with patch(
            "litellm.proxy.proxy_server.llm_router",
            mock_router_with_slash_fallbacks,
        ):
            response = client.get("/fallback/gpt-3.5-turbo")

        assert response.status_code == 200
        assert response.json()["model"] == "gpt-3.5-turbo"
