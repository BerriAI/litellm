import os
import sys
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth

# Import proxy_server module first to ensure it's initialized
import litellm.proxy.proxy_server as ps

# Now we can safely import app
from litellm.proxy.proxy_server import app

client = TestClient(app)


@pytest.mark.asyncio
async def test_list_search_tools_db_only(monkeypatch):
    """Test listing search tools when only DB tools exist"""
    # Mock DB tools
    db_tools = [
        {
            "search_tool_id": "test-id-1",
            "search_tool_name": "db-tool-1",
            "litellm_params": {"search_provider": "perplexity", "api_key": "sk-test"},
            "search_tool_info": {"description": "DB tool 1"},
            "created_at": datetime(2023, 11, 9, 12, 34, 56),
            "updated_at": datetime(2023, 11, 9, 13, 45, 12),
        }
    ]

    # Mock SearchToolRegistry
    mock_registry = MagicMock()
    mock_registry.get_all_search_tools_from_db = AsyncMock(return_value=db_tools)
    with patch(
        "litellm.proxy.search_endpoints.search_tool_management.SEARCH_TOOL_REGISTRY",
        mock_registry,
    ):
        # Mock prisma_client
        mock_prisma = MagicMock()
        with patch("litellm.proxy.proxy_server.prisma_client", mock_prisma):
            # Mock proxy_config
            mock_proxy_config = MagicMock()
            mock_proxy_config.get_config = AsyncMock(return_value={})
            mock_proxy_config.parse_search_tools = MagicMock(return_value=None)
            with patch("litellm.proxy.proxy_server.proxy_config", mock_proxy_config):
                # Mock auth
                from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

                app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(
                    user_role=LitellmUserRoles.PROXY_ADMIN, user_id="admin_user"
                )

                try:
                    test_client = TestClient(app)
                    response = test_client.get("/search_tools/list")
                    assert response.status_code == 200
                    data = response.json()
                    assert "search_tools" in data
                    assert len(data["search_tools"]) == 1

                    tool = data["search_tools"][0]
                    assert tool["search_tool_id"] == "test-id-1"
                    assert tool["search_tool_name"] == "db-tool-1"
                    assert tool["is_from_config"] is False
                    # Verify datetime conversion to ISO string
                    assert tool["created_at"] == "2023-11-09T12:34:56"
                    assert tool["updated_at"] == "2023-11-09T13:45:12"
                    # Verify masking of sensitive values
                    assert tool["litellm_params"]["api_key"] != "sk-test"
                    assert "****" in tool["litellm_params"]["api_key"]
                    assert tool["litellm_params"]["search_provider"] == "perplexity"
                finally:
                    app.dependency_overrides.pop(user_api_key_auth, None)


@pytest.mark.asyncio
async def test_list_search_tools_config_only(monkeypatch):
    """Test listing search tools when only config tools exist"""
    # Mock DB tools - empty
    db_tools = []

    # Mock config tools
    config_tools = [
        {
            "search_tool_name": "config-tool-1",
            "litellm_params": {"search_provider": "tavily", "api_key": "tvly-secret-key"},
            "search_tool_info": {"description": "Config tool 1"},
        }
    ]

    # Mock SearchToolRegistry
    mock_registry = MagicMock()
    mock_registry.get_all_search_tools_from_db = AsyncMock(return_value=db_tools)
    with patch(
        "litellm.proxy.search_endpoints.search_tool_management.SEARCH_TOOL_REGISTRY",
        mock_registry,
    ):
        # Mock prisma_client
        mock_prisma = MagicMock()
        with patch("litellm.proxy.proxy_server.prisma_client", mock_prisma):
            # Mock proxy_config
            mock_proxy_config = MagicMock()
            mock_proxy_config.get_config = AsyncMock(return_value={"search_tools": config_tools})
            mock_proxy_config.parse_search_tools = MagicMock(return_value=config_tools)
            with patch("litellm.proxy.proxy_server.proxy_config", mock_proxy_config):
                # Mock auth
                from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

                app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(
                    user_role=LitellmUserRoles.PROXY_ADMIN, user_id="admin_user"
                )

                try:
                    test_client = TestClient(app)
                    response = test_client.get("/search_tools/list")
                    assert response.status_code == 200
                    data = response.json()
                    assert "search_tools" in data
                    assert len(data["search_tools"]) == 1

                    tool = data["search_tools"][0]
                    assert tool["search_tool_name"] == "config-tool-1"
                    assert tool["is_from_config"] is True
                    assert tool["search_tool_id"] is None
                    assert tool["created_at"] is None
                    assert tool["updated_at"] is None
                    # Verify masking
                    assert "tv****ey" in tool["litellm_params"]["api_key"]
                finally:
                    app.dependency_overrides.pop(user_api_key_auth, None)


@pytest.mark.asyncio
async def test_list_search_tools_filters_duplicate_config_tools(monkeypatch):
    """
    Test that config tools with the same name as DB tools are filtered out.
    This tests the new filtering logic added in lines 139-142.
    """
    # Mock DB tools
    db_tools = [
        {
            "search_tool_id": "db-id-1",
            "search_tool_name": "existing-tool",
            "litellm_params": {"search_provider": "perplexity", "api_key": "sk-db"},
            "search_tool_info": {"description": "DB tool"},
            "created_at": datetime(2023, 11, 9, 12, 34, 56),
            "updated_at": datetime(2023, 11, 9, 13, 45, 12),
        }
    ]

    # Mock config tools - one duplicate, one unique
    config_tools = [
        {
            "search_tool_name": "existing-tool",  # Duplicate - should be filtered
            "litellm_params": {"search_provider": "tavily", "api_key": "tvly-config"},
            "search_tool_info": {"description": "Config tool - duplicate"},
        },
        {
            "search_tool_name": "unique-config-tool",  # Unique - should be included
            "litellm_params": {"search_provider": "tavily", "api_key": "tvly-unique"},
            "search_tool_info": {"description": "Config tool - unique"},
        },
    ]

    # Mock SearchToolRegistry
    mock_registry = MagicMock()
    mock_registry.get_all_search_tools_from_db = AsyncMock(return_value=db_tools)
    with patch(
        "litellm.proxy.search_endpoints.search_tool_management.SEARCH_TOOL_REGISTRY",
        mock_registry,
    ):
        # Mock prisma_client
        mock_prisma = MagicMock()
        with patch("litellm.proxy.proxy_server.prisma_client", mock_prisma):
            # Mock proxy_config
            mock_proxy_config = MagicMock()
            mock_proxy_config.get_config = AsyncMock(return_value={"search_tools": config_tools})
            mock_proxy_config.parse_search_tools = MagicMock(return_value=config_tools)
            with patch("litellm.proxy.proxy_server.proxy_config", mock_proxy_config):
                # Mock auth
                from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

                app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(
                    user_role=LitellmUserRoles.PROXY_ADMIN, user_id="admin_user"
                )

                try:
                    test_client = TestClient(app)
                    response = test_client.get("/search_tools/list")
                    assert response.status_code == 200
                    data = response.json()
                    assert "search_tools" in data
                    # Should have 1 DB tool + 1 unique config tool (duplicate filtered out)
                    assert len(data["search_tools"]) == 2

                    # Verify DB tool is present
                    db_tool = next(
                        (t for t in data["search_tools"] if t["search_tool_name"] == "existing-tool"),
                        None,
                    )
                    assert db_tool is not None
                    assert db_tool["is_from_config"] is False
                    assert db_tool["search_tool_id"] == "db-id-1"
                    # Verify masking of sensitive values in DB tool
                    assert db_tool["litellm_params"]["api_key"] != "sk-db"
                    assert "****" in db_tool["litellm_params"]["api_key"]
                    assert db_tool["litellm_params"]["search_provider"] == "perplexity"

                    # Verify unique config tool is present
                    config_tool = next(
                        (t for t in data["search_tools"] if t["search_tool_name"] == "unique-config-tool"),
                        None,
                    )
                    assert config_tool is not None
                    assert config_tool["is_from_config"] is True

                    # Verify duplicate config tool is NOT present
                    duplicate_tool = next(
                        (
                            t
                            for t in data["search_tools"]
                            if t["search_tool_name"] == "existing-tool" and t["is_from_config"] is True
                        ),
                        None,
                    )
                    assert duplicate_tool is None
                finally:
                    app.dependency_overrides.pop(user_api_key_auth, None)


@pytest.mark.asyncio
async def test_list_search_tools_datetime_conversion(monkeypatch):
    """
    Test that datetime objects in DB tools are properly converted to ISO format strings.
    This tests the new datetime conversion logic using _convert_datetime_to_str.
    """
    # Mock DB tools with datetime objects
    db_tools = [
        {
            "search_tool_id": "test-id-1",
            "search_tool_name": "datetime-test-tool",
            "litellm_params": {"search_provider": "perplexity", "api_key": "sk-test"},
            "search_tool_info": {"description": "Test tool"},
            "created_at": datetime(2024, 1, 15, 10, 30, 45, 123456),
            "updated_at": datetime(2024, 1, 16, 14, 20, 30, 789012),
        },
        {
            "search_tool_id": "test-id-2",
            "search_tool_name": "null-datetime-tool",
            "litellm_params": {"search_provider": "tavily", "api_key": "tvly-test"},
            "search_tool_info": None,
            "created_at": None,
            "updated_at": None,
        },
        {
            "search_tool_id": "test-id-3",
            "search_tool_name": "string-datetime-tool",
            "litellm_params": {"search_provider": "perplexity", "api_key": "sk-test"},
            "search_tool_info": {"description": "Already string"},
            "created_at": "2024-01-17T08:15:00",  # Already a string
            "updated_at": "2024-01-18T09:25:00",  # Already a string
        },
    ]

    # Mock SearchToolRegistry
    mock_registry = MagicMock()
    mock_registry.get_all_search_tools_from_db = AsyncMock(return_value=db_tools)
    with patch(
        "litellm.proxy.search_endpoints.search_tool_management.SEARCH_TOOL_REGISTRY",
        mock_registry,
    ):
        # Mock prisma_client
        mock_prisma = MagicMock()
        with patch("litellm.proxy.proxy_server.prisma_client", mock_prisma):
            # Mock proxy_config
            mock_proxy_config = MagicMock()
            mock_proxy_config.get_config = AsyncMock(return_value={})
            mock_proxy_config.parse_search_tools = MagicMock(return_value=None)
            with patch("litellm.proxy.proxy_server.proxy_config", mock_proxy_config):
                # Mock auth
                from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

                app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(
                    user_role=LitellmUserRoles.PROXY_ADMIN, user_id="admin_user"
                )

                try:
                    test_client = TestClient(app)
                    response = test_client.get("/search_tools/list")
                    assert response.status_code == 200
                    data = response.json()
                    assert "search_tools" in data
                    assert len(data["search_tools"]) == 3

                    # Test datetime conversion for tool 1
                    tool1 = next(
                        (t for t in data["search_tools"] if t["search_tool_name"] == "datetime-test-tool"),
                        None,
                    )
                    assert tool1 is not None
                    assert isinstance(tool1["created_at"], str)
                    assert tool1["created_at"] == "2024-01-15T10:30:45.123456"
                    assert isinstance(tool1["updated_at"], str)
                    assert tool1["updated_at"] == "2024-01-16T14:20:30.789012"
                    # Verify masking of sensitive values
                    assert tool1["litellm_params"]["api_key"] != "sk-test"
                    assert "****" in tool1["litellm_params"]["api_key"]

                    # Test None handling for tool 2
                    tool2 = next(
                        (t for t in data["search_tools"] if t["search_tool_name"] == "null-datetime-tool"),
                        None,
                    )
                    assert tool2 is not None
                    assert tool2["created_at"] is None
                    assert tool2["updated_at"] is None
                    # Verify masking of sensitive values
                    assert tool2["litellm_params"]["api_key"] != "tvly-test"
                    assert "****" in tool2["litellm_params"]["api_key"]

                    # Test string passthrough for tool 3
                    tool3 = next(
                        (t for t in data["search_tools"] if t["search_tool_name"] == "string-datetime-tool"),
                        None,
                    )
                    assert tool3 is not None
                    assert tool3["created_at"] == "2024-01-17T08:15:00"
                    assert tool3["updated_at"] == "2024-01-18T09:25:00"
                    # Verify masking of sensitive values
                    assert tool3["litellm_params"]["api_key"] != "sk-test"
                    assert "****" in tool3["litellm_params"]["api_key"]
                finally:
                    app.dependency_overrides.pop(user_api_key_auth, None)


@pytest.mark.asyncio
async def test_list_search_tools_config_error_handling(monkeypatch):
    """Test that config errors are handled gracefully"""
    # Mock DB tools
    db_tools = [
        {
            "search_tool_id": "test-id-1",
            "search_tool_name": "db-tool-1",
            "litellm_params": {"search_provider": "perplexity", "api_key": "sk-test"},
            "search_tool_info": {"description": "DB tool"},
            "created_at": datetime(2023, 11, 9, 12, 34, 56),
            "updated_at": datetime(2023, 11, 9, 13, 45, 12),
        }
    ]

    # Mock SearchToolRegistry
    mock_registry = MagicMock()
    mock_registry.get_all_search_tools_from_db = AsyncMock(return_value=db_tools)
    with patch(
        "litellm.proxy.search_endpoints.search_tool_management.SEARCH_TOOL_REGISTRY",
        mock_registry,
    ):
        # Mock prisma_client
        mock_prisma = MagicMock()
        with patch("litellm.proxy.proxy_server.prisma_client", mock_prisma):
            # Mock proxy_config to raise an error
            mock_proxy_config = MagicMock()
            mock_proxy_config.get_config = AsyncMock(side_effect=Exception("Config error"))
            with patch("litellm.proxy.proxy_server.proxy_config", mock_proxy_config):
                # Mock auth
                from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

                app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(
                    user_role=LitellmUserRoles.PROXY_ADMIN, user_id="admin_user"
                )

                try:
                    # Should still succeed and return DB tools only
                    response = client.get("/search_tools/list")
                    assert response.status_code == 200
                    data = response.json()
                    assert "search_tools" in data
                    # Should only have DB tools since config failed
                    assert len(data["search_tools"]) == 1
                    assert data["search_tools"][0]["search_tool_name"] == "db-tool-1"
                    # Verify masking of sensitive values
                    assert data["search_tools"][0]["litellm_params"]["api_key"] != "sk-test"
                    assert "****" in data["search_tools"][0]["litellm_params"]["api_key"]
                finally:
                    app.dependency_overrides.pop(user_api_key_auth, None)


@pytest.mark.asyncio
async def test_list_search_tools_no_prisma_client(monkeypatch):
    """Test error handling when prisma_client is None"""
    with patch("litellm.proxy.proxy_server.prisma_client", None):
        from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

        app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(
            user_role=LitellmUserRoles.PROXY_ADMIN, user_id="admin_user"
        )

        try:
            test_client = TestClient(app)
            response = test_client.get("/search_tools/list")
            assert response.status_code == 500
            data = response.json()
            assert "Prisma client not initialized" in data["detail"]
        finally:
            app.dependency_overrides.pop(user_api_key_auth, None)


@pytest.mark.asyncio
async def test_list_search_tools_db_masking_sensitive_values(monkeypatch):
    """
    Test that sensitive values in DB search tools are properly masked.
    This tests the new masking logic added for database search tools.
    """
    # Mock DB tools with various sensitive fields
    db_tools = [
        {
            "search_tool_id": "test-id-1",
            "search_tool_name": "perplexity-tool",
            "litellm_params": {
                "search_provider": "perplexity",
                "api_key": "pplx-sk-1234567890abcdef",
                "api_base": "https://api.perplexity.ai",
            },
            "search_tool_info": {"description": "Perplexity tool"},
            "created_at": datetime(2023, 11, 9, 12, 34, 56),
            "updated_at": datetime(2023, 11, 9, 13, 45, 12),
        },
        {
            "search_tool_id": "test-id-2",
            "search_tool_name": "tavily-tool",
            "litellm_params": {
                "search_provider": "tavily",
                "api_key": "tvly-secret-key-12345",
                "api_base": "https://api.tavily.com",
            },
            "search_tool_info": {"description": "Tavily tool"},
            "created_at": datetime(2023, 11, 9, 12, 34, 56),
            "updated_at": datetime(2023, 11, 9, 13, 45, 12),
        },
        {
            "search_tool_id": "test-id-3",
            "search_tool_name": "tool-with-token",
            "litellm_params": {
                "search_provider": "custom",
                "access_token": "token-abcdefghijklmnop",
                "secret_key": "secret-xyz123",
            },
            "search_tool_info": {"description": "Tool with token"},
            "created_at": datetime(2023, 11, 9, 12, 34, 56),
            "updated_at": datetime(2023, 11, 9, 13, 45, 12),
        },
        {
            "search_tool_id": "test-id-4",
            "search_tool_name": "tool-with-non-sensitive",
            "litellm_params": {
                "search_provider": "custom",
                "max_results": 10,
                "timeout": 30,
            },
            "search_tool_info": {"description": "Tool without sensitive fields"},
            "created_at": datetime(2023, 11, 9, 12, 34, 56),
            "updated_at": datetime(2023, 11, 9, 13, 45, 12),
        },
    ]

    # Mock SearchToolRegistry
    mock_registry = MagicMock()
    mock_registry.get_all_search_tools_from_db = AsyncMock(return_value=db_tools)
    with patch(
        "litellm.proxy.search_endpoints.search_tool_management.SEARCH_TOOL_REGISTRY",
        mock_registry,
    ):
        # Mock prisma_client
        mock_prisma = MagicMock()
        with patch("litellm.proxy.proxy_server.prisma_client", mock_prisma):
            # Mock proxy_config
            mock_proxy_config = MagicMock()
            mock_proxy_config.get_config = AsyncMock(return_value={})
            mock_proxy_config.parse_search_tools = MagicMock(return_value=None)
            with patch("litellm.proxy.proxy_server.proxy_config", mock_proxy_config):
                # Mock auth
                from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

                app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(
                    user_role=LitellmUserRoles.PROXY_ADMIN, user_id="admin_user"
                )

                try:
                    test_client = TestClient(app)
                    response = test_client.get("/search_tools/list")
                    assert response.status_code == 200
                    data = response.json()
                    assert "search_tools" in data
                    assert len(data["search_tools"]) == 4

                    # Test tool 1: api_key should be masked
                    tool1 = next(
                        (t for t in data["search_tools"] if t["search_tool_name"] == "perplexity-tool"),
                        None,
                    )
                    assert tool1 is not None
                    assert tool1["litellm_params"]["api_key"] != "pplx-sk-1234567890abcdef"
                    assert "****" in tool1["litellm_params"]["api_key"]
                    assert tool1["litellm_params"]["search_provider"] == "perplexity"
                    assert tool1["litellm_params"]["api_base"] == "https://api.perplexity.ai"

                    # Test tool 2: api_key should be masked
                    tool2 = next(
                        (t for t in data["search_tools"] if t["search_tool_name"] == "tavily-tool"),
                        None,
                    )
                    assert tool2 is not None
                    assert tool2["litellm_params"]["api_key"] != "tvly-secret-key-12345"
                    assert "****" in tool2["litellm_params"]["api_key"]
                    assert tool2["litellm_params"]["search_provider"] == "tavily"

                    # Test tool 3: access_token and secret_key should be masked
                    tool3 = next(
                        (t for t in data["search_tools"] if t["search_tool_name"] == "tool-with-token"),
                        None,
                    )
                    assert tool3 is not None
                    assert tool3["litellm_params"]["access_token"] != "token-abcdefghijklmnop"
                    assert "****" in tool3["litellm_params"]["access_token"]
                    assert tool3["litellm_params"]["secret_key"] != "secret-xyz123"
                    assert "****" in tool3["litellm_params"]["secret_key"]

                    # Test tool 4: non-sensitive fields should remain unmasked
                    tool4 = next(
                        (t for t in data["search_tools"] if t["search_tool_name"] == "tool-with-non-sensitive"),
                        None,
                    )
                    assert tool4 is not None
                    assert tool4["litellm_params"]["max_results"] == 10
                    assert tool4["litellm_params"]["timeout"] == 30
                    assert tool4["litellm_params"]["search_provider"] == "custom"
                finally:
                    app.dependency_overrides.pop(user_api_key_auth, None)
