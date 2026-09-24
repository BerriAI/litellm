import contextlib
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient


from litellm.proxy._types import (
    LiteLLM_ObjectPermissionTable,
    LiteLLM_TeamTable,
    LitellmUserRoles,
    UserAPIKeyAuth,
)

# Import proxy_server module first to ensure it's initialized
import litellm.proxy.proxy_server as ps

# Now we can safely import app
from litellm.proxy.proxy_server import app
from litellm.types.search import SearchToolInfoResponse

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
            "litellm_params": {
                "search_provider": "tavily",
                "api_key": "tvly-secret-key",
            },
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
            mock_proxy_config.get_config = AsyncMock(
                return_value={"search_tools": config_tools}
            )
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
            mock_proxy_config.get_config = AsyncMock(
                return_value={"search_tools": config_tools}
            )
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
                        (
                            t
                            for t in data["search_tools"]
                            if t["search_tool_name"] == "existing-tool"
                        ),
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
                        (
                            t
                            for t in data["search_tools"]
                            if t["search_tool_name"] == "unique-config-tool"
                        ),
                        None,
                    )
                    assert config_tool is not None
                    assert config_tool["is_from_config"] is True

                    # Verify duplicate config tool is NOT present
                    duplicate_tool = next(
                        (
                            t
                            for t in data["search_tools"]
                            if t["search_tool_name"] == "existing-tool"
                            and t["is_from_config"] is True
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
                        (
                            t
                            for t in data["search_tools"]
                            if t["search_tool_name"] == "datetime-test-tool"
                        ),
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
                        (
                            t
                            for t in data["search_tools"]
                            if t["search_tool_name"] == "null-datetime-tool"
                        ),
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
                        (
                            t
                            for t in data["search_tools"]
                            if t["search_tool_name"] == "string-datetime-tool"
                        ),
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
            mock_proxy_config.get_config = AsyncMock(
                side_effect=Exception("Config error")
            )
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
                    assert (
                        data["search_tools"][0]["litellm_params"]["api_key"]
                        != "sk-test"
                    )
                    assert (
                        "****" in data["search_tools"][0]["litellm_params"]["api_key"]
                    )
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
                        (
                            t
                            for t in data["search_tools"]
                            if t["search_tool_name"] == "perplexity-tool"
                        ),
                        None,
                    )
                    assert tool1 is not None
                    assert (
                        tool1["litellm_params"]["api_key"] != "pplx-sk-1234567890abcdef"
                    )
                    assert "****" in tool1["litellm_params"]["api_key"]
                    assert tool1["litellm_params"]["search_provider"] == "perplexity"
                    assert (
                        tool1["litellm_params"]["api_base"]
                        == "https://api.perplexity.ai"
                    )

                    # Test tool 2: api_key should be masked
                    tool2 = next(
                        (
                            t
                            for t in data["search_tools"]
                            if t["search_tool_name"] == "tavily-tool"
                        ),
                        None,
                    )
                    assert tool2 is not None
                    assert tool2["litellm_params"]["api_key"] != "tvly-secret-key-12345"
                    assert "****" in tool2["litellm_params"]["api_key"]
                    assert tool2["litellm_params"]["search_provider"] == "tavily"

                    # Test tool 3: access_token and secret_key should be masked
                    tool3 = next(
                        (
                            t
                            for t in data["search_tools"]
                            if t["search_tool_name"] == "tool-with-token"
                        ),
                        None,
                    )
                    assert tool3 is not None
                    assert (
                        tool3["litellm_params"]["access_token"]
                        != "token-abcdefghijklmnop"
                    )
                    assert "****" in tool3["litellm_params"]["access_token"]
                    assert tool3["litellm_params"]["secret_key"] != "secret-xyz123"
                    assert "****" in tool3["litellm_params"]["secret_key"]

                    # Test tool 4: non-sensitive fields should remain unmasked
                    tool4 = next(
                        (
                            t
                            for t in data["search_tools"]
                            if t["search_tool_name"] == "tool-with-non-sensitive"
                        ),
                        None,
                    )
                    assert tool4 is not None
                    assert tool4["litellm_params"]["max_results"] == 10
                    assert tool4["litellm_params"]["timeout"] == 30
                    assert tool4["litellm_params"]["search_provider"] == "custom"
                finally:
                    app.dependency_overrides.pop(user_api_key_auth, None)


@pytest.mark.asyncio
async def test_get_all_search_tools_from_db_retries_on_transport_error():
    """`SearchToolRegistry.get_all_search_tools_from_db` self-heals across one
    ClientNotConnectedError via call_with_db_reconnect_retry."""
    import prisma
    from litellm.proxy.search_endpoints.search_tool_registry import (
        SearchToolRegistry,
    )

    invocations: list = []

    async def _flaky_find_many(**kwargs):
        invocations.append(None)
        if len(invocations) == 1:
            raise prisma.errors.ClientNotConnectedError()
        return []

    mock_prisma_client = MagicMock()
    mock_prisma_client.db.litellm_searchtoolstable.find_many = AsyncMock(
        side_effect=_flaky_find_many
    )
    mock_prisma_client.attempt_db_reconnect = AsyncMock(return_value=True)
    mock_prisma_client._db_auth_reconnect_timeout_seconds = 2.0
    mock_prisma_client._db_auth_reconnect_lock_timeout_seconds = 0.1

    result = await SearchToolRegistry.get_all_search_tools_from_db(
        prisma_client=mock_prisma_client
    )

    assert result == []
    assert len(invocations) == 2
    mock_prisma_client.attempt_db_reconnect.assert_awaited_once()
    reconnect_kwargs = mock_prisma_client.attempt_db_reconnect.await_args.kwargs
    assert (
        reconnect_kwargs["reason"]
        == "get_all_search_tools_from_db_lookup_failure"
    )


@contextlib.contextmanager
def _mock_search_tool_backend(db_tools):
    """Patch the DB registry, prisma client, and config so /search_tools/list
    returns exactly ``db_tools`` (no config-defined tools)."""
    mock_registry = MagicMock()
    mock_registry.get_all_search_tools_from_db = AsyncMock(return_value=db_tools)
    mock_proxy_config = MagicMock()
    mock_proxy_config.get_config = AsyncMock(return_value={})
    mock_proxy_config.parse_search_tools = MagicMock(return_value=None)
    with (
        patch(
            "litellm.proxy.search_endpoints.search_tool_management.SEARCH_TOOL_REGISTRY",
            mock_registry,
        ),
        patch("litellm.proxy.proxy_server.prisma_client", MagicMock()),
        patch("litellm.proxy.proxy_server.proxy_config", mock_proxy_config),
    ):
        yield


def _scoping_db_tools():
    return [
        {
            "search_tool_id": "db-id-1",
            "search_tool_name": "db-tool-1",
            "litellm_params": {
                "search_provider": "perplexity",
                "api_key": "pplx-secret-1",
                "api_base": "https://api.perplexity.ai",
            },
            "search_tool_info": {"description": "Perplexity"},
            "created_at": datetime(2024, 1, 1),
            "updated_at": datetime(2024, 1, 1),
        },
        {
            "search_tool_id": "db-id-2",
            "search_tool_name": "db-tool-2",
            "litellm_params": {
                "search_provider": "tavily",
                "api_key": "tvly-secret-2",
                "api_base": "https://api.tavily.com",
            },
            "search_tool_info": {"description": "Tavily"},
            "created_at": datetime(2024, 1, 1),
            "updated_at": datetime(2024, 1, 1),
        },
        {
            "search_tool_id": "db-id-3",
            "search_tool_name": "db-tool-3",
            "litellm_params": {"search_provider": "exa", "api_key": "exa-secret-3"},
            "search_tool_info": {"description": "Exa"},
            "created_at": datetime(2024, 1, 1),
            "updated_at": datetime(2024, 1, 1),
        },
    ]


@contextlib.contextmanager
def _override_auth(user):
    from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

    app.dependency_overrides[user_api_key_auth] = lambda: user
    try:
        yield
    finally:
        app.dependency_overrides.pop(user_api_key_auth, None)


@pytest.mark.asyncio
async def test_list_search_tools_scoped_to_key_object_permission():
    """
    Regression: an internal user whose key is restricted to specific search tools
    must only see those tools. Before the fix /search_tools/list returned every
    configured tool, leaking ids, api_base, and metadata for tools it cannot call.
    """
    restricted_user = UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER,
        user_id="internal_user",
        object_permission=LiteLLM_ObjectPermissionTable(
            object_permission_id="op-key",
            search_tools=["db-tool-1"],
        ),
    )

    with (
        _mock_search_tool_backend(_scoping_db_tools()),
        _override_auth(restricted_user),
    ):
        response = TestClient(app).get("/search_tools/list")

    assert response.status_code == 200
    tools = response.json()["search_tools"]
    assert [t["search_tool_name"] for t in tools] == ["db-tool-1"]
    leaked = {t["litellm_params"].get("api_base") for t in tools}
    assert "https://api.tavily.com" not in leaked


@pytest.mark.asyncio
async def test_list_search_tools_unrestricted_internal_user_sees_all():
    """An internal user with no search_tools allowlist is unrestricted and sees every tool."""
    unrestricted_user = UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER, user_id="internal_user"
    )

    with (
        _mock_search_tool_backend(_scoping_db_tools()),
        _override_auth(unrestricted_user),
    ):
        response = TestClient(app).get("/search_tools/list")

    assert response.status_code == 200
    names = {t["search_tool_name"] for t in response.json()["search_tools"]}
    assert names == {"db-tool-1", "db-tool-2", "db-tool-3"}


@pytest.mark.asyncio
async def test_list_search_tools_scoped_to_team_object_permission():
    """A team-level search_tools allowlist also scopes the listing for a non-admin caller."""
    team_member = UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER,
        user_id="internal_user",
        team_id="team-1",
    )
    team_object = LiteLLM_TeamTable(
        team_id="team-1",
        object_permission=LiteLLM_ObjectPermissionTable(
            object_permission_id="op-team",
            search_tools=["db-tool-2"],
        ),
    )

    with (
        _mock_search_tool_backend(_scoping_db_tools()),
        patch(
            "litellm.proxy.auth.auth_checks.get_team_object",
            AsyncMock(return_value=team_object),
        ),
        _override_auth(team_member),
    ):
        response = TestClient(app).get("/search_tools/list")

    assert response.status_code == 200
    assert [t["search_tool_name"] for t in response.json()["search_tools"]] == [
        "db-tool-2"
    ]


@pytest.mark.asyncio
async def test_list_search_tools_admin_with_restricted_key_still_sees_all():
    """Proxy admins bypass search-tool scoping even if their key carries an allowlist."""
    admin_user = UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN,
        user_id="admin_user",
        object_permission=LiteLLM_ObjectPermissionTable(
            object_permission_id="op-admin",
            search_tools=["db-tool-1"],
        ),
    )

    with _mock_search_tool_backend(_scoping_db_tools()), _override_auth(admin_user):
        response = TestClient(app).get("/search_tools/list")

    assert response.status_code == 200
    names = {t["search_tool_name"] for t in response.json()["search_tools"]}
    assert names == {"db-tool-1", "db-tool-2", "db-tool-3"}


def _search_tool_responses(*names: str) -> list[SearchToolInfoResponse]:
    return [
        SearchToolInfoResponse(
            search_tool_id=f"id-{name}",
            search_tool_name=name,
            litellm_params={"search_provider": "perplexity"},
            search_tool_info=None,
            created_at=None,
            updated_at=None,
            is_from_config=False,
        )
        for name in names
    ]


def _team_ids_looked_up(lookup: AsyncMock) -> list[str]:
    return [awaited.args[0] for awaited in lookup.await_args_list]


@pytest.mark.asyncio
async def test_list_search_tools_dashboard_session_key_does_not_look_up_the_ui_team():
    """
    Regression: the Admin UI session key is stamped with the reserved team id
    ``litellm-dashboard``, which has no row in LiteLLM_TeamTable. Resolving it as a real team
    raised 404, which the endpoint reported as a 500, so the Search Tools page was broken for
    every non-admin browsing the dashboard.
    """
    from litellm.constants import UI_SESSION_TOKEN_TEAM_ID

    dashboard_session_user = UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER,
        user_id="internal_user",
        team_id=UI_SESSION_TOKEN_TEAM_ID,
    )
    ui_team_is_not_a_real_team = AsyncMock(
        side_effect=HTTPException(
            status_code=404,
            detail={"error": f"Team doesn't exist in db. Team={UI_SESSION_TOKEN_TEAM_ID}."},
        )
    )

    with (
        _mock_search_tool_backend(_scoping_db_tools()),
        patch(
            "litellm.proxy.auth.auth_checks.get_team_object",
            ui_team_is_not_a_real_team,
        ),
        _override_auth(dashboard_session_user),
    ):
        response = TestClient(app).get("/search_tools/list")

    assert response.status_code == 200
    names = {t["search_tool_name"] for t in response.json()["search_tools"]}
    assert names == {"db-tool-1", "db-tool-2", "db-tool-3"}
    ui_team_is_not_a_real_team.assert_not_awaited()


@pytest.mark.asyncio
async def test_filter_visible_search_tools_dashboard_session_still_honors_key_allowlist():
    """
    Skipping the synthetic team must not widen visibility: a dashboard session whose key
    carries a search_tools allowlist stays scoped to it.
    """
    from litellm.constants import UI_SESSION_TOKEN_TEAM_ID
    from litellm.proxy.search_endpoints.search_tool_management import (
        _filter_visible_search_tools,
    )

    restricted_dashboard_session = UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER,
        user_id="internal_user",
        team_id=UI_SESSION_TOKEN_TEAM_ID,
        object_permission=LiteLLM_ObjectPermissionTable(
            object_permission_id="op-key",
            search_tools=["db-tool-3"],
        ),
    )
    lookup = AsyncMock()

    visible = await _filter_visible_search_tools(
        _search_tool_responses("db-tool-1", "db-tool-2", "db-tool-3"),
        restricted_dashboard_session,
        lookup,
    )

    assert [t["search_tool_name"] for t in visible] == ["db-tool-3"]
    lookup.assert_not_awaited()


@pytest.mark.asyncio
async def test_filter_visible_search_tools_still_applies_a_real_team_allowlist():
    """A caller with a real team is still resolved and scoped by that team's allowlist."""
    from litellm.proxy.search_endpoints.search_tool_management import (
        _filter_visible_search_tools,
    )

    team_member = UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER,
        user_id="internal_user",
        team_id="team-1",
    )
    lookup = AsyncMock(
        return_value=LiteLLM_TeamTable(
            team_id="team-1",
            object_permission=LiteLLM_ObjectPermissionTable(
                object_permission_id="op-team",
                search_tools=["db-tool-2"],
            ),
        )
    )

    visible = await _filter_visible_search_tools(
        _search_tool_responses("db-tool-1", "db-tool-2", "db-tool-3"),
        team_member,
        lookup,
    )

    assert [t["search_tool_name"] for t in visible] == ["db-tool-2"]
    assert _team_ids_looked_up(lookup) == ["team-1"]


@pytest.mark.asyncio
async def test_filter_visible_search_tools_propagates_a_real_team_lookup_failure():
    """
    A caller whose real team cannot be resolved must not fall through to "no team", which
    would drop that team's allowlist and show tools the caller may not call.
    """
    from litellm.proxy.search_endpoints.search_tool_management import (
        _filter_visible_search_tools,
    )

    team_member = UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER,
        user_id="internal_user",
        team_id="deleted-team",
    )
    lookup = AsyncMock(side_effect=HTTPException(status_code=404, detail={"error": "Team doesn't exist in db."}))

    with pytest.raises(HTTPException) as exc_info:
        await _filter_visible_search_tools(
            _search_tool_responses("db-tool-1", "db-tool-2"),
            team_member,
            lookup,
        )

    assert exc_info.value.status_code == 404
    assert _team_ids_looked_up(lookup) == ["deleted-team"]


@pytest.mark.asyncio
async def test_list_search_tools_reports_a_missing_real_team_as_404():
    """
    The endpoint surfaces a genuine team lookup failure with its own status instead of
    masking it as a 500 or quietly returning an unscoped list.
    """
    team_member = UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER,
        user_id="internal_user",
        team_id="deleted-team",
    )

    with (
        _mock_search_tool_backend(_scoping_db_tools()),
        patch(
            "litellm.proxy.auth.auth_checks.get_team_object",
            AsyncMock(
                side_effect=HTTPException(
                    status_code=404,
                    detail={"error": "Team doesn't exist in db. Team=deleted-team."},
                )
            ),
        ),
        _override_auth(team_member),
    ):
        response = TestClient(app).get("/search_tools/list")

    assert response.status_code == 404
    assert "search_tools" not in response.json()


# ---------------------------------------------------------------------------
# Router sync on management writes (LIT-3379)
#
# The proxy resolves prisma_client / proxy_config / llm_router from
# litellm.proxy.proxy_server module globals at call time and reaches its DB layer through a
# module-level registry singleton, so there is no constructor or parameter to inject through.
# Patching those globals is the only seam that exercises the endpoint end to end.
# ---------------------------------------------------------------------------


def _search_tool_row(name: str, provider: str = "tavily") -> dict:
    return {
        "search_tool_id": f"{name}-id",
        "search_tool_name": name,
        "litellm_params": {"search_provider": provider, "api_key": "sk-test"},
        "search_tool_info": {"description": name},
    }


def _fake_registry(db_rows: list) -> MagicMock:
    """A registry singleton whose writes land in db_rows, so the refresh reads back real state."""

    async def _add(search_tool, **_):
        row = _search_tool_row(
            search_tool["search_tool_name"],
            provider=search_tool.get("litellm_params", {}).get("search_provider", "tavily"),
        )
        db_rows.append(row)
        return row

    async def _update(search_tool_id, search_tool, **_):
        row = _search_tool_row(
            search_tool["search_tool_name"],
            provider=search_tool.get("litellm_params", {}).get("search_provider", "tavily"),
        )
        db_rows[:] = [row if existing["search_tool_id"] == search_tool_id else existing for existing in db_rows]
        return row

    async def _delete(search_tool_id, **_):
        db_rows[:] = [existing for existing in db_rows if existing["search_tool_id"] != search_tool_id]
        return {"message": "deleted", "search_tool_name": search_tool_id}

    async def _get_by_id(search_tool_id, **_):
        return next((row for row in db_rows if row["search_tool_id"] == search_tool_id), None)

    registry = MagicMock()
    registry.add_search_tool_to_db = AsyncMock(side_effect=_add)
    registry.update_search_tool_in_db = AsyncMock(side_effect=_update)
    registry.delete_search_tool_from_db = AsyncMock(side_effect=_delete)
    registry.get_search_tool_by_id_from_db = AsyncMock(side_effect=_get_by_id)
    return registry


@contextlib.contextmanager
def _live_router_and_db(db_rows: list):
    """Drive the endpoints against a real ProxyConfig so the router refresh actually runs."""
    from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
    from litellm.proxy.proxy_server import ProxyConfig

    proxy_config = ProxyConfig()
    proxy_config.update_config_state({})
    fake_router = MagicMock()
    fake_router.search_tools = list(db_rows)

    with contextlib.ExitStack() as stack:
        stack.enter_context(patch("litellm.proxy.proxy_server.prisma_client", MagicMock()))  # test-quality-ok: proxy globals are the only seam; see the module note above
        stack.enter_context(patch("litellm.proxy.proxy_server.proxy_config", proxy_config))  # test-quality-ok: proxy globals are the only seam; see the module note above
        stack.enter_context(patch("litellm.proxy.proxy_server.llm_router", fake_router))  # test-quality-ok: proxy globals are the only seam; see the module note above
        stack.enter_context(
            patch(  # test-quality-ok: proxy globals are the only seam; see the module note above
                "litellm.proxy.search_endpoints.search_tool_management.SEARCH_TOOL_REGISTRY",
                _fake_registry(db_rows),
            )
        )
        stack.enter_context(
            patch(  # test-quality-ok: proxy globals are the only seam; see the module note above
                "litellm.proxy.search_endpoints.search_tool_registry.SearchToolRegistry.get_all_search_tools_from_db",
                AsyncMock(side_effect=lambda **_: list(db_rows)),
            )
        )
        app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(
            user_role=LitellmUserRoles.PROXY_ADMIN, user_id="admin_user"
        )
        try:
            yield fake_router
        finally:
            app.dependency_overrides.pop(user_api_key_auth, None)


@pytest.mark.asyncio
async def test_create_search_tool_reaches_the_router_before_the_response():
    """A UI-created tool must be usable immediately, not only after the next config reload tick."""
    with _live_router_and_db([]) as fake_router:
        response = TestClient(app).post(
            "/search_tools",
            json={
                "search_tool": {
                    "search_tool_name": "tavily-search",
                    "litellm_params": {"search_provider": "tavily"},
                }
            },
        )

    assert response.status_code == 200
    assert [tool["search_tool_name"] for tool in fake_router.search_tools] == ["tavily-search"]


@pytest.mark.asyncio
async def test_update_search_tool_reaches_the_router_before_the_response():
    with _live_router_and_db([_search_tool_row("tavily-search", provider="tavily")]) as fake_router:
        response = TestClient(app).put(
            "/search_tools/tavily-search-id",
            json={
                "search_tool": {
                    "search_tool_name": "tavily-search",
                    "litellm_params": {"search_provider": "exa_ai"},
                }
            },
        )

    assert response.status_code == 200
    assert fake_router.search_tools[0]["litellm_params"]["search_provider"] == "exa_ai"


@pytest.mark.asyncio
async def test_delete_search_tool_removes_it_from_the_router():
    """Deleting the last tool must clear the router; the old empty-list guard left it live."""
    with _live_router_and_db([_search_tool_row("tavily-search")]) as fake_router:
        response = TestClient(app).delete("/search_tools/tavily-search-id")

    assert response.status_code == 200
    assert fake_router.search_tools == []


@pytest.mark.asyncio
async def test_create_search_tool_survives_a_failing_router_refresh():
    """The row is already committed, so a refresh failure must not turn into a 500."""
    with _live_router_and_db([]):
        with patch(  # test-quality-ok: forcing the refresh to fail needs the refresh itself replaced
            "litellm.proxy.proxy_server.ProxyConfig.reload_search_tools_from_db",
            AsyncMock(side_effect=RuntimeError("registry boom")),
        ):
            response = TestClient(app).post(
                "/search_tools",
                json={
                    "search_tool": {
                        "search_tool_name": "tavily-search",
                        "litellm_params": {"search_provider": "tavily"},
                    }
                },
            )

    assert response.status_code == 200
    assert response.json()["search_tool_name"] == "tavily-search"
