"""Unit tests for fetch_tool_management.py endpoints.

All heavy dependencies (FastAPI, proxy_server, auth) are mocked.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch, PropertyMock

from fastapi import HTTPException


class TestGetEndpoints:
    """Test GET endpoints."""

    @pytest.fixture(autouse=True)
    def setup(self, monkeypatch):
        """Mock proxy_server.prisma_client before importing."""
        self.mock_prisma = MagicMock()
        self.mock_config = MagicMock()
        self.mock_prisma_client_module = MagicMock()
        self.mock_prisma_client_module.prisma_client = self.mock_prisma
        self.mock_prisma_client_module.proxy_config = self.mock_config

        self.mock_registry = MagicMock()
        self.mock_registry.get_all_fetch_tools_from_db = AsyncMock(return_value=[])
        self.mock_registry.get_fetch_tool_from_db = AsyncMock(return_value=None)

        monkeypatch.setattr(
            "litellm.proxy.proxy_server",
            self.mock_prisma_client_module,
        )
        monkeypatch.setattr(
            "litellm.proxy.fetch_endpoints.fetch_tool_management.FETCH_TOOL_REGISTRY",
            self.mock_registry,
        )

        # Now import after patching
        from litellm.proxy.fetch_endpoints import fetch_tool_management
        self.module = fetch_tool_management

    @pytest.mark.asyncio
    async def test_list_fetch_tools_success_empty(self, setup):
        """Test listing with empty DB."""
        result = await self.module.list_fetch_tools()
        assert result == {"fetch_tools": []}

    @pytest.mark.asyncio
    async def test_list_fetch_tools_with_tools(self, setup):
        """Test listing with tools from DB."""
        mock_tool = {
            "fetch_tool_id": "1",
            "fetch_tool_name": "test-tool",
            "litellm_params": {"api_key": "sk-secret"},
            "fetch_tool_info": {},
        }
        self.mock_registry.get_all_fetch_tools_from_db = AsyncMock(
            return_value=[mock_tool]
        )

        result = await self.module.list_fetch_tools()
        tools = result["fetch_tools"]
        assert len(tools) == 1
        assert tools[0]["fetch_tool_name"] == "test-tool"
        # API keys are masked by _get_masked_values
        assert "api_key" in tools[0]["litellm_params"]

    @pytest.mark.asyncio
    async def test_list_fetch_tools_with_config_tools(self, setup):
        """Test listing merges DB + config tools."""
        db_tool = {
            "fetch_tool_id": "1",
            "fetch_tool_name": "db-tool",
            "litellm_params": {},
            "fetch_tool_info": {},
        }
        self.mock_registry.get_all_fetch_tools_from_db = AsyncMock(
            return_value=[db_tool]
        )

        # Config has a tool with same name → not added twice
        self.mock_config.config = {"fetch_tools": [{"fetch_tool_name": "db-tool"}]}
        self.mock_config.parse_fetch_tools = MagicMock(return_value=[])

        result = await self.module.list_fetch_tools()
        assert len(result["fetch_tools"]) == 1

    @pytest.mark.asyncio
    async def test_get_fetch_tool_found(self, setup):
        """Test getting a tool that exists."""
        self.mock_registry.get_fetch_tool_from_db = AsyncMock(
            return_value={
                "fetch_tool_id": "123",
                "fetch_tool_name": "test",
            }
        )
        result = await self.module.get_fetch_tool("123")
        assert result["fetch_tool_id"] == "123"

    @pytest.mark.asyncio
    async def test_get_fetch_tool_not_found(self, setup):
        """Test getting a tool that does not exist."""
        self.mock_registry.get_fetch_tool_from_db = AsyncMock(return_value=None)

        with pytest.raises(HTTPException) as exc_info:
            await self.module.get_fetch_tool("missing")
        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_list_available_providers(self, setup):
        """Test listing available providers."""
        result = await self.module.list_available_fetch_providers()
        assert "providers" in result
        assert len(result["providers"]) >= 1
        assert result["providers"][0]["provider_name"] == "firecrawl"


class TestCreateEndpoint:
    """Test POST /fetch_tools."""

    @pytest.fixture(autouse=True)
    def setup(self, monkeypatch):
        self.mock_prisma = MagicMock()
        self.mock_prisma_client_module = MagicMock()
        self.mock_prisma_client_module.prisma_client = self.mock_prisma

        self.mock_registry = MagicMock()
        self.mock_registry.add_fetch_tool_to_db = AsyncMock(
            return_value={"fetch_tool_id": "new-id", "fetch_tool_name": "new-tool"}
        )

        monkeypatch.setattr(
            "litellm.proxy.proxy_server",
            self.mock_prisma_client_module,
        )
        monkeypatch.setattr(
            "litellm.proxy.fetch_endpoints.fetch_tool_management.FETCH_TOOL_REGISTRY",
            self.mock_registry,
        )

        from litellm.proxy.fetch_endpoints import fetch_tool_management
        self.module = fetch_tool_management

    @pytest.mark.asyncio
    async def test_create_success(self, setup):
        """Test creating a tool."""
        request = {
            "fetch_tool": {
                "fetch_tool_name": "new-tool",
                "litellm_params": {"provider": "firecrawl"},
            }
        }
        result = await self.module.create_fetch_tool(request)
        assert result["fetch_tool_id"] == "new-id"

    @pytest.mark.asyncio
    async def test_create_error(self, setup):
        """Test creation error raises HTTPException."""
        self.mock_registry.add_fetch_tool_to_db = AsyncMock(
            side_effect=Exception("db fail")
        )

        request = {"fetch_tool": {"fetch_tool_name": "fail"}}

        with pytest.raises(HTTPException) as exc_info:
            await self.module.create_fetch_tool(request)
        assert exc_info.value.status_code == 500


class TestUpdateEndpoint:
    """Test PUT /fetch_tools/{id}."""

    @pytest.fixture(autouse=True)
    def setup(self, monkeypatch):
        self.mock_prisma = MagicMock()
        self.mock_prisma_client_module = MagicMock()
        self.mock_prisma_client_module.prisma_client = self.mock_prisma

        self.mock_registry = MagicMock()
        self.mock_registry.update_fetch_tool_in_db = AsyncMock(
            return_value={"fetch_tool_id": "123", "fetch_tool_name": "updated"}
        )

        monkeypatch.setattr(
            "litellm.proxy.proxy_server",
            self.mock_prisma_client_module,
        )
        monkeypatch.setattr(
            "litellm.proxy.fetch_endpoints.fetch_tool_management.FETCH_TOOL_REGISTRY",
            self.mock_registry,
        )

        from litellm.proxy.fetch_endpoints import fetch_tool_management
        self.module = fetch_tool_management

    @pytest.mark.asyncio
    async def test_update_success(self, setup):
        """Test updating a tool."""
        request = {"fetch_tool": {"fetch_tool_name": "updated", "litellm_params": {}}}
        result = await self.module.update_fetch_tool("123", request)
        assert result["fetch_tool_name"] == "updated"

    @pytest.mark.asyncio
    async def test_update_error(self, setup):
        """Test update error."""
        self.mock_registry.update_fetch_tool_in_db = AsyncMock(
            side_effect=Exception("not found")
        )

        request = {"fetch_tool": {"fetch_tool_name": "updated"}}
        with pytest.raises(HTTPException) as exc_info:
            await self.module.update_fetch_tool("bad-id", request)
        assert exc_info.value.status_code == 500


class TestDeleteEndpoint:
    """Test DELETE /fetch_tools/{id}."""

    @pytest.fixture(autouse=True)
    def setup(self, monkeypatch):
        self.mock_prisma = MagicMock()
        self.mock_prisma_client_module = MagicMock()
        self.mock_prisma_client_module.prisma_client = self.mock_prisma

        self.mock_registry = MagicMock()
        self.mock_registry.delete_fetch_tool_from_db = AsyncMock(
            return_value={"message": "deleted"}
        )

        monkeypatch.setattr(
            "litellm.proxy.proxy_server",
            self.mock_prisma_client_module,
        )
        monkeypatch.setattr(
            "litellm.proxy.fetch_endpoints.fetch_tool_management.FETCH_TOOL_REGISTRY",
            self.mock_registry,
        )

        from litellm.proxy.fetch_endpoints import fetch_tool_management
        self.module = fetch_tool_management

    @pytest.mark.asyncio
    async def test_delete_success(self, setup):
        """Test deleting a tool."""
        result = await self.module.delete_fetch_tool("123")
        assert result["message"] == "deleted"

    @pytest.mark.asyncio
    async def test_delete_error(self, setup):
        """Test delete error."""
        self.mock_registry.delete_fetch_tool_from_db = AsyncMock(
            side_effect=Exception("not found")
        )

        with pytest.raises(HTTPException) as exc_info:
            await self.module.delete_fetch_tool("bad-id")
        assert exc_info.value.status_code == 500
