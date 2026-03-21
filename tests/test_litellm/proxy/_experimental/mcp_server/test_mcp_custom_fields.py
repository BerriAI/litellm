"""
Test suite for MCP server custom fields functionality.

Tests that mcp_info can accept arbitrary custom fields in addition to predefined ones.
"""
import pytest
import sys
import os
from unittest.mock import Mock, patch
from typing import Dict, Any

# Add the path to find the modules
sys.path.insert(
    0, os.path.abspath("../../../..")
)  # Adjust the path as needed

from litellm.proxy._experimental.mcp_server.mcp_server_manager import MCPServerManager
from litellm.types.mcp import MCPAuth
from litellm.proxy._types import LiteLLM_MCPServerTable


class TestMCPCustomFields:
    """Test custom fields functionality in MCP server configuration."""

    async def test_custom_fields_preserved_from_config(self):
        """Test that custom fields in mcp_info are preserved when loading from config."""
        manager = MCPServerManager()

        # Mock config with custom fields
        mock_config = {
            "test_server": {
                "url": "http://localhost:3000",
                "transport": "http",
                "auth_type": "bearer_token",
                "authentication_token": "test-token",
                "mcp_info": {
                    "server_name": "Test Server",
                    "description": "A test server",
                    "custom_field_1": "custom_value_1",
                    "custom_field_2": {"nested": "value"},
                    "custom_field_3": ["list", "values"],
                    "priority": 10,
                    "tags": ["production", "api"]
                }
            }
        }

        # Load servers from config
        await manager.load_servers_from_config(mock_config)

        # Get the loaded server
        servers = list(manager.config_mcp_servers.values())
        assert len(servers) == 1

        server = servers[0]
        mcp_info = server.mcp_info

        # Verify standard fields are preserved
        assert mcp_info["server_name"] == "Test Server"
        assert mcp_info["description"] == "A test server"

        # Verify custom fields are preserved
        assert mcp_info["custom_field_1"] == "custom_value_1"
        assert mcp_info["custom_field_2"] == {"nested": "value"}
        assert mcp_info["custom_field_3"] == ["list", "values"]
        assert mcp_info["priority"] == 10
        assert mcp_info["tags"] == ["production", "api"]

    async def test_custom_fields_preserved_from_database(self):
        """Test that custom fields in mcp_info are preserved when adding from database."""
        manager = MCPServerManager()

        # Mock database record with custom fields
        mock_server = LiteLLM_MCPServerTable(
            server_id="test-server-id",
            server_name="Test Server",
            alias=None,
            description="A test server",
            url="http://localhost:3000",
            transport="http",
            auth_type=MCPAuth.bearer_token,
            mcp_info={
                "server_name": "Test Server",
                "description": "A test server",
                "custom_db_field": "database_value",
                "metadata": {"source": "database"},
                "version": "1.0.0",
            },
            command=None,
            args=[],
            env={},
            mcp_access_groups=[],
        )

        # Add server to manager
        await manager.add_server(mock_server)

        # Get the added server
        server = manager.get_mcp_server_by_id("test-server-id")
        assert server is not None

        mcp_info = server.mcp_info

        # Verify standard fields are preserved
        assert mcp_info["server_name"] == "Test Server"
        assert mcp_info["description"] == "A test server"

        # Verify custom fields are preserved
        assert mcp_info["custom_db_field"] == "database_value"
        assert mcp_info["metadata"] == {"source": "database"}
        assert mcp_info["version"] == "1.0.0"

    async def test_empty_mcp_info_handled_gracefully(self):
        """Test that empty or missing mcp_info is handled gracefully."""
        manager = MCPServerManager()

        # Config with empty mcp_info
        mock_config = {
            "test_server": {
                "url": "http://localhost:3000",
                "transport": "http",
                "mcp_info": {}
            }
        }

        await manager.load_servers_from_config(mock_config)

        servers = list(manager.config_mcp_servers.values())
        assert len(servers) == 1

        server = servers[0]
        mcp_info = server.mcp_info

        # Should have default server_name
        assert mcp_info["server_name"] == "test_server"

    async def test_missing_mcp_info_creates_defaults(self):
        """Test that missing mcp_info creates appropriate defaults."""
        manager = MCPServerManager()

        # Config without mcp_info
        mock_config = {
            "test_server": {
                "url": "http://localhost:3000",
                "transport": "http",
                "description": "Server description"
            }
        }

        await manager.load_servers_from_config(mock_config)

        servers = list(manager.config_mcp_servers.values())
        assert len(servers) == 1

        server = servers[0]
        mcp_info = server.mcp_info

        # Should have default server_name and description from config
        assert mcp_info["server_name"] == "test_server"
        assert mcp_info["description"] == "Server description"

    async def test_config_description_fallback(self):
        """Test that description from config level is used as fallback."""
        manager = MCPServerManager()

        # Config with description at server level but not in mcp_info
        mock_config = {
            "test_server": {
                "url": "http://localhost:3000",
                "transport": "http",
                "description": "Config level description",
                "mcp_info": {
                    "custom_field": "custom_value"
                }
            }
        }

        await manager.load_servers_from_config(mock_config)

        servers = list(manager.config_mcp_servers.values())
        server = servers[0]
        mcp_info = server.mcp_info

        # Should use config level description as fallback
        assert mcp_info["description"] == "Config level description"
        assert mcp_info["custom_field"] == "custom_value"

    async def test_mcp_info_description_takes_precedence(self):
        """Test that description in mcp_info takes precedence over config level."""
        manager = MCPServerManager()

        # Config with description at both levels
        mock_config = {
            "test_server": {
                "url": "http://localhost:3000",
                "transport": "http",
                "description": "Config level description",
                "mcp_info": {
                    "description": "MCP info description",
                    "custom_field": "custom_value"
                }
            }
        }

        await manager.load_servers_from_config(mock_config)

        servers = list(manager.config_mcp_servers.values())
        server = servers[0]
        mcp_info = server.mcp_info

        # Should use mcp_info description, not config level
        assert mcp_info["description"] == "MCP info description"
        assert mcp_info["custom_field"] == "custom_value"
