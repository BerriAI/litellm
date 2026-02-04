"""
Tests for Claude Code Marketplace endpoints.

Tests:
1. Register a plugin
2. Get marketplace.json (list enabled plugins)
"""

import json
import os
import sys
import time
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.path.insert(0, os.path.abspath("../.."))

import litellm
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.proxy_server import LitellmUserRoles
from litellm.caching.caching import DualCache
from litellm.types.proxy.claude_code_endpoints import RegisterPluginRequest

# Import the functions we're testing
from litellm.proxy.anthropic_endpoints.claude_code_endpoints.claude_code_marketplace import (
    register_plugin,
    get_marketplace,
)


class MockPluginRecord:
    """Mock plugin record that mimics Prisma model behavior."""

    def __init__(self, name, version, description, manifest_json, enabled=True, created_by=None):
        self.id = f"plugin-{name}-{int(time.time())}"
        self.name = name
        self.version = version
        self.description = description
        self.manifest_json = manifest_json
        self.files_json = "{}"
        self.enabled = enabled
        self.created_at = datetime.now(timezone.utc)
        self.updated_at = datetime.now(timezone.utc)
        self.created_by = created_by


@pytest.fixture
def mock_prisma_client():
    """Create a mock PrismaClient that doesn't require Prisma binaries."""
    # In-memory storage for plugins
    plugins_store = {}

    # Create mock client
    mock_client = MagicMock()
    mock_client.proxy_logging_obj = MagicMock()

    # Mock the db attribute
    mock_client.db = MagicMock()

    # Mock the plugin table with async methods
    mock_table = MagicMock()

    async def find_unique(where):
        """Mock find_unique - returns plugin if exists, None otherwise."""
        plugin_name = where.get("name")
        return plugins_store.get(plugin_name)

    async def find_many(where=None):
        """Mock find_many - returns list of plugins matching where clause."""
        if where is None or where == {}:
            return list(plugins_store.values())
        enabled = where.get("enabled")
        if enabled is not None:
            return [p for p in plugins_store.values() if p.enabled == enabled]
        return list(plugins_store.values())

    async def create(data):
        """Mock create - creates a new plugin."""
        plugin_name = data["name"]
        manifest = data.get("manifest_json", "{}")
        plugin = MockPluginRecord(
            name=plugin_name,
            version=data.get("version"),
            description=data.get("description"),
            manifest_json=manifest,
            enabled=data.get("enabled", True),
            created_by=data.get("created_by"),
        )
        plugins_store[plugin_name] = plugin
        return plugin

    async def update(where, data):
        """Mock update - updates an existing plugin."""
        plugin_name = where.get("name")
        if plugin_name not in plugins_store:
            raise ValueError(f"Plugin {plugin_name} not found")
        plugin = plugins_store[plugin_name]
        # Update fields
        if "version" in data:
            plugin.version = data["version"]
        if "description" in data:
            plugin.description = data["description"]
        if "manifest_json" in data:
            plugin.manifest_json = data["manifest_json"]
        if "enabled" in data:
            plugin.enabled = data["enabled"]
        if "updated_at" in data:
            plugin.updated_at = data["updated_at"]
        return plugin

    async def delete(where):
        """Mock delete - deletes a plugin."""
        plugin_name = where.get("name")
        if plugin_name in plugins_store:
            del plugins_store[plugin_name]
        return None

    async def connect():
        """Mock connect - no-op."""
        pass

    # Set up async mocks
    mock_table.find_unique = AsyncMock(side_effect=find_unique)
    mock_table.find_many = AsyncMock(side_effect=find_many)
    mock_table.create = AsyncMock(side_effect=create)
    mock_table.update = AsyncMock(side_effect=update)
    mock_table.delete = AsyncMock(side_effect=delete)

    mock_client.db.litellm_claudecodeplugintable = mock_table
    mock_client.connect = AsyncMock(side_effect=connect)

    # Store plugins_store on the mock for cleanup if needed
    mock_client._plugins_store = plugins_store

    return mock_client


@pytest.mark.asyncio
async def test_register_plugin(mock_prisma_client):
    """Test registering a plugin in the marketplace."""
    setattr(litellm.proxy.proxy_server, "prisma_client", mock_prisma_client)
    setattr(litellm.proxy.proxy_server, "master_key", "sk-1234")

    await litellm.proxy.proxy_server.prisma_client.connect()

    # Create a unique plugin name for this test
    plugin_name = f"test-plugin-{int(time.time())}"

    request = RegisterPluginRequest(
        name=plugin_name,
        source={"source": "github", "repo": "test-org/test-repo"},
        version="1.0.0",
        description="Test plugin for unit tests",
    )

    user_api_key_dict = UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN,
        api_key="sk-1234",
        user_id="test-user",
    )

    response = await register_plugin(
        request=request,
        user_api_key_dict=user_api_key_dict,
    )

    assert response["status"] == "success"
    assert response["action"] == "created"
    assert response["plugin"]["name"] == plugin_name
    assert response["plugin"]["version"] == "1.0.0"
    assert response["plugin"]["enabled"] is True

    # Verify the plugin was stored in the mock
    stored_plugin = await mock_prisma_client.db.litellm_claudecodeplugintable.find_unique(
        where={"name": plugin_name}
    )
    assert stored_plugin is not None
    assert stored_plugin.name == plugin_name

    # Cleanup - delete the plugin
    await mock_prisma_client.db.litellm_claudecodeplugintable.delete(
        where={"name": plugin_name}
    )


@pytest.mark.asyncio
async def test_get_marketplace(mock_prisma_client):
    """Test getting marketplace.json with registered plugins."""
    setattr(litellm.proxy.proxy_server, "prisma_client", mock_prisma_client)
    setattr(litellm.proxy.proxy_server, "master_key", "sk-1234")

    await litellm.proxy.proxy_server.prisma_client.connect()

    # First register a plugin
    plugin_name = f"test-marketplace-plugin-{int(time.time())}"

    request = RegisterPluginRequest(
        name=plugin_name,
        source={"source": "github", "repo": "test-org/marketplace-test"},
        version="2.0.0",
        description="Test plugin for marketplace test",
    )

    user_api_key_dict = UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN,
        api_key="sk-1234",
        user_id="test-user",
    )

    await register_plugin(
        request=request,
        user_api_key_dict=user_api_key_dict,
    )

    # Now get the marketplace
    response = await get_marketplace()

    # Response is a JSONResponse, get the body
    body = json.loads(response.body.decode())

    assert body["name"] == "litellm"
    assert "plugins" in body

    # Find our plugin in the list
    our_plugin = next(
        (p for p in body["plugins"] if p["name"] == plugin_name),
        None
    )
    assert our_plugin is not None
    assert our_plugin["source"] == {"source": "github", "repo": "test-org/marketplace-test"}
    assert our_plugin["version"] == "2.0.0"

    # Cleanup
    await mock_prisma_client.db.litellm_claudecodeplugintable.delete(
        where={"name": plugin_name}
    )
