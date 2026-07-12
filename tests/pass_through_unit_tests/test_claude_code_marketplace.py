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
from litellm.types.proxy.claude_code_endpoints import RegisterPluginRequest

# Import the functions we're testing
from litellm.proxy.anthropic_endpoints.claude_code_endpoints.claude_code_marketplace import (
    register_plugin,
    get_marketplace,
)


class MockPluginRecord:
    """Mock plugin record that mimics Prisma model behavior."""

    def __init__(
        self,
        name,
        version,
        description,
        manifest_json,
        enabled=True,
        created_by=None,
        marketplace_id=None,
    ):
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
        self.marketplace_id = marketplace_id


class MockMarketplaceRecord:
    """Mock LiteLLM_SkillMarketplaceTable record."""

    def __init__(self, id, name, enabled=True):
        self.id = id
        self.name = name
        self.enabled = enabled


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

    def _matches_clause(plugin, clause):
        """Support the small subset of Prisma where-clause shapes get_marketplace
        actually issues: {"enabled": bool} and {"name": {"in": [...]}}."""
        if "enabled" in clause:
            return plugin.enabled == clause["enabled"]
        if "name" in clause:
            return plugin.name in clause["name"].get("in", [])
        return False

    async def find_many(where=None):
        """Mock find_many - returns list of plugins matching where clause."""
        if where is None or where == {}:
            return list(plugins_store.values())
        if "OR" in where:
            return [p for p in plugins_store.values() if any(_matches_clause(p, c) for c in where["OR"])]
        return [p for p in plugins_store.values() if _matches_clause(p, where)]

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
            marketplace_id=data.get("marketplace_id"),
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

    # In-memory storage for marketplace rows, keyed by id.
    marketplaces_store = {}

    mock_marketplace_table = MagicMock()

    async def marketplace_find_many(where=None):
        rows = list(marketplaces_store.values())
        if not where:
            return rows
        if "id" in where and "in" in where["id"]:
            rows = [r for r in rows if r.id in where["id"]["in"]]
        if "enabled" in where:
            rows = [r for r in rows if r.enabled == where["enabled"]]
        return rows

    mock_marketplace_table.find_many = AsyncMock(side_effect=marketplace_find_many)
    mock_client.db.litellm_skillmarketplacetable = mock_marketplace_table
    mock_client._marketplaces_store = marketplaces_store
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
    stored_plugin = (
        await mock_prisma_client.db.litellm_claudecodeplugintable.find_unique(
            where={"name": plugin_name}
        )
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
    our_plugin = next((p for p in body["plugins"] if p["name"] == plugin_name), None)
    assert our_plugin is not None
    assert our_plugin["source"] == {
        "source": "github",
        "repo": "test-org/marketplace-test",
    }
    assert our_plugin["version"] == "2.0.0"

    # Cleanup
    await mock_prisma_client.db.litellm_claudecodeplugintable.delete(
        where={"name": plugin_name}
    )


@pytest.mark.asyncio
async def test_get_marketplace_no_key_unaffected_by_imported_disabled_skills(
    mock_prisma_client,
):
    """Backward-compat regression test: an unauthenticated request to
    marketplace.json must be byte-for-byte the same as before the
    marketplace-import feature existed - it only ever sees enabled=True rows,
    even once a marketplace sync has imported (and left disabled) new skills.
    """
    setattr(litellm.proxy.proxy_server, "prisma_client", mock_prisma_client)
    setattr(litellm.proxy.proxy_server, "master_key", "sk-1234")
    await litellm.proxy.proxy_server.prisma_client.connect()

    public_plugin_name = f"public-plugin-{int(time.time())}"
    imported_disabled_name = f"imported-private-skill-{int(time.time())}"

    user_api_key_dict = UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN,
        api_key="sk-1234",
        user_id="test-user",
    )
    await register_plugin(
        request=RegisterPluginRequest(
            name=public_plugin_name,
            source={"source": "github", "repo": "test-org/public-plugin"},
            version="1.0.0",
            description="Always-public plugin",
        ),
        user_api_key_dict=user_api_key_dict,
    )
    # Simulate a marketplace sync having imported a skill, left disabled
    # pending admin opt-in (mirrors resolve_and_sync's upsert semantics).
    await mock_prisma_client.db.litellm_claudecodeplugintable.create(
        data={
            "name": imported_disabled_name,
            "version": "1.0.0",
            "description": "Imported but not yet enabled",
            "manifest_json": json.dumps(
                {"source": {"source": "github", "repo": "org/private-skill"}}
            ),
            "enabled": False,
        }
    )

    response_before_key_check = await get_marketplace(user_api_key_dict=None)
    body_before = json.loads(response_before_key_check.body.decode())

    response_no_key = await get_marketplace(user_api_key_dict=None)
    body_no_key = json.loads(response_no_key.body.decode())

    # No-key responses must be deterministic/identical across calls, and must
    # never include the imported-but-disabled skill.
    assert body_no_key == body_before
    names = {p["name"] for p in body_no_key["plugins"]}
    assert public_plugin_name in names
    assert imported_disabled_name not in names

    # Cleanup
    await mock_prisma_client.db.litellm_claudecodeplugintable.delete(
        where={"name": public_plugin_name}
    )
    await mock_prisma_client.db.litellm_claudecodeplugintable.delete(
        where={"name": imported_disabled_name}
    )


@pytest.mark.asyncio
async def test_get_marketplace_with_key_unlocks_allowed_imported_skill(
    mock_prisma_client,
):
    """A key whose object_permission.allowed_skills grants an imported
    (still disabled) skill sees that skill in marketplace.json alongside the
    always-public entries - without a key, that same skill stays hidden."""
    from litellm.models.object_permission import LiteLLM_ObjectPermissionTable

    setattr(litellm.proxy.proxy_server, "prisma_client", mock_prisma_client)
    setattr(litellm.proxy.proxy_server, "master_key", "sk-1234")
    await litellm.proxy.proxy_server.prisma_client.connect()

    imported_disabled_name = f"imported-scoped-skill-{int(time.time())}"
    await mock_prisma_client.db.litellm_claudecodeplugintable.create(
        data={
            "name": imported_disabled_name,
            "version": "1.0.0",
            "description": "Imported, granted to one key only",
            "manifest_json": json.dumps(
                {"source": {"source": "github", "repo": "org/scoped-skill"}}
            ),
            "enabled": False,
        }
    )

    response_no_key = await get_marketplace(user_api_key_dict=None)
    body_no_key = json.loads(response_no_key.body.decode())
    assert imported_disabled_name not in {p["name"] for p in body_no_key["plugins"]}

    scoped_key = UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER,
        api_key="sk-scoped",
        user_id="scoped-user",
        object_permission=LiteLLM_ObjectPermissionTable(
            object_permission_id="perm-1", allowed_skills=[imported_disabled_name]
        ),
    )
    response_with_key = await get_marketplace(user_api_key_dict=scoped_key)
    body_with_key = json.loads(response_with_key.body.decode())
    names_with_key = {p["name"] for p in body_with_key["plugins"]}
    assert imported_disabled_name in names_with_key

    # Cleanup
    await mock_prisma_client.db.litellm_claudecodeplugintable.delete(
        where={"name": imported_disabled_name}
    )


@pytest.mark.asyncio
async def test_get_marketplace_hides_skill_from_disabled_marketplace_even_with_grant(
    mock_prisma_client,
):
    """Regression test: allowed_skills is a standing grant on a key/team/org's
    object_permission - it doesn't get cleared just because an admin later
    disables the marketplace that owned the skill (DELETE
    /claude-code/marketplaces/{name}, which cascades plugin.enabled=False but
    leaves any pre-existing allowed_skills grants untouched). Without
    filtering on the owning marketplace's own enabled state, a previously
    granted key would keep seeing a skill from a marketplace an admin
    explicitly shut off."""
    from litellm.models.object_permission import LiteLLM_ObjectPermissionTable

    setattr(litellm.proxy.proxy_server, "prisma_client", mock_prisma_client)
    setattr(litellm.proxy.proxy_server, "master_key", "sk-1234")
    await litellm.proxy.proxy_server.prisma_client.connect()

    marketplace_id = f"marketplace-{int(time.time())}"
    mock_prisma_client._marketplaces_store[marketplace_id] = MockMarketplaceRecord(
        id=marketplace_id, name="untrusted-marketplace", enabled=False
    )

    skill_name = f"untrusted-marketplace--skill-{int(time.time())}"
    await mock_prisma_client.db.litellm_claudecodeplugintable.create(
        data={
            "name": skill_name,
            "version": "1.0.0",
            "description": "Skill owned by a since-disabled marketplace",
            "manifest_json": json.dumps(
                {"source": {"source": "github", "repo": "org/untrusted-skill"}}
            ),
            "enabled": True,  # cascade-disable races aside, this asserts the belt-and-suspenders check too
            "marketplace_id": marketplace_id,
        }
    )

    granted_key = UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER,
        api_key="sk-granted",
        user_id="granted-user",
        object_permission=LiteLLM_ObjectPermissionTable(
            object_permission_id="perm-2", allowed_skills=[skill_name]
        ),
    )
    response = await get_marketplace(user_api_key_dict=granted_key)
    body = json.loads(response.body.decode())

    assert skill_name not in {p["name"] for p in body["plugins"]}

    # Cleanup
    await mock_prisma_client.db.litellm_claudecodeplugintable.delete(where={"name": skill_name})
    del mock_prisma_client._marketplaces_store[marketplace_id]


@pytest.mark.asyncio
async def test_register_plugin_git_subdir(mock_prisma_client):
    """Test registering a plugin with git-subdir source type."""
    setattr(litellm.proxy.proxy_server, "prisma_client", mock_prisma_client)
    setattr(litellm.proxy.proxy_server, "master_key", "sk-1234")

    await litellm.proxy.proxy_server.prisma_client.connect()

    plugin_name = f"test-subdir-plugin-{int(time.time())}"

    request = RegisterPluginRequest(
        name=plugin_name,
        source={
            "source": "git-subdir",
            "url": "https://github.com/test-org/monorepo.git",
            "path": "plugins/my-plugin",
        },
        version="1.0.0",
        description="Test git-subdir plugin",
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
    assert response["plugin"]["source"]["source"] == "git-subdir"
    assert (
        response["plugin"]["source"]["url"]
        == "https://github.com/test-org/monorepo.git"
    )
    assert response["plugin"]["source"]["path"] == "plugins/my-plugin"
    assert response["plugin"]["enabled"] is True

    # Cleanup
    await mock_prisma_client.db.litellm_claudecodeplugintable.delete(
        where={"name": plugin_name}
    )
