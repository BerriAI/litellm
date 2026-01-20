"""
Tests for Claude Code Marketplace endpoints.

Tests:
1. Register a plugin
2. Get marketplace.json (list enabled plugins)
"""

import os
import sys
import time

import pytest

sys.path.insert(0, os.path.abspath("../.."))

import litellm
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.proxy_server import LitellmUserRoles
from litellm.proxy.utils import PrismaClient, ProxyLogging
from litellm.caching.caching import DualCache
from litellm.types.proxy.claude_code_endpoints import RegisterPluginRequest

# Import the functions we're testing
from litellm.proxy.anthropic_endpoints.claude_code_endpoints.claude_code_marketplace import (
    register_plugin,
    get_marketplace,
)

proxy_logging_obj = ProxyLogging(user_api_key_cache=DualCache())


@pytest.fixture
def prisma_client():
    from litellm.proxy.proxy_cli import append_query_params

    params = {"connection_limit": 100, "pool_timeout": 60}
    database_url = os.getenv("DATABASE_URL")
    modified_url = append_query_params(database_url, params)
    os.environ["DATABASE_URL"] = modified_url

    prisma_client = PrismaClient(
        database_url=os.environ["DATABASE_URL"], proxy_logging_obj=proxy_logging_obj
    )

    litellm.proxy.proxy_server.litellm_proxy_budget_name = (
        f"litellm-proxy-budget-{time.time()}"
    )

    return prisma_client


@pytest.mark.asyncio
async def test_register_plugin(prisma_client):
    """Test registering a plugin in the marketplace."""
    setattr(litellm.proxy.proxy_server, "prisma_client", prisma_client)
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

    # Cleanup - delete the plugin
    await prisma_client.db.litellm_claudecodeplugintable.delete(
        where={"name": plugin_name}
    )


@pytest.mark.asyncio
async def test_get_marketplace(prisma_client):
    """Test getting marketplace.json with registered plugins."""
    setattr(litellm.proxy.proxy_server, "prisma_client", prisma_client)
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
    import json
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
    await prisma_client.db.litellm_claudecodeplugintable.delete(
        where={"name": plugin_name}
    )
