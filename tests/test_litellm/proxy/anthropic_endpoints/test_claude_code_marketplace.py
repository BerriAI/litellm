"""
Unit tests for claude_code_marketplace.py source validation.

Covers the git-subdir source type added alongside the existing github and url types.
"""

import pytest
from fastapi import HTTPException
from unittest.mock import AsyncMock, MagicMock

import litellm
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.proxy_server import LitellmUserRoles
from litellm.types.proxy.claude_code_endpoints import RegisterPluginRequest
from litellm.proxy.anthropic_endpoints.claude_code_endpoints.claude_code_marketplace import (
    register_plugin,
)


def _make_mock_prisma():
    """Stateful prisma mock that supports find_unique, create, and update."""
    store: dict = {}

    mock_client = MagicMock()
    mock_client.proxy_logging_obj = MagicMock()
    mock_table = MagicMock()

    async def _find_unique(where):
        return store.get(where.get("name"))

    async def _create(data):
        record = MagicMock()
        record.id = "test-id"
        record.name = data["name"]
        record.version = data.get("version")
        record.description = data.get("description")
        record.manifest_json = data.get("manifest_json", "{}")
        record.enabled = data.get("enabled", True)
        store[data["name"]] = record
        return record

    async def _update(where, data):
        record = store[where["name"]]
        for k, v in data.items():
            setattr(record, k, v)
        return record

    mock_table.find_unique = AsyncMock(side_effect=_find_unique)
    mock_table.create = AsyncMock(side_effect=_create)
    mock_table.update = AsyncMock(side_effect=_update)
    mock_client.db.litellm_claudecodeplugintable = mock_table
    return mock_client


_USER = UserAPIKeyAuth(
    user_role=LitellmUserRoles.PROXY_ADMIN,
    api_key="sk-1234",
    user_id="test-user",
)

_GIT_SUBDIR_SOURCE = {
    "source": "git-subdir",
    "url": "https://github.com/org/monorepo.git",
    "path": "plugins/my-plugin",
}


@pytest.mark.asyncio
async def test_register_plugin_git_subdir_success():
    """git-subdir with both url and path fields registers successfully."""
    setattr(litellm.proxy.proxy_server, "prisma_client", _make_mock_prisma())
    setattr(litellm.proxy.proxy_server, "master_key", "sk-1234")

    request = RegisterPluginRequest(name="my-monorepo-plugin", source=_GIT_SUBDIR_SOURCE)

    response = await register_plugin(request=request, user_api_key_dict=_USER)

    assert response["status"] == "success"
    assert response["action"] == "created"
    assert response["plugin"]["source"]["source"] == "git-subdir"
    assert response["plugin"]["source"]["path"] == "plugins/my-plugin"


@pytest.mark.asyncio
async def test_register_plugin_git_subdir_update():
    """Registering the same git-subdir plugin twice returns action=updated."""
    setattr(litellm.proxy.proxy_server, "prisma_client", _make_mock_prisma())
    setattr(litellm.proxy.proxy_server, "master_key", "sk-1234")

    request = RegisterPluginRequest(
        name="my-monorepo-plugin", source=_GIT_SUBDIR_SOURCE, version="1.0.0"
    )
    await register_plugin(request=request, user_api_key_dict=_USER)

    request2 = RegisterPluginRequest(
        name="my-monorepo-plugin", source=_GIT_SUBDIR_SOURCE, version="2.0.0"
    )
    response = await register_plugin(request=request2, user_api_key_dict=_USER)

    assert response["status"] == "success"
    assert response["action"] == "updated"


@pytest.mark.asyncio
async def test_register_plugin_git_subdir_missing_url():
    """git-subdir without url field raises HTTP 400."""
    setattr(litellm.proxy.proxy_server, "prisma_client", _make_mock_prisma())
    setattr(litellm.proxy.proxy_server, "master_key", "sk-1234")

    request = RegisterPluginRequest(
        name="bad-plugin",
        source={"source": "git-subdir", "path": "plugins/my-plugin"},
    )

    with pytest.raises(HTTPException) as exc_info:
        await register_plugin(request=request, user_api_key_dict=_USER)

    assert exc_info.value.status_code == 400
    assert "url" in exc_info.value.detail["error"]


@pytest.mark.asyncio
async def test_register_plugin_git_subdir_empty_url():
    """git-subdir with empty url raises HTTP 400."""
    setattr(litellm.proxy.proxy_server, "prisma_client", _make_mock_prisma())
    setattr(litellm.proxy.proxy_server, "master_key", "sk-1234")

    request = RegisterPluginRequest(
        name="bad-plugin",
        source={"source": "git-subdir", "url": "", "path": "plugins/my-plugin"},
    )

    with pytest.raises(HTTPException) as exc_info:
        await register_plugin(request=request, user_api_key_dict=_USER)

    assert exc_info.value.status_code == 400
    assert "url" in exc_info.value.detail["error"]


@pytest.mark.asyncio
async def test_register_plugin_git_subdir_missing_path():
    """git-subdir without path field raises HTTP 400."""
    setattr(litellm.proxy.proxy_server, "prisma_client", _make_mock_prisma())
    setattr(litellm.proxy.proxy_server, "master_key", "sk-1234")

    request = RegisterPluginRequest(
        name="bad-plugin",
        source={"source": "git-subdir", "url": "https://github.com/org/monorepo.git"},
    )

    with pytest.raises(HTTPException) as exc_info:
        await register_plugin(request=request, user_api_key_dict=_USER)

    assert exc_info.value.status_code == 400
    assert "path" in exc_info.value.detail["error"]


@pytest.mark.asyncio
async def test_register_plugin_git_subdir_empty_path():
    """git-subdir with empty path raises HTTP 400."""
    setattr(litellm.proxy.proxy_server, "prisma_client", _make_mock_prisma())
    setattr(litellm.proxy.proxy_server, "master_key", "sk-1234")

    request = RegisterPluginRequest(
        name="bad-plugin",
        source={"source": "git-subdir", "url": "https://github.com/org/monorepo.git", "path": ""},
    )

    with pytest.raises(HTTPException) as exc_info:
        await register_plugin(request=request, user_api_key_dict=_USER)

    assert exc_info.value.status_code == 400
    assert "path" in exc_info.value.detail["error"]


@pytest.mark.asyncio
async def test_register_plugin_git_subdir_path_traversal():
    """git-subdir with path traversal segments raises HTTP 400."""
    setattr(litellm.proxy.proxy_server, "prisma_client", _make_mock_prisma())
    setattr(litellm.proxy.proxy_server, "master_key", "sk-1234")

    for bad_path in [
        "../../etc/passwd",
        "../secrets",
        "/absolute/path",
        "plugins\\..\\..\\secrets",  # backslash traversal
        "plugins/%2e%2e/secrets",   # percent-encoded traversal
        "plugins/%2E%2E/secrets",   # uppercase percent-encoded traversal
        "plugins/%252e%252e/secrets",  # double-encoded traversal
    ]:
        request = RegisterPluginRequest(
            name="bad-plugin",
            source={
                "source": "git-subdir",
                "url": "https://github.com/org/monorepo.git",
                "path": bad_path,
            },
        )

        with pytest.raises(HTTPException) as exc_info:
            await register_plugin(request=request, user_api_key_dict=_USER)

        assert exc_info.value.status_code == 400
        assert "relative" in exc_info.value.detail["error"]


@pytest.mark.asyncio
async def test_register_plugin_unknown_source_type():
    """Unknown source type raises HTTP 400 listing all valid types."""
    setattr(litellm.proxy.proxy_server, "prisma_client", _make_mock_prisma())
    setattr(litellm.proxy.proxy_server, "master_key", "sk-1234")

    request = RegisterPluginRequest(
        name="bad-plugin",
        source={"source": "ftp", "url": "ftp://example.com/repo"},
    )

    with pytest.raises(HTTPException) as exc_info:
        await register_plugin(request=request, user_api_key_dict=_USER)

    assert exc_info.value.status_code == 400
    assert "git-subdir" in exc_info.value.detail["error"]
