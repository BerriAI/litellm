"""
Unit tests for claude_code_marketplace.py source validation.

Covers the git-subdir source type added alongside the existing github and url types.
"""

import json

import pytest
from fastapi import HTTPException
from unittest.mock import AsyncMock, MagicMock

import litellm
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.proxy_server import LitellmUserRoles
from litellm.types.proxy.claude_code_endpoints import (
    RegisterPluginRequest,
    RejectPluginRequest,
    UpdatePluginRequest,
)
from litellm.proxy.anthropic_endpoints.claude_code_endpoints.claude_code_marketplace import (
    approve_plugin,
    delete_plugin,
    disable_plugin,
    enable_plugin,
    get_marketplace,
    get_plugin,
    list_plugins,
    register_plugin,
    reject_plugin,
    update_plugin,
)


def _make_mock_prisma():
    """Stateful prisma mock that supports find_unique, find_many, create, and update."""
    store: dict = {}

    mock_client = MagicMock()
    mock_client.proxy_logging_obj = MagicMock()
    mock_table = MagicMock()

    async def _find_unique(where):
        return store.get(where.get("name"))

    def _matches(record, where):
        for key, expected in where.items():
            if key == "OR":
                if not any(_matches(record, clause) for clause in expected):
                    return False
                continue
            actual = getattr(record, key)
            if isinstance(expected, dict) and "in" in expected:
                if actual not in expected["in"]:
                    return False
            elif actual != expected:
                return False
        return True

    async def _find_many(where=None):
        records = list(store.values())
        if not where:
            return records
        return [r for r in records if _matches(r, where)]

    async def _create(data):
        record = MagicMock()
        record.id = f"test-id-{data['name']}"
        record.name = data["name"]
        record.version = data.get("version")
        record.description = data.get("description")
        record.manifest_json = data.get("manifest_json", "{}")
        record.enabled = data.get("enabled", True)
        record.approval_status = data.get("approval_status", "active")
        record.review_notes = data.get("review_notes")
        record.reviewed_by = data.get("reviewed_by")
        record.reviewed_at = data.get("reviewed_at")
        record.created_by = data.get("created_by")
        record.created_at = data.get("created_at")
        record.updated_at = data.get("updated_at")
        store[data["name"]] = record
        return record

    async def _update(where, data):
        record = store[where["name"]]
        for k, v in data.items():
            setattr(record, k, v)
        return record

    mock_table.find_unique = AsyncMock(side_effect=_find_unique)
    mock_table.find_many = AsyncMock(side_effect=_find_many)
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


@pytest.fixture(autouse=True)
def _patch_proxy_globals(monkeypatch):
    """Scope prisma_client/master_key mutations to each test via monkeypatch."""
    monkeypatch.setattr(litellm.proxy.proxy_server, "prisma_client", _make_mock_prisma())
    monkeypatch.setattr(litellm.proxy.proxy_server, "master_key", "sk-1234")


@pytest.mark.asyncio
async def test_register_plugin_git_subdir_success():
    """git-subdir with both url and path fields registers successfully."""
    request = RegisterPluginRequest(name="my-monorepo-plugin", source=_GIT_SUBDIR_SOURCE)

    response = await register_plugin(request=request, user_api_key_dict=_USER)

    assert response.status == "success"
    assert response.action == "created"
    assert response.plugin.source["source"] == "git-subdir"
    assert response.plugin.source["path"] == "plugins/my-plugin"


async def _read_stored_manifest(name: str) -> dict:
    table = litellm.proxy.proxy_server.prisma_client.db.litellm_claudecodeplugintable
    record = await table.find_unique(where={"name": name})
    return json.loads(record.manifest_json)


@pytest.mark.asyncio
async def test_register_plugin_duplicate_name_conflicts():
    """A second POST with an existing name returns 409 and leaves the stored plugin untouched."""
    name = "my-monorepo-plugin"
    await register_plugin(
        request=RegisterPluginRequest(name=name, source=_GIT_SUBDIR_SOURCE, version="1.0.0"),
        user_api_key_dict=_USER,
    )

    stored_before = await _read_stored_manifest(name)
    assert stored_before["version"] == "1.0.0"

    conflicting = RegisterPluginRequest(
        name=name,
        source={
            "source": "git-subdir",
            "url": "https://github.com/org/other.git",
            "path": "plugins/other-plugin",
        },
        version="2.0.0",
    )
    with pytest.raises(HTTPException) as exc_info:
        await register_plugin(request=conflicting, user_api_key_dict=_USER)

    assert exc_info.value.status_code == 409
    assert "already exists" in exc_info.value.detail["error"]

    stored_after = await _read_stored_manifest(name)
    assert stored_after == stored_before
    assert stored_after["version"] == "1.0.0"
    assert stored_after["source"]["url"] == "https://github.com/org/monorepo.git"


@pytest.mark.asyncio
async def test_update_plugin_replaces_existing_source():
    """PUT updates an existing plugin: action=updated and the stored source is replaced."""
    name = "my-monorepo-plugin"
    await register_plugin(
        request=RegisterPluginRequest(name=name, source=_GIT_SUBDIR_SOURCE, version="1.0.0"),
        user_api_key_dict=_USER,
    )

    new_source = {"source": "github", "repo": "org/replacement"}
    response = await update_plugin(
        plugin_name=name,
        request=UpdatePluginRequest(source=new_source, version="2.0.0", description="updated"),
        user_api_key_dict=_USER,
    )

    assert response.status == "success"
    assert response.action == "updated"
    assert response.plugin.version == "2.0.0"
    assert response.plugin.source == new_source

    stored = await _read_stored_manifest(name)
    assert stored["source"] == new_source
    assert stored["version"] == "2.0.0"


@pytest.mark.asyncio
async def test_update_plugin_not_found():
    """PUT on a name that does not exist raises HTTP 404."""
    with pytest.raises(HTTPException) as exc_info:
        await update_plugin(
            plugin_name="does-not-exist",
            request=UpdatePluginRequest(source=_GIT_SUBDIR_SOURCE),
            user_api_key_dict=_USER,
        )

    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_register_plugin_create_race_maps_unique_violation_to_409():
    """A concurrent insert that slips past the find_unique pre-check (create raises
    the unique-constraint error) is mapped to 409, not surfaced as a 500."""
    from prisma.errors import UniqueViolationError

    table = litellm.proxy.proxy_server.prisma_client.db.litellm_claudecodeplugintable
    table.create = AsyncMock(side_effect=UniqueViolationError({}, message="duplicate name"))

    with pytest.raises(HTTPException) as exc_info:
        await register_plugin(
            request=RegisterPluginRequest(name="racy-plugin", source=_GIT_SUBDIR_SOURCE),
            user_api_key_dict=_USER,
        )

    assert exc_info.value.status_code == 409
    assert "already exists" in exc_info.value.detail["error"]


@pytest.mark.asyncio
async def test_update_plugin_db_error_maps_to_structured_500():
    """A data-layer failure during the update (e.g. a dropped DB connection) is caught and
    returned as a structured 500, not swallowed silently or leaked as an unhandled error."""
    from prisma.errors import PrismaError

    name = "my-monorepo-plugin"
    await register_plugin(
        request=RegisterPluginRequest(name=name, source=_GIT_SUBDIR_SOURCE, version="1.0.0"),
        user_api_key_dict=_USER,
    )

    table = litellm.proxy.proxy_server.prisma_client.db.litellm_claudecodeplugintable
    table.update = AsyncMock(side_effect=PrismaError("connection lost"))

    with pytest.raises(HTTPException) as exc_info:
        await update_plugin(
            plugin_name=name,
            request=UpdatePluginRequest(source={"source": "github", "repo": "org/replacement"}),
            user_api_key_dict=_USER,
        )

    assert exc_info.value.status_code == 500
    assert "connection lost" in exc_info.value.detail["error"]


@pytest.mark.asyncio
async def test_get_marketplace_skips_plugin_with_null_manifest():
    await register_plugin(
        request=RegisterPluginRequest(name="good-plugin", source=_GIT_SUBDIR_SOURCE, version="1.0.0"),
        user_api_key_dict=_USER,
    )

    table = litellm.proxy.proxy_server.prisma_client.db.litellm_claudecodeplugintable
    await table.create(data={"name": "null-manifest-plugin", "manifest_json": None, "enabled": True})

    response = await get_marketplace()

    assert response.status_code == 200
    body = json.loads(response.body)
    assert [plugin["name"] for plugin in body["plugins"]] == ["good-plugin"]


@pytest.mark.asyncio
async def test_register_plugin_git_subdir_missing_url():
    """git-subdir without url field raises HTTP 400."""
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
    request = RegisterPluginRequest(
        name="bad-plugin",
        source={
            "source": "git-subdir",
            "url": "https://github.com/org/monorepo.git",
            "path": "",
        },
    )

    with pytest.raises(HTTPException) as exc_info:
        await register_plugin(request=request, user_api_key_dict=_USER)

    assert exc_info.value.status_code == 400
    assert "path" in exc_info.value.detail["error"]


@pytest.mark.asyncio
async def test_register_plugin_git_subdir_path_traversal():
    """git-subdir with path traversal segments raises HTTP 400."""
    for bad_path in [
        "../../etc/passwd",
        "../secrets",
        "/absolute/path",
        "plugins\\..\\..\\secrets",  # backslash traversal
        "plugins/%2e%2e/secrets",  # percent-encoded traversal
        "plugins/%2E%2E/secrets",  # uppercase percent-encoded traversal
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
    request = RegisterPluginRequest(
        name="bad-plugin",
        source={"source": "ftp", "url": "ftp://example.com/repo"},
    )

    with pytest.raises(HTTPException) as exc_info:
        await register_plugin(request=request, user_api_key_dict=_USER)

    assert exc_info.value.status_code == 400
    assert "git-subdir" in exc_info.value.detail["error"]


_SUBMITTER = UserAPIKeyAuth(
    user_role=LitellmUserRoles.INTERNAL_USER,
    api_key="sk-submitter",
    user_id="submitter-user",
)

_OTHER_USER = UserAPIKeyAuth(
    user_role=LitellmUserRoles.INTERNAL_USER,
    api_key="sk-other",
    user_id="other-user",
)


async def _stored(name: str):
    table = litellm.proxy.proxy_server.prisma_client.db.litellm_claudecodeplugintable
    return await table.find_unique(where={"name": name})


@pytest.mark.asyncio
async def test_non_admin_submission_is_pending_and_unpublished():
    response = await register_plugin(
        request=RegisterPluginRequest(name="submitted-skill", source=_GIT_SUBDIR_SOURCE),
        user_api_key_dict=_SUBMITTER,
    )

    assert response.action == "submitted_for_review"
    assert response.plugin.approval_status == "pending_review"
    assert response.plugin.enabled is False

    stored = await _stored("submitted-skill")
    assert stored.approval_status == "pending_review"
    assert stored.enabled is False
    assert stored.created_by == "submitter-user"


@pytest.mark.asyncio
async def test_admin_registration_stays_auto_approved():
    response = await register_plugin(
        request=RegisterPluginRequest(name="admin-skill", source=_GIT_SUBDIR_SOURCE),
        user_api_key_dict=_USER,
    )

    assert response.action == "created"
    assert response.plugin.approval_status == "active"
    assert response.plugin.enabled is True


@pytest.mark.asyncio
async def test_pending_submission_is_not_served_by_marketplace():
    await register_plugin(
        request=RegisterPluginRequest(name="submitted-skill", source=_GIT_SUBDIR_SOURCE),
        user_api_key_dict=_SUBMITTER,
    )
    await register_plugin(
        request=RegisterPluginRequest(name="admin-skill", source=_GIT_SUBDIR_SOURCE),
        user_api_key_dict=_USER,
    )

    body = json.loads((await get_marketplace()).body)

    assert [plugin["name"] for plugin in body["plugins"]] == ["admin-skill"]


@pytest.mark.asyncio
async def test_approval_publishes_submission_and_records_reviewer():
    await register_plugin(
        request=RegisterPluginRequest(name="submitted-skill", source=_GIT_SUBDIR_SOURCE),
        user_api_key_dict=_SUBMITTER,
    )

    response = await approve_plugin(plugin_name="submitted-skill", user_api_key_dict=_USER)

    assert response.approval_status == "active"
    assert response.enabled is True
    assert response.reviewed_by == "test-user"
    assert response.reviewed_at is not None

    body = json.loads((await get_marketplace()).body)
    assert [plugin["name"] for plugin in body["plugins"]] == ["submitted-skill"]


@pytest.mark.asyncio
async def test_rejection_keeps_submission_unpublished_with_notes():
    await register_plugin(
        request=RegisterPluginRequest(name="submitted-skill", source=_GIT_SUBDIR_SOURCE),
        user_api_key_dict=_SUBMITTER,
    )

    response = await reject_plugin(
        plugin_name="submitted-skill",
        request=RejectPluginRequest(review_notes="point at the skill folder"),
        user_api_key_dict=_USER,
    )

    assert response.approval_status == "rejected"
    assert response.enabled is False

    body = json.loads((await get_marketplace()).body)
    assert body["plugins"] == []

    stored = await _stored("submitted-skill")
    assert stored.review_notes == "point at the skill folder"


@pytest.mark.asyncio
async def test_non_admin_cannot_review():
    await register_plugin(
        request=RegisterPluginRequest(name="submitted-skill", source=_GIT_SUBDIR_SOURCE),
        user_api_key_dict=_SUBMITTER,
    )

    with pytest.raises(HTTPException) as exc_info:
        await approve_plugin(plugin_name="submitted-skill", user_api_key_dict=_SUBMITTER)

    assert exc_info.value.status_code == 403
    stored = await _stored("submitted-skill")
    assert stored.approval_status == "pending_review"
    assert stored.enabled is False


@pytest.mark.asyncio
async def test_review_unknown_skill_returns_404():
    with pytest.raises(HTTPException) as exc_info:
        await approve_plugin(plugin_name="does-not-exist", user_api_key_dict=_USER)

    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_submitter_sees_own_pending_skill_but_not_another_users():
    await register_plugin(
        request=RegisterPluginRequest(name="submitted-skill", source=_GIT_SUBDIR_SOURCE),
        user_api_key_dict=_SUBMITTER,
    )
    await register_plugin(
        request=RegisterPluginRequest(name="other-skill", source=_GIT_SUBDIR_SOURCE),
        user_api_key_dict=_OTHER_USER,
    )
    await register_plugin(
        request=RegisterPluginRequest(name="admin-skill", source=_GIT_SUBDIR_SOURCE),
        user_api_key_dict=_USER,
    )

    submitter_view = await list_plugins(user_api_key_dict=_SUBMITTER)
    admin_view = await list_plugins(user_api_key_dict=_USER)

    assert sorted(p.name for p in submitter_view.plugins) == ["admin-skill", "submitted-skill"]
    assert sorted(p.name for p in admin_view.plugins) == ["admin-skill", "other-skill", "submitted-skill"]


@pytest.mark.asyncio
async def test_admin_can_filter_the_pending_queue():
    await register_plugin(
        request=RegisterPluginRequest(name="submitted-skill", source=_GIT_SUBDIR_SOURCE),
        user_api_key_dict=_SUBMITTER,
    )
    await register_plugin(
        request=RegisterPluginRequest(name="admin-skill", source=_GIT_SUBDIR_SOURCE),
        user_api_key_dict=_USER,
    )

    queue = await list_plugins(approval_status="pending_review", user_api_key_dict=_USER)

    assert [p.name for p in queue.plugins] == ["submitted-skill"]


@pytest.mark.asyncio
async def test_get_plugin_hides_another_users_pending_submission():
    await register_plugin(
        request=RegisterPluginRequest(name="submitted-skill", source=_GIT_SUBDIR_SOURCE),
        user_api_key_dict=_SUBMITTER,
    )

    with pytest.raises(HTTPException) as exc_info:
        await get_plugin(plugin_name="submitted-skill", user_api_key_dict=_OTHER_USER)

    assert exc_info.value.status_code == 404

    own = await get_plugin(plugin_name="submitted-skill", user_api_key_dict=_SUBMITTER)
    assert own["approval_status"] == "pending_review"


@pytest.mark.asyncio
async def test_enable_cannot_bypass_review():
    await register_plugin(
        request=RegisterPluginRequest(name="submitted-skill", source=_GIT_SUBDIR_SOURCE),
        user_api_key_dict=_SUBMITTER,
    )

    with pytest.raises(HTTPException) as exc_info:
        await enable_plugin(plugin_name="submitted-skill", user_api_key_dict=_USER)

    assert exc_info.value.status_code == 409
    stored = await _stored("submitted-skill")
    assert stored.enabled is False


@pytest.mark.asyncio
async def test_non_admin_cannot_publish_or_unpublish():
    await register_plugin(
        request=RegisterPluginRequest(name="admin-skill", source=_GIT_SUBDIR_SOURCE),
        user_api_key_dict=_USER,
    )

    with pytest.raises(HTTPException) as exc_info:
        await disable_plugin(plugin_name="admin-skill", user_api_key_dict=_SUBMITTER)

    assert exc_info.value.status_code == 403
    stored = await _stored("admin-skill")
    assert stored.enabled is True


@pytest.mark.asyncio
async def test_submitter_edit_sends_skill_back_to_review():
    await register_plugin(
        request=RegisterPluginRequest(name="submitted-skill", source=_GIT_SUBDIR_SOURCE),
        user_api_key_dict=_SUBMITTER,
    )
    await approve_plugin(plugin_name="submitted-skill", user_api_key_dict=_USER)

    await update_plugin(
        plugin_name="submitted-skill",
        request=UpdatePluginRequest(source={"source": "github", "repo": "org/changed"}),
        user_api_key_dict=_SUBMITTER,
    )

    stored = await _stored("submitted-skill")
    assert stored.approval_status == "pending_review"
    assert stored.enabled is False


@pytest.mark.asyncio
async def test_unrelated_user_cannot_edit_or_delete_a_submission():
    await register_plugin(
        request=RegisterPluginRequest(name="submitted-skill", source=_GIT_SUBDIR_SOURCE),
        user_api_key_dict=_SUBMITTER,
    )

    with pytest.raises(HTTPException) as update_exc:
        await update_plugin(
            plugin_name="submitted-skill",
            request=UpdatePluginRequest(source={"source": "github", "repo": "org/hijacked"}),
            user_api_key_dict=_OTHER_USER,
        )
    assert update_exc.value.status_code == 403

    with pytest.raises(HTTPException) as delete_exc:
        await delete_plugin(plugin_name="submitted-skill", user_api_key_dict=_OTHER_USER)
    assert delete_exc.value.status_code == 403

    assert await _stored("submitted-skill") is not None


@pytest.mark.asyncio
async def test_approving_an_already_active_skill_is_rejected():
    await register_plugin(
        request=RegisterPluginRequest(name="admin-skill", source=_GIT_SUBDIR_SOURCE),
        user_api_key_dict=_USER,
    )

    with pytest.raises(HTTPException) as exc_info:
        await approve_plugin(plugin_name="admin-skill", user_api_key_dict=_USER)

    assert exc_info.value.status_code == 400
