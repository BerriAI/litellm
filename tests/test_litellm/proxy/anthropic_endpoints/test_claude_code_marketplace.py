"""
Unit tests for claude_code_marketplace.py source validation.

Covers the git-subdir source type added alongside the existing github and url types.
"""

import json
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from unittest.mock import AsyncMock, MagicMock

import litellm
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.common_utils.resource_ownership import (
    get_primary_resource_owner_scope,
    get_resource_owner_scopes,
)
from litellm.proxy.proxy_server import LitellmUserRoles
from litellm.types.proxy.claude_code_endpoints import (
    ApprovePluginRequest,
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

    async def _update_many(where, data):
        matched = [record for record in store.values() if _matches(record, where)]
        for record in matched:
            for k, v in data.items():
                setattr(record, k, v)
        return len(matched)

    mock_table.find_unique = AsyncMock(side_effect=_find_unique)
    mock_table.find_many = AsyncMock(side_effect=_find_many)
    mock_table.create = AsyncMock(side_effect=_create)
    mock_table.update = AsyncMock(side_effect=_update)
    mock_table.update_many = AsyncMock(side_effect=_update_many)
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


async def _fingerprint(name: str) -> str:
    """Read the fingerprint the way a reviewer does, off the skill they are looking at."""
    details = await get_plugin(plugin_name=name, user_api_key_dict=_USER)
    return details["manifest_fingerprint"]


async def _approve(name: str, *, reviewer=None, fingerprint: str | None = None):
    return await approve_plugin(
        plugin_name=name,
        request=ApprovePluginRequest(
            reviewed_fingerprint=fingerprint if fingerprint is not None else await _fingerprint(name)
        ),
        user_api_key_dict=reviewer if reviewer is not None else _USER,
    )


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

    response = await _approve("submitted-skill")

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
        await _approve("submitted-skill", reviewer=_SUBMITTER)

    assert exc_info.value.status_code == 403
    stored = await _stored("submitted-skill")
    assert stored.approval_status == "pending_review"
    assert stored.enabled is False


@pytest.mark.asyncio
async def test_review_unknown_skill_returns_404():
    with pytest.raises(HTTPException) as exc_info:
        await _approve("does-not-exist", fingerprint="any-fingerprint")

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
    await _approve("submitted-skill")

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
    """404 rather than 403, since a pending skill is hidden from this caller and the status must not out it."""
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
    assert update_exc.value.status_code == 404

    with pytest.raises(HTTPException) as delete_exc:
        await delete_plugin(plugin_name="submitted-skill", user_api_key_dict=_OTHER_USER)
    assert delete_exc.value.status_code == 404

    assert await _stored("submitted-skill") is not None


@pytest.mark.asyncio
async def test_approving_an_already_active_skill_is_rejected():
    await register_plugin(
        request=RegisterPluginRequest(name="admin-skill", source=_GIT_SUBDIR_SOURCE),
        user_api_key_dict=_USER,
    )

    with pytest.raises(HTTPException) as exc_info:
        await _approve("admin-skill")

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_approval_is_refused_when_the_submission_changed_after_it_was_read():
    """The reviewer's window is minutes long, so an edit landing inside it must not be published."""
    await register_plugin(
        request=RegisterPluginRequest(name="submitted-skill", source=_GIT_SUBDIR_SOURCE),
        user_api_key_dict=_SUBMITTER,
    )
    reviewed_fingerprint = await _fingerprint("submitted-skill")

    await update_plugin(
        plugin_name="submitted-skill",
        request=UpdatePluginRequest(source={"source": "github", "repo": "org/swapped-in-after-review"}),
        user_api_key_dict=_SUBMITTER,
    )

    with pytest.raises(HTTPException) as exc_info:
        await _approve("submitted-skill", fingerprint=reviewed_fingerprint)

    assert exc_info.value.status_code == 409

    stored = await _stored("submitted-skill")
    assert stored.approval_status == "pending_review"
    assert stored.enabled is False

    body = json.loads((await get_marketplace()).body)
    assert body["plugins"] == []


@pytest.mark.asyncio
async def test_approval_of_the_reviewed_content_still_succeeds_after_an_unrelated_edit_is_reviewed():
    """Re-reading the changed submission is all it takes to approve it, so the guard is not a dead end."""
    await register_plugin(
        request=RegisterPluginRequest(name="submitted-skill", source=_GIT_SUBDIR_SOURCE),
        user_api_key_dict=_SUBMITTER,
    )
    await update_plugin(
        plugin_name="submitted-skill",
        request=UpdatePluginRequest(source={"source": "github", "repo": "org/second-attempt"}),
        user_api_key_dict=_SUBMITTER,
    )

    response = await _approve("submitted-skill")

    assert response.approval_status == "active"
    assert response.enabled is True

    body = json.loads((await get_marketplace()).body)
    assert [plugin["source"]["repo"] for plugin in body["plugins"]] == ["org/second-attempt"]


@pytest.mark.asyncio
async def test_approval_write_does_not_publish_an_edit_that_lands_after_the_fingerprint_check():
    """The read and the write are separate round trips, so the write itself has to be a compare-and-set."""
    await register_plugin(
        request=RegisterPluginRequest(name="submitted-skill", source=_GIT_SUBDIR_SOURCE),
        user_api_key_dict=_SUBMITTER,
    )
    reviewed_fingerprint = await _fingerprint("submitted-skill")

    table = litellm.proxy.proxy_server.prisma_client.db.litellm_claudecodeplugintable
    unpatched_find_unique = table.find_unique.side_effect

    async def _edit_after_the_review_read(where):
        record = await unpatched_find_unique(where)
        # A real read hands back a snapshot, so the fingerprint check sees the pre-edit content
        # and passes; only the write can still catch the edit.
        snapshot = SimpleNamespace(
            name=record.name,
            manifest_json=record.manifest_json,
            approval_status=record.approval_status,
        )
        table.find_unique = AsyncMock(side_effect=unpatched_find_unique)
        await update_plugin(
            plugin_name="submitted-skill",
            request=UpdatePluginRequest(source={"source": "github", "repo": "org/raced-in"}),
            user_api_key_dict=_SUBMITTER,
        )
        return snapshot

    table.find_unique = AsyncMock(side_effect=_edit_after_the_review_read)

    with pytest.raises(HTTPException) as exc_info:
        await approve_plugin(
            plugin_name="submitted-skill",
            request=ApprovePluginRequest(reviewed_fingerprint=reviewed_fingerprint),
            user_api_key_dict=_USER,
        )

    assert exc_info.value.status_code == 409

    stored = await _stored("submitted-skill")
    assert stored.enabled is False
    assert json.loads(stored.manifest_json)["source"]["repo"] == "org/raced-in"


@pytest.mark.asyncio
async def test_rejection_does_not_require_a_fingerprint():
    """Rejecting leaves the skill unpublished either way, so it is not bound to the reviewed content."""
    await register_plugin(
        request=RegisterPluginRequest(name="submitted-skill", source=_GIT_SUBDIR_SOURCE),
        user_api_key_dict=_SUBMITTER,
    )
    await update_plugin(
        plugin_name="submitted-skill",
        request=UpdatePluginRequest(source={"source": "github", "repo": "org/changed"}),
        user_api_key_dict=_SUBMITTER,
    )

    response = await reject_plugin(
        plugin_name="submitted-skill",
        request=RejectPluginRequest(review_notes="not this one"),
        user_api_key_dict=_USER,
    )

    assert response.approval_status == "rejected"
    assert response.enabled is False


_IDENTITY_LESS = UserAPIKeyAuth(user_role=LitellmUserRoles.INTERNAL_USER)


async def _refusal(coro) -> tuple[int, object]:
    with pytest.raises(HTTPException) as exc_info:
        await coro
    return exc_info.value.status_code, exc_info.value.detail


@pytest.mark.asyncio
async def test_update_of_a_hidden_skill_is_indistinguishable_from_an_absent_one():
    """A 403 here would tell an unrelated user that the name is taken by a pending submission."""
    await register_plugin(
        request=RegisterPluginRequest(name="submitted-skill", source=_GIT_SUBDIR_SOURCE),
        user_api_key_dict=_SUBMITTER,
    )
    edit = UpdatePluginRequest(source={"source": "github", "repo": "org/probe"})

    hidden = await _refusal(update_plugin(plugin_name="submitted-skill", request=edit, user_api_key_dict=_OTHER_USER))
    absent = await _refusal(update_plugin(plugin_name="no-such-skill", request=edit, user_api_key_dict=_OTHER_USER))

    assert absent == (404, {"error": "Plugin 'no-such-skill' not found"})
    assert hidden == (404, {"error": "Plugin 'submitted-skill' not found"})
    assert (await _stored("submitted-skill")).approval_status == "pending_review"


@pytest.mark.asyncio
async def test_delete_of_a_hidden_skill_is_indistinguishable_from_an_absent_one():
    await register_plugin(
        request=RegisterPluginRequest(name="submitted-skill", source=_GIT_SUBDIR_SOURCE),
        user_api_key_dict=_SUBMITTER,
    )

    hidden = await _refusal(delete_plugin(plugin_name="submitted-skill", user_api_key_dict=_OTHER_USER))
    absent = await _refusal(delete_plugin(plugin_name="no-such-skill", user_api_key_dict=_OTHER_USER))

    assert absent == (404, {"error": "Plugin 'no-such-skill' not found"})
    assert hidden == (404, {"error": "Plugin 'submitted-skill' not found"})
    assert await _stored("submitted-skill") is not None


@pytest.mark.asyncio
async def test_a_published_skill_still_refuses_a_non_owner_with_403():
    """Hiding is only for skills the caller cannot see; an active skill is public, so the refusal stays 403."""
    await register_plugin(
        request=RegisterPluginRequest(name="admin-skill", source=_GIT_SUBDIR_SOURCE),
        user_api_key_dict=_USER,
    )

    updated = await _refusal(
        update_plugin(
            plugin_name="admin-skill",
            request=UpdatePluginRequest(source={"source": "github", "repo": "org/probe"}),
            user_api_key_dict=_OTHER_USER,
        )
    )
    deleted = await _refusal(delete_plugin(plugin_name="admin-skill", user_api_key_dict=_OTHER_USER))

    assert updated[0] == 403
    assert deleted[0] == 403


@pytest.mark.asyncio
async def test_submission_without_an_attributable_identity_is_refused():
    """created_by would be null, leaving a pending row its own submitter can never list, read, or withdraw."""
    assert get_primary_resource_owner_scope(_IDENTITY_LESS) is None
    assert get_resource_owner_scopes(_IDENTITY_LESS) == []

    status_code, _ = await _refusal(
        register_plugin(
            request=RegisterPluginRequest(name="orphan-skill", source=_GIT_SUBDIR_SOURCE),
            user_api_key_dict=_IDENTITY_LESS,
        )
    )

    assert status_code == 403
    assert await _stored("orphan-skill") is None


@pytest.mark.asyncio
async def test_rejecting_an_approved_skill_is_refused_and_leaves_it_published():
    """Reject hides a skill from every non-owner, so on an approved skill it is a takedown, not a review step."""
    await register_plugin(
        request=RegisterPluginRequest(name="submitted-skill", source=_GIT_SUBDIR_SOURCE),
        user_api_key_dict=_SUBMITTER,
    )
    await _approve("submitted-skill")

    status_code, _ = await _refusal(
        reject_plugin(
            plugin_name="submitted-skill",
            request=RejectPluginRequest(review_notes="taking this down"),
            user_api_key_dict=_USER,
        )
    )

    assert status_code == 400

    stored = await _stored("submitted-skill")
    assert stored.approval_status == "active"
    assert stored.enabled is True
    assert stored.review_notes != "taking this down"

    body = json.loads((await get_marketplace()).body)
    assert [plugin["name"] for plugin in body["plugins"]] == ["submitted-skill"]
