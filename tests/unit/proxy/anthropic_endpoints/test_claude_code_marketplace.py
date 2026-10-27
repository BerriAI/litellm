"""
Unit tests for claude_code_marketplace.py source validation.

Covers the git-subdir and archive source types added alongside the existing github and url types.
"""

import io
import json
import struct
import tracemalloc
import zipfile
import zlib
from typing import Final
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from fastapi import HTTPException
from pydantic import ValidationError

import litellm
from litellm.proxy._types import LiteLLM_ObjectPermissionTable, ProxyException, UserAPIKeyAuth
from litellm.proxy.anthropic_endpoints.claude_code_endpoints import claude_code_marketplace
from litellm.proxy.anthropic_endpoints.claude_code_endpoints import claude_code_plugin_archive as plugin_archive
from litellm.proxy.anthropic_endpoints.claude_code_endpoints.claude_code_marketplace import (
    delete_plugin,
    disable_plugin,
    enable_plugin,
    get_marketplace,
    get_plugin,
    get_plugin_archive,
    list_plugins,
    register_plugin,
    update_plugin,
)
from litellm.proxy.proxy_server import LitellmUserRoles
from litellm.types.proxy.claude_code_endpoints import (
    MarketplaceResponse,
    RegisterPluginRequest,
    UpdatePluginRequest,
)
from tests._master_key import MASTER_KEY


def _make_mock_prisma():
    """Stateful prisma mock that supports find_unique, find_many, create, and update."""
    store: dict = {}

    mock_client = MagicMock()
    mock_client.proxy_logging_obj = MagicMock()
    mock_table = MagicMock()

    async def _find_unique(where):
        return store.get(where.get("name"))

    def _matches(record, where) -> bool:
        if "OR" in where:
            return any(_matches(record, clause) for clause in where["OR"])
        if "enabled" in where and record.enabled != where["enabled"]:
            return False
        return "name" not in where or record.name in where["name"]["in"]

    async def _find_many(where=None):
        return [r for r in store.values() if _matches(r, where or {})]

    async def _create(data):
        record = MagicMock()
        record.id = "test-id"
        record.name = data["name"]
        record.version = data.get("version")
        record.description = data.get("description")
        record.manifest_json = data.get("manifest_json", "{}")
        record.enabled = data.get("enabled", True)
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
    api_key=MASTER_KEY,
    user_id="test-user",
)

_NON_ADMIN_USER = UserAPIKeyAuth(
    user_role=LitellmUserRoles.INTERNAL_USER,
    api_key="sk-5678",
    user_id="regular-user",
)

_GIT_SUBDIR_SOURCE = {
    "source": "git-subdir",
    "url": "https://github.com/org/monorepo.git",
    "path": "plugins/my-plugin",
}

_ARCHIVE_SOURCE = {
    "source": "archive",
    "url": "https://skills-bucket.s3.us-east-1.amazonaws.com/plugins/s3-skill-1.0.0.zip",
    "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
}


def _zip_bytes(files: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return buffer.getvalue()


_REAL_BUILD_ARCHIVE = plugin_archive.build_archive


_PLUGIN_ZIP = plugin_archive.normalize_archive(
    _zip_bytes({".claude-plugin/plugin.json": b"{}", "skills/s/SKILL.md": b"# s"}), strip_top_level=False
)
_BUILT = plugin_archive.BuiltArchive(_PLUGIN_ZIP, plugin_archive.sha256_hex(_PLUGIN_ZIP), None)


@pytest.mark.asyncio
async def test_plugin_archive_get_enforces_download_limit_while_reading(monkeypatch):
    seen: dict[str, object] = {}

    async def _fake_async_safe_get(client, url: str, **kwargs):
        seen["client"] = client
        seen["url"] = url
        seen["kwargs"] = kwargs
        return httpx.Response(200, content=b"ok")

    monkeypatch.setattr(plugin_archive, "async_safe_get", _fake_async_safe_get)

    body = await plugin_archive._get("https://example.com/plugin.zip", {"Authorization": "Bearer token"})

    assert body == b"ok"
    assert seen["url"] == "https://example.com/plugin.zip"
    assert seen["kwargs"] == {
        "headers": {"Authorization": "Bearer token"},
        "max_response_bytes": plugin_archive.MAX_DOWNLOAD_BYTES,
    }


@pytest.fixture(autouse=True)
def _patch_proxy_globals(monkeypatch, tmp_path):
    """Scope prisma_client/master_key mutations to each test via monkeypatch."""
    monkeypatch.setattr(litellm.proxy.proxy_server, "prisma_client", _make_mock_prisma())
    monkeypatch.setattr(litellm.proxy.proxy_server, "master_key", MASTER_KEY)
    monkeypatch.setattr(
        plugin_archive,
        "build_archive",
        AsyncMock(side_effect=plugin_archive.ArchiveBuildError("offline in unit tests")),
    )
    monkeypatch.setenv("LITELLM_PLUGIN_ARCHIVE_DIR", str(tmp_path))


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
async def test_update_plugin_deleted_mid_update_returns_404():
    """A concurrent delete between the find_unique pre-check and the update makes prisma's
    update return None; that must surface the same 404 as a plain miss, not an AttributeError."""
    name = "my-monorepo-plugin"
    await register_plugin(
        request=RegisterPluginRequest(name=name, source=_GIT_SUBDIR_SOURCE, version="1.0.0"),
        user_api_key_dict=_USER,
    )

    table = litellm.proxy.proxy_server.prisma_client.db.litellm_claudecodeplugintable
    table.update = AsyncMock(return_value=None)

    with pytest.raises(HTTPException) as exc_info:
        await update_plugin(
            plugin_name=name,
            request=UpdatePluginRequest(source={"source": "github", "repo": "org/replacement"}),
            user_api_key_dict=_USER,
        )

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == {"error": f"Plugin '{name}' not found"}


@pytest.mark.asyncio
async def test_get_marketplace_skips_plugin_with_null_manifest():
    await register_plugin(
        request=RegisterPluginRequest(name="good-plugin", source=_GIT_SUBDIR_SOURCE, version="1.0.0"),
        user_api_key_dict=_USER,
    )

    table = litellm.proxy.proxy_server.prisma_client.db.litellm_claudecodeplugintable
    await table.create(data={"name": "null-manifest-plugin", "manifest_json": None, "enabled": True})

    response = await get_marketplace(request=MagicMock())

    assert response.status_code == 200
    body = json.loads(response.body)
    assert [plugin["name"] for plugin in body["plugins"]] == ["good-plugin"]


def _granted_user(skills: list[str]) -> UserAPIKeyAuth:
    return UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER,
        api_key="sk-granted",
        user_id="granted-user",
        object_permission=LiteLLM_ObjectPermissionTable(object_permission_id="perm-1", skills=skills),
    )


async def _register_public_and_private_plugins() -> None:
    for name, enabled in (("public-skill", True), ("private-skill", False)):
        await register_plugin(
            request=RegisterPluginRequest(name=name, source=_GIT_SUBDIR_SOURCE, version="1.0.0"),
            user_api_key_dict=_USER,
        )
        if not enabled:
            await disable_plugin(plugin_name=name, user_api_key_dict=_USER)


async def _listed_names(user: UserAPIKeyAuth) -> set[str]:
    response = await list_plugins(user_api_key_dict=user)
    return {plugin.name for plugin in response.plugins}


@pytest.mark.asyncio
async def test_list_plugins_shows_disabled_plugin_only_to_granted_key_or_admin():
    await _register_public_and_private_plugins()

    assert await _listed_names(_NON_ADMIN_USER) == {"public-skill"}
    assert await _listed_names(_granted_user(["other-skill"])) == {"public-skill"}
    assert await _listed_names(_granted_user(["private-skill"])) == {"public-skill", "private-skill"}
    assert await _listed_names(_USER) == {"public-skill", "private-skill"}


@pytest.mark.asyncio
async def test_get_plugin_returns_403_for_disabled_plugin_the_key_is_not_granted():
    await _register_public_and_private_plugins()

    with pytest.raises(HTTPException) as exc_info:
        await get_plugin(plugin_name="private-skill", user_api_key_dict=_NON_ADMIN_USER)
    assert exc_info.value.status_code == 403

    assert (await get_plugin(plugin_name="public-skill", user_api_key_dict=_NON_ADMIN_USER))["name"] == "public-skill"
    granted = await get_plugin(plugin_name="private-skill", user_api_key_dict=_granted_user(["private-skill"]))
    assert granted["name"] == "private-skill"
    assert granted["enabled"] is False


async def _marketplace_names(key: str | None) -> list[str]:
    response = await get_marketplace(request=MagicMock(), key=key)
    assert response.status_code == 200
    return sorted(plugin["name"] for plugin in json.loads(response.body)["plugins"])


@pytest.mark.asyncio
async def test_get_marketplace_key_query_param_adds_granted_disabled_plugins(monkeypatch):
    await _register_public_and_private_plugins()
    keys = {"sk-granted": _granted_user(["private-skill"]), "sk-plain": _NON_ADMIN_USER}

    async def _fake_auth(request, api_key: str) -> UserAPIKeyAuth:
        token = api_key.removeprefix("Bearer ")
        if token not in keys:
            raise ProxyException(message="invalid key", type="auth_error", param="key", code=401)
        return keys[token]

    monkeypatch.setattr(claude_code_marketplace, "user_api_key_auth", _fake_auth)

    assert await _marketplace_names(None) == ["public-skill"]
    assert await _marketplace_names("sk-plain") == ["public-skill"]
    assert await _marketplace_names("sk-granted") == ["private-skill", "public-skill"]

    with pytest.raises(ProxyException) as exc_info:
        await get_marketplace(request=MagicMock(), key="sk-bogus")
    assert exc_info.value.code == "401"


@pytest.mark.asyncio
@pytest.mark.parametrize("preference", ["available", "auto_install", "required"])
async def test_get_marketplace_emits_installation_preference(preference):
    await register_plugin(
        request=RegisterPluginRequest(
            name="s3-skill",
            source=_ARCHIVE_SOURCE,
            serve_archive=False,
            installation_preference=preference,
        ),
        user_api_key_dict=_USER,
    )

    marketplace = json.loads((await get_marketplace(request=MagicMock())).body)

    assert marketplace["plugins"] == [
        {"name": "s3-skill", "source": _ARCHIVE_SOURCE, "version": "1.0.0", "installationPreference": preference}
    ]


@pytest.mark.parametrize(
    "build_request",
    [
        lambda preference: RegisterPluginRequest(
            name="s3-skill", source=_ARCHIVE_SOURCE, serve_archive=False, installation_preference=preference
        ),
        lambda preference: UpdatePluginRequest(
            source=_ARCHIVE_SOURCE, serve_archive=False, installation_preference=preference
        ),
    ],
    ids=["register", "update"],
)
def test_plugin_request_rejects_unknown_installation_preference(build_request):
    with pytest.raises(ValidationError, match="installation_preference"):
        build_request("auto-install")


@pytest.mark.asyncio
async def test_get_marketplace_omits_installation_preference_when_unset():
    await register_plugin(
        request=RegisterPluginRequest(name="manual-install-plugin", source=_GIT_SUBDIR_SOURCE),
        user_api_key_dict=_USER,
    )

    marketplace = json.loads((await get_marketplace(request=MagicMock())).body)

    assert "installationPreference" not in marketplace["plugins"][0]


@pytest.mark.asyncio
async def test_update_plugin_propagates_installation_preference_to_get_and_list():
    name = "updated-install-plugin"
    await register_plugin(
        request=RegisterPluginRequest(name=name, source=_ARCHIVE_SOURCE, serve_archive=False),
        user_api_key_dict=_USER,
    )

    response = await update_plugin(
        plugin_name=name,
        request=UpdatePluginRequest(
            source=_ARCHIVE_SOURCE,
            serve_archive=False,
            installation_preference="auto_install",
        ),
        user_api_key_dict=_USER,
    )

    plugin = await get_plugin(plugin_name=name, user_api_key_dict=_USER)
    listed_plugin = (await list_plugins(user_api_key_dict=_USER)).plugins[0]

    assert response.plugin.installation_preference == "auto_install"
    assert plugin["installation_preference"] == "auto_install"
    assert listed_plugin.installation_preference == "auto_install"


@pytest.mark.asyncio
async def test_update_plugin_without_installation_preference_clears_it():
    name = "cleared-install-plugin"
    await register_plugin(
        request=RegisterPluginRequest(
            name=name, source=_ARCHIVE_SOURCE, serve_archive=False, installation_preference="auto_install"
        ),
        user_api_key_dict=_USER,
    )

    await update_plugin(
        plugin_name=name,
        request=UpdatePluginRequest(source=_ARCHIVE_SOURCE, serve_archive=False),
        user_api_key_dict=_USER,
    )

    marketplace = json.loads((await get_marketplace(request=MagicMock())).body)
    assert "installationPreference" not in marketplace["plugins"][0]


@pytest.mark.asyncio
@pytest.mark.parametrize("stored_value", ["auto-install", 1, True, None])
async def test_stored_unknown_installation_preference_is_not_served(stored_value, caplog):
    name = "hand-edited-plugin"
    table = litellm.proxy.proxy_server.prisma_client.db.litellm_claudecodeplugintable
    await table.create(
        data={
            "name": name,
            "manifest_json": json.dumps(
                {"name": name, "source": _ARCHIVE_SOURCE, "installation_preference": stored_value}
            ),
        }
    )

    marketplace = json.loads((await get_marketplace(request=MagicMock())).body)
    listed_plugin = (await list_plugins(user_api_key_dict=_USER)).plugins[0]
    plugin = await get_plugin(plugin_name=name, user_api_key_dict=_USER)

    assert marketplace["plugins"] == [{"name": name, "source": _ARCHIVE_SOURCE}]
    assert listed_plugin.installation_preference is None
    assert plugin["installation_preference"] is None
    unknown_value_warnings = [r for r in caplog.records if "unknown installation_preference" in r.getMessage()]
    assert len(unknown_value_warnings) == (0 if stored_value is None else 3)


@pytest.mark.asyncio
async def test_register_accepts_the_camel_case_installation_preference_key():
    request = RegisterPluginRequest.model_validate(
        {
            "name": "s3-skill",
            "source": _ARCHIVE_SOURCE,
            "serve_archive": False,
            "installationPreference": "auto_install",
        }
    )

    response = await register_plugin(request=request, user_api_key_dict=_USER)
    marketplace = json.loads((await get_marketplace(request=MagicMock())).body)

    assert response.plugin.installation_preference == "auto_install"
    assert (await _read_stored_manifest("s3-skill"))["installation_preference"] == "auto_install"
    assert marketplace["plugins"][0]["installationPreference"] == "auto_install"


_NON_ARCHIVE_SOURCES = [
    {"source": "github", "repo": "org/repo"},
    {"source": "url", "url": "https://github.com/org/repo.git"},
    _GIT_SUBDIR_SOURCE,
]
_UNPINNED_ARCHIVE_SOURCE = {"source": "archive", "url": "https://artifacts.example.com/plugin.zip"}


async def _assert_preference_refused(source, preference):
    with pytest.raises(HTTPException) as register_error:
        await register_plugin(
            request=RegisterPluginRequest(
                name="no-op-plugin", source=source, serve_archive=False, installation_preference=preference
            ),
            user_api_key_dict=_USER,
        )
    await register_plugin(
        request=RegisterPluginRequest(name="no-op-plugin", source=source, serve_archive=False), user_api_key_dict=_USER
    )
    with pytest.raises(HTTPException) as update_error:
        await update_plugin(
            plugin_name="no-op-plugin",
            request=UpdatePluginRequest(source=source, serve_archive=False, installation_preference=preference),
            user_api_key_dict=_USER,
        )

    assert register_error.value.status_code == 400
    assert update_error.value.status_code == 400
    assert "installationPreference" not in json.loads((await get_marketplace(request=MagicMock())).body)["plugins"][0]


@pytest.mark.asyncio
@pytest.mark.parametrize("preference", ["available", "auto_install", "required"])
@pytest.mark.parametrize("source", _NON_ARCHIVE_SOURCES)
async def test_every_preference_needs_an_archive_source(preference, source):
    await _assert_preference_refused(source, preference)


@pytest.mark.asyncio
@pytest.mark.parametrize("preference", ["auto_install", "required"])
async def test_self_installing_preference_needs_the_archive_sha256(preference):
    await _assert_preference_refused(_UNPINNED_ARCHIVE_SOURCE, preference)


@pytest.mark.asyncio
async def test_available_preference_is_accepted_on_an_unpinned_archive():
    response = await register_plugin(
        request=RegisterPluginRequest(
            name="listed-plugin",
            source=_UNPINNED_ARCHIVE_SOURCE,
            serve_archive=False,
            installation_preference="available",
        ),
        user_api_key_dict=_USER,
    )
    marketplace = json.loads((await get_marketplace(request=MagicMock())).body)

    assert response.plugin.installation_preference == "available"
    assert marketplace["plugins"][0]["installationPreference"] == "available"


@pytest.mark.parametrize(
    "build_request",
    [
        lambda body: RegisterPluginRequest.model_validate({"name": "s3-skill", **body}),
        UpdatePluginRequest.model_validate,
    ],
    ids=["register", "update"],
)
def test_plugin_request_rejects_both_preference_spellings(build_request):
    with pytest.raises(ValidationError, match="not both"):
        build_request(
            {
                "source": _ARCHIVE_SOURCE,
                "serve_archive": False,
                "installation_preference": "auto_install",
                "installationPreference": "available",
            }
        )


@pytest.mark.parametrize(
    "build_request",
    [
        lambda body: RegisterPluginRequest.model_validate({"name": "s3-skill", **body}),
        UpdatePluginRequest.model_validate,
    ],
    ids=["register", "update"],
)
@pytest.mark.parametrize(
    "null_key, value_key",
    [("installation_preference", "installationPreference"), ("installationPreference", "installation_preference")],
)
def test_plugin_request_reads_the_spelling_that_carries_a_value(build_request, null_key, value_key):
    request = build_request(
        {"source": _ARCHIVE_SOURCE, "serve_archive": False, null_key: None, value_key: "auto_install"}
    )

    assert request.installation_preference == "auto_install"


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
    assert "archive" in exc_info.value.detail["error"]


@pytest.mark.asyncio
async def test_archive_source_registers_and_is_served_verbatim_in_marketplace():
    response = await register_plugin(
        request=RegisterPluginRequest(name="s3-skill", source=_ARCHIVE_SOURCE, serve_archive=False),
        user_api_key_dict=_USER,
    )

    assert response.action == "created"
    assert response.plugin.source == _ARCHIVE_SOURCE

    marketplace = json.loads((await get_marketplace(request=MagicMock())).body)
    assert marketplace["plugins"] == [{"name": "s3-skill", "source": _ARCHIVE_SOURCE, "version": "1.0.0"}]


@pytest.mark.asyncio
async def test_archive_source_without_sha256_is_accepted():
    source = {"source": "archive", "url": "https://artifacts.example.com/plugin.zip"}

    response = await register_plugin(
        request=RegisterPluginRequest(name="unpinned-skill", source=source, serve_archive=False),
        user_api_key_dict=_USER,
    )

    assert response.plugin.source == source


@pytest.mark.asyncio
async def test_update_plugin_to_archive_source():
    name = "my-monorepo-plugin"
    await register_plugin(
        request=RegisterPluginRequest(name=name, source=_GIT_SUBDIR_SOURCE),
        user_api_key_dict=_USER,
    )

    response = await update_plugin(
        plugin_name=name,
        request=UpdatePluginRequest(source=_ARCHIVE_SOURCE, serve_archive=False),
        user_api_key_dict=_USER,
    )

    assert response.action == "updated"
    assert (await _read_stored_manifest(name))["source"] == _ARCHIVE_SOURCE


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "source, expected_fragment",
    [
        ({"source": "archive"}, "url"),
        ({"source": "archive", "url": ""}, "url"),
        ({"source": "archive", "url": "http://artifacts.example.com/plugin.zip"}, "https"),
        ({"source": "archive", "url": "s3://skills-bucket/plugin.zip"}, "https"),
        ({"source": "archive", "url": "https://"}, "https"),
        ({"source": "archive", "url": "https:///plugin.zip"}, "https"),
        ({"source": "archive", "url": "https://[::1/plugin.zip"}, "https"),
        ({"source": "archive", "url": "https://artifacts.example.com/plugin.zip", "sha256": "a" * 63}, "sha256"),
        ({"source": "archive", "url": "https://artifacts.example.com/plugin.zip", "sha256": "a" * 65}, "sha256"),
        ({"source": "archive", "url": "https://artifacts.example.com/plugin.zip", "sha256": "g" * 64}, "sha256"),
    ],
)
async def test_register_plugin_archive_rejects_malformed_source(source, expected_fragment):
    with pytest.raises(HTTPException) as exc_info:
        await register_plugin(request=RegisterPluginRequest(name="bad-plugin", source=source), user_api_key_dict=_USER)

    assert exc_info.value.status_code == 400
    assert expected_fragment in exc_info.value.detail["error"]


@pytest.mark.asyncio
async def test_register_plugin_rejects_non_admin():
    """A non-admin key cannot add an entry to the marketplace catalog."""
    request = RegisterPluginRequest(name="attacker-plugin", source=_GIT_SUBDIR_SOURCE)

    with pytest.raises(HTTPException) as exc_info:
        await register_plugin(request=request, user_api_key_dict=_NON_ADMIN_USER)

    assert exc_info.value.status_code == 403

    table = litellm.proxy.proxy_server.prisma_client.db.litellm_claudecodeplugintable
    assert await table.find_unique(where={"name": "attacker-plugin"}) is None


@pytest.mark.asyncio
async def test_update_plugin_rejects_non_admin_overwrite():
    """A non-admin key cannot overwrite an existing plugin's source."""
    name = "trusted-plugin"
    await register_plugin(
        request=RegisterPluginRequest(name=name, source=_GIT_SUBDIR_SOURCE, version="1.0.0"),
        user_api_key_dict=_USER,
    )

    malicious_source = {"source": "github", "repo": "attacker/malicious-repo"}
    with pytest.raises(HTTPException) as exc_info:
        await update_plugin(
            plugin_name=name,
            request=UpdatePluginRequest(source=malicious_source),
            user_api_key_dict=_NON_ADMIN_USER,
        )

    assert exc_info.value.status_code == 403

    stored = await _read_stored_manifest(name)
    assert stored["source"] == _GIT_SUBDIR_SOURCE


@pytest.mark.asyncio
async def test_enable_disable_delete_plugin_reject_non_admin():
    """Non-admin keys cannot enable, disable, or delete catalog entries."""
    name = "trusted-plugin-2"
    await register_plugin(
        request=RegisterPluginRequest(name=name, source=_GIT_SUBDIR_SOURCE, version="1.0.0"),
        user_api_key_dict=_USER,
    )

    for coro in (
        enable_plugin(plugin_name=name, user_api_key_dict=_NON_ADMIN_USER),
        disable_plugin(plugin_name=name, user_api_key_dict=_NON_ADMIN_USER),
        delete_plugin(plugin_name=name, user_api_key_dict=_NON_ADMIN_USER),
    ):
        with pytest.raises(HTTPException) as exc_info:
            await coro
        assert exc_info.value.status_code == 403

    table = litellm.proxy.proxy_server.prisma_client.db.litellm_claudecodeplugintable
    assert (await table.find_unique(where={"name": name})).enabled is True


@pytest.mark.asyncio
async def test_register_plugin():
    plugin_name: Final = "test-plugin"

    request: Final = RegisterPluginRequest(
        name=plugin_name,
        source={"source": "github", "repo": "test-org/test-repo"},
        version="1.0.0",
        description="Test plugin for unit tests",
    )

    user_api_key_dict: Final = UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN,
        api_key=MASTER_KEY,
        user_id="test-user",
    )

    response: Final = await register_plugin(
        request=request,
        user_api_key_dict=user_api_key_dict,
    )

    assert response.status == "success"
    assert response.action == "created"
    assert response.plugin.name == plugin_name
    assert response.plugin.version == "1.0.0"
    assert response.plugin.enabled is True

    stored_plugin: Final = await litellm.proxy.proxy_server.prisma_client.db.litellm_claudecodeplugintable.find_unique(
        where={"name": plugin_name}
    )
    assert stored_plugin is not None
    assert stored_plugin.name == plugin_name


@pytest.mark.asyncio
async def test_get_marketplace():
    plugin_name: Final = "test-marketplace-plugin"

    request: Final = RegisterPluginRequest(
        name=plugin_name,
        source={"source": "github", "repo": "test-org/marketplace-test"},
        version="2.0.0",
        description="Test plugin for marketplace test",
    )

    user_api_key_dict: Final = UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN,
        api_key=MASTER_KEY,
        user_id="test-user",
    )

    await register_plugin(
        request=request,
        user_api_key_dict=user_api_key_dict,
    )

    response: Final = await get_marketplace(request=MagicMock())

    body: Final = json.loads(response.body.decode())

    assert body["name"] == "litellm"
    assert "plugins" in body

    our_plugin: Final = next((p for p in body["plugins"] if p["name"] == plugin_name), None)
    assert our_plugin is not None
    assert our_plugin["source"] == {
        "source": "github",
        "repo": "test-org/marketplace-test",
    }
    assert our_plugin["version"] == "2.0.0"


@pytest.mark.asyncio
async def test_register_plugin_git_subdir():
    plugin_name: Final = "test-subdir-plugin"

    request: Final = RegisterPluginRequest(
        name=plugin_name,
        source={
            "source": "git-subdir",
            "url": "https://github.com/test-org/monorepo.git",
            "path": "plugins/my-plugin",
        },
        version="1.0.0",
        description="Test git-subdir plugin",
    )

    user_api_key_dict: Final = UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN,
        api_key=MASTER_KEY,
        user_id="test-user",
    )

    response: Final = await register_plugin(
        request=request,
        user_api_key_dict=user_api_key_dict,
    )

    assert response.status == "success"
    assert response.action == "created"
    assert response.plugin.name == plugin_name
    assert response.plugin.source["source"] == "git-subdir"
    assert response.plugin.source["url"] == "https://github.com/test-org/monorepo.git"
    assert response.plugin.source["path"] == "plugins/my-plugin"
    assert response.plugin.enabled is True


def test_marketplace_response_round_trips_installation_preference():
    catalog: Final = {
        "name": "litellm",
        "owner": {"name": "LiteLLM", "email": "support@litellm.ai"},
        "plugins": [
            {
                "name": "auto",
                "source": {"source": "url", "url": "https://example.com/auto.tgz"},
                "installationPreference": "auto_install",
            },
            {"name": "manual", "source": {"source": "url", "url": "https://example.com/manual.tgz"}},
        ],
    }

    round_tripped: Final = MarketplaceResponse.model_validate(catalog).model_dump(exclude_none=True)

    assert round_tripped["plugins"][0]["installationPreference"] == "auto_install"
    assert "installationPreference" not in round_tripped["plugins"][1]


def test_marketplace_response_rejects_unknown_installation_preference():
    catalog: Final = {
        "name": "litellm",
        "owner": {"name": "LiteLLM", "email": "support@litellm.ai"},
        "plugins": [
            {
                "name": "auto",
                "source": {"source": "url", "url": "https://example.com/auto.tgz"},
                "installationPreference": "sometimes",
            }
        ],
    }

    with pytest.raises(ValidationError):
        MarketplaceResponse.model_validate(catalog)


@pytest.fixture
def archive_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("LITELLM_PLUGIN_ARCHIVE_DIR", str(tmp_path))
    return tmp_path


def _marketplace_request(base_url: str = "https://litellm.example.com/") -> MagicMock:
    request = MagicMock()
    request.base_url = base_url
    return request


def _patch_build(monkeypatch, built=_BUILT):
    build = AsyncMock(return_value=built)
    monkeypatch.setattr(plugin_archive, "build_archive", build)
    return build


@pytest.mark.asyncio
async def test_archive_source_is_hosted_by_default(archive_dir, monkeypatch):
    build = _patch_build(monkeypatch)

    response = await register_plugin(
        request=RegisterPluginRequest(name="s3-skill", source=_ARCHIVE_SOURCE), user_api_key_dict=_USER
    )

    build.assert_awaited_once()
    assert response.plugin.serve_archive is True
    manifest = await _read_stored_manifest("s3-skill")
    assert manifest["archive_sha256"] == _BUILT.sha256
    assert (archive_dir / "s3-skill.zip").read_bytes() == _PLUGIN_ZIP


@pytest.mark.asyncio
async def test_git_source_is_not_hosted_by_default(archive_dir, monkeypatch):
    build = _patch_build(monkeypatch)

    response = await register_plugin(
        request=RegisterPluginRequest(name="git-skill", source=_GIT_SUBDIR_SOURCE), user_api_key_dict=_USER
    )

    build.assert_not_awaited()
    assert response.plugin.serve_archive is False
    assert not list(archive_dir.iterdir())


@pytest.mark.asyncio
async def test_git_source_can_opt_in_to_hosting(archive_dir, monkeypatch):
    build = _patch_build(monkeypatch, plugin_archive.BuiltArchive(_PLUGIN_ZIP, _BUILT.sha256, "a" * 40))

    await register_plugin(
        request=RegisterPluginRequest(name="git-skill", source=_GIT_SUBDIR_SOURCE, serve_archive=True),
        user_api_key_dict=_USER,
    )

    build.assert_awaited_once_with(_GIT_SUBDIR_SOURCE)
    assert (await _read_stored_manifest("git-skill"))["archive_ref"] == "a" * 40


@pytest.mark.asyncio
async def test_archive_source_can_opt_out_of_hosting(archive_dir, monkeypatch):
    build = _patch_build(monkeypatch)

    await register_plugin(
        request=RegisterPluginRequest(name="s3-skill", source=_ARCHIVE_SOURCE, serve_archive=False),
        user_api_key_dict=_USER,
    )

    build.assert_not_awaited()
    assert (await _read_stored_manifest("s3-skill")).get("serve_archive") is None


@pytest.mark.asyncio
async def test_unbuildable_source_is_rejected_without_registering(archive_dir, monkeypatch):
    monkeypatch.setattr(
        plugin_archive, "build_archive", AsyncMock(side_effect=plugin_archive.ArchiveBuildError("no plugin.json"))
    )

    with pytest.raises(HTTPException) as exc:
        await register_plugin(
            request=RegisterPluginRequest(name="s3-skill", source=_ARCHIVE_SOURCE, serve_archive=True),
            user_api_key_dict=_USER,
        )

    assert exc.value.status_code == 400
    assert "no plugin.json" in exc.value.detail["error"]
    table = litellm.proxy.proxy_server.prisma_client.db.litellm_claudecodeplugintable
    assert await table.find_unique(where={"name": "s3-skill"}) is None


@pytest.mark.asyncio
async def test_marketplace_advertises_same_origin_archive(archive_dir, monkeypatch):
    _patch_build(monkeypatch)
    await register_plugin(
        request=RegisterPluginRequest(name="s3-skill", source=_ARCHIVE_SOURCE), user_api_key_dict=_USER
    )
    await register_plugin(
        request=RegisterPluginRequest(name="git-skill", source=_GIT_SUBDIR_SOURCE), user_api_key_dict=_USER
    )

    response = await get_marketplace(request=_marketplace_request())

    entries = {p["name"]: p["source"] for p in json.loads(response.body)["plugins"]}
    assert entries["s3-skill"] == {
        "source": "archive",
        "url": "https://litellm.example.com/claude-code/plugins/s3-skill/archive.zip",
        "sha256": _BUILT.sha256,
    }
    assert entries["git-skill"] == _GIT_SUBDIR_SOURCE


@pytest.mark.asyncio
async def test_archive_download_returns_stored_zip(archive_dir, monkeypatch):
    build = _patch_build(monkeypatch)
    await register_plugin(
        request=RegisterPluginRequest(name="s3-skill", source=_ARCHIVE_SOURCE), user_api_key_dict=_USER
    )
    build.reset_mock()

    response = await get_plugin_archive("s3-skill", request=_marketplace_request())

    assert response.media_type == "application/zip"
    assert response.body == _PLUGIN_ZIP
    build.assert_not_awaited()


@pytest.mark.asyncio
async def test_archive_download_refetches_when_file_is_missing(archive_dir, monkeypatch):
    build = _patch_build(monkeypatch)
    await register_plugin(
        request=RegisterPluginRequest(name="s3-skill", source=_ARCHIVE_SOURCE), user_api_key_dict=_USER
    )
    (archive_dir / "s3-skill.zip").unlink()
    build.reset_mock()

    response = await get_plugin_archive("s3-skill", request=_marketplace_request())

    build.assert_awaited_once()
    assert response.body == _PLUGIN_ZIP
    assert (archive_dir / "s3-skill.zip").read_bytes() == _PLUGIN_ZIP


@pytest.mark.asyncio
async def test_archive_refetch_reuses_pinned_commit(archive_dir, monkeypatch):
    build = _patch_build(monkeypatch, plugin_archive.BuiltArchive(_PLUGIN_ZIP, _BUILT.sha256, "b" * 40))
    await register_plugin(
        request=RegisterPluginRequest(name="git-skill", source=_GIT_SUBDIR_SOURCE, serve_archive=True),
        user_api_key_dict=_USER,
    )
    (archive_dir / "git-skill.zip").unlink()
    build.reset_mock()

    await get_plugin_archive("git-skill", request=_marketplace_request())

    build.assert_awaited_once_with(_GIT_SUBDIR_SOURCE, pinned_ref="b" * 40)


@pytest.mark.asyncio
async def test_archive_refetch_with_drifted_content_is_not_served(archive_dir, monkeypatch):
    build = _patch_build(monkeypatch)
    await register_plugin(
        request=RegisterPluginRequest(name="s3-skill", source=_ARCHIVE_SOURCE), user_api_key_dict=_USER
    )
    (archive_dir / "s3-skill.zip").unlink()
    build.return_value = plugin_archive.BuiltArchive(b"other", plugin_archive.sha256_hex(b"other"), None)

    with pytest.raises(HTTPException) as exc:
        await get_plugin_archive("s3-skill", request=_marketplace_request())

    assert exc.value.status_code == 502
    assert not (archive_dir / "s3-skill.zip").exists()


@pytest.mark.asyncio
async def test_archive_download_of_unhosted_plugin_is_404(archive_dir):
    await register_plugin(
        request=RegisterPluginRequest(name="git-skill", source=_GIT_SUBDIR_SOURCE), user_api_key_dict=_USER
    )

    with pytest.raises(HTTPException) as exc:
        await get_plugin_archive("git-skill", request=_marketplace_request())

    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_archive_download_hides_disabled_plugin_from_anonymous_callers(archive_dir, monkeypatch):
    _patch_build(monkeypatch)
    await register_plugin(
        request=RegisterPluginRequest(name="s3-skill", source=_ARCHIVE_SOURCE), user_api_key_dict=_USER
    )
    await disable_plugin("s3-skill", user_api_key_dict=_USER)

    with pytest.raises(HTTPException) as exc:
        await get_plugin_archive("s3-skill", request=_marketplace_request())

    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_delete_plugin_removes_stored_archive(archive_dir, monkeypatch):
    _patch_build(monkeypatch)
    await register_plugin(
        request=RegisterPluginRequest(name="s3-skill", source=_ARCHIVE_SOURCE), user_api_key_dict=_USER
    )
    prisma = litellm.proxy.proxy_server.prisma_client
    prisma.db.litellm_claudecodeplugintable.delete = AsyncMock()

    await delete_plugin("s3-skill", user_api_key_dict=_USER)

    assert not (archive_dir / "s3-skill.zip").exists()


def test_normalize_strips_github_wrapper_and_selects_subdir():
    raw = _zip_bytes(
        {
            "repo-abc123/README.md": b"root",
            "repo-abc123/plugins/p/.claude-plugin/plugin.json": b"{}",
            "repo-abc123/plugins/p/skills/a/SKILL.md": b"# a",
        }
    )

    data = plugin_archive.normalize_archive(raw, strip_top_level=True, subdir="plugins/p")

    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        assert archive.namelist() == [".claude-plugin/plugin.json", "skills/a/SKILL.md"]


def test_normalize_is_deterministic():
    raw = _zip_bytes({"w/.claude-plugin/plugin.json": b"{}", "w/a.txt": b"a"})

    assert plugin_archive.normalize_archive(raw, strip_top_level=True) == plugin_archive.normalize_archive(
        raw, strip_top_level=True
    )


def test_normalize_unwraps_a_wrapped_archive_source():
    raw = _zip_bytes({"my-plugin/.claude-plugin/plugin.json": b"{}"})

    data = plugin_archive.normalize_archive(raw, strip_top_level=False)

    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        assert archive.namelist() == [".claude-plugin/plugin.json"]


def test_normalize_requires_plugin_json():
    with pytest.raises(plugin_archive.ArchiveBuildError, match=r"plugin\.json"):
        plugin_archive.normalize_archive(_zip_bytes({"README.md": b"x"}), strip_top_level=False)


@pytest.mark.parametrize("member", ["../evil.txt", "/etc/passwd", "a/../../evil.txt"])
def test_normalize_rejects_path_traversal(member):
    raw = _zip_bytes({".claude-plugin/plugin.json": b"{}", member: b"x"})

    with pytest.raises(plugin_archive.ArchiveBuildError, match="unsafe path"):
        plugin_archive.normalize_archive(raw, strip_top_level=False)


def test_normalize_rejects_symlinks():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(".claude-plugin/plugin.json", b"{}")
        link = zipfile.ZipInfo("link")
        link.external_attr = 0o120777 << 16
        archive.writestr(link, "/etc/passwd")

    with pytest.raises(plugin_archive.ArchiveBuildError, match="symlink"):
        plugin_archive.normalize_archive(buffer.getvalue(), strip_top_level=False)


def test_normalize_rejects_oversized_plugin(monkeypatch):
    monkeypatch.setattr(plugin_archive, "MAX_ARCHIVE_BYTES", 10)

    with pytest.raises(plugin_archive.ArchiveBuildError, match="larger than"):
        plugin_archive.normalize_archive(_zip_bytes({".claude-plugin/plugin.json": b"x" * 100}), strip_top_level=False)


def _zip_with_understated_size(payload_bytes: int, declared_size: int) -> bytes:
    payload = b"\0" * payload_bytes
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(".claude-plugin/plugin.json", b"{}")
        archive.writestr("bomb.bin", payload)
    raw = bytearray(buffer.getvalue())
    crc = struct.pack("<I", zlib.crc32(payload[:declared_size]))
    for signature, name_index, name_offset, crc_offset, size_offset in (
        (b"PK\x03\x04", raw.index(b"bomb.bin"), 30, 14, 22),
        (b"PK\x01\x02", raw.rindex(b"bomb.bin"), 46, 16, 24),
    ):
        start = name_index - name_offset
        assert raw[start : start + 4] == signature
        raw[start + crc_offset : start + crc_offset + 4] = crc
        raw[start + size_offset : start + size_offset + 4] = struct.pack("<I", declared_size)
    return bytes(raw)


def test_normalize_does_not_expand_a_member_past_its_declared_size():
    forged = _zip_with_understated_size(payload_bytes=50 * 1024 * 1024, declared_size=2)
    tracemalloc.start()
    try:
        result = plugin_archive.normalize_archive(forged, strip_top_level=False)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert peak < 10 * 1024 * 1024
    with zipfile.ZipFile(io.BytesIO(result)) as archive:
        assert archive.read("bomb.bin") == b"\0\0"


def test_normalize_rejects_a_member_whose_contents_do_not_match_its_checksum():
    forged = _zip_with_understated_size(payload_bytes=4096, declared_size=2)
    corrupted = forged.replace(struct.pack("<I", zlib.crc32(b"\0\0")), b"\x00\x00\x00\x00")

    with pytest.raises(plugin_archive.ArchiveBuildError, match="corrupt"):
        plugin_archive.normalize_archive(corrupted, strip_top_level=False)


def test_normalize_rejects_unsupported_compression_methods():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_BZIP2) as archive:
        archive.writestr(".claude-plugin/plugin.json", b"{}")

    with pytest.raises(plugin_archive.ArchiveBuildError, match="compression"):
        plugin_archive.normalize_archive(buffer.getvalue(), strip_top_level=False)


@pytest.mark.asyncio
async def test_build_archive_verifies_declared_sha256(monkeypatch):
    monkeypatch.setattr(plugin_archive, "build_archive", _REAL_BUILD_ARCHIVE)
    monkeypatch.setattr(
        plugin_archive, "_get", AsyncMock(return_value=_zip_bytes({".claude-plugin/plugin.json": b"{}"}))
    )

    with pytest.raises(plugin_archive.ArchiveBuildError, match="sha256"):
        await _REAL_BUILD_ARCHIVE(_ARCHIVE_SOURCE)


@pytest.mark.asyncio
async def test_build_archive_rejects_non_github_git_hosts(monkeypatch):
    monkeypatch.setattr(plugin_archive, "build_archive", _REAL_BUILD_ARCHIVE)
    with pytest.raises(plugin_archive.ArchiveBuildError, match="GitHub"):
        await _REAL_BUILD_ARCHIVE(
            {"source": "git-subdir", "url": "https://gitlab.com/org/repo.git", "path": "plugins/p"}
        )


@pytest.mark.asyncio
async def test_build_archive_from_github_pins_the_commit(monkeypatch):
    monkeypatch.setattr(plugin_archive, "build_archive", _REAL_BUILD_ARCHIVE)
    commit = "c" * 40
    raw = _zip_bytes({"repo-c/plugins/my-plugin/.claude-plugin/plugin.json": b"{}"})

    async def fake_get(url, headers=None):
        return commit.encode() if "api.github.com" in url else raw

    get = AsyncMock(side_effect=fake_get)
    monkeypatch.setattr(plugin_archive, "_get", get)

    built = await _REAL_BUILD_ARCHIVE(_GIT_SUBDIR_SOURCE.copy() | {"url": "https://github.com/org/repo.git"})

    assert built.ref == commit
    assert get.await_args_list[-1].args[0] == f"https://codeload.github.com/org/repo/zip/{commit}"


@pytest.mark.asyncio
async def test_default_hosting_surfaces_the_build_error_and_does_not_create_the_plugin(archive_dir):
    """A hosted archive that cannot be built is an error for the admin, not a log line."""
    with pytest.raises(HTTPException) as exc:
        await register_plugin(
            request=RegisterPluginRequest(name="s3-skill", source=_ARCHIVE_SOURCE), user_api_key_dict=_USER
        )

    assert exc.value.status_code == 400
    assert exc.value.detail["error"] == "Cannot host plugin 's3-skill' as an archive: offline in unit tests"
    table = litellm.proxy.proxy_server.prisma_client.db.litellm_claudecodeplugintable
    assert await table.find_unique(where={"name": "s3-skill"}) is None


@pytest.mark.asyncio
async def test_update_with_an_unbuildable_archive_keeps_the_existing_plugin(archive_dir):
    await register_plugin(
        request=RegisterPluginRequest(name="s3-skill", source=_ARCHIVE_SOURCE, serve_archive=False),
        user_api_key_dict=_USER,
    )

    with pytest.raises(HTTPException) as exc:
        await update_plugin(
            plugin_name="s3-skill",
            request=UpdatePluginRequest(source=_ARCHIVE_SOURCE),
            user_api_key_dict=_USER,
        )

    assert exc.value.status_code == 400
    assert (await _read_stored_manifest("s3-skill")).get("serve_archive") is None


def test_normalize_limits_size_to_the_selected_subdirectory(monkeypatch):
    monkeypatch.setattr(plugin_archive, "MAX_ARCHIVE_BYTES", 400)
    raw = _zip_bytes(
        {
            "repo-abc/plugins/p/.claude-plugin/plugin.json": b"{}",
            "repo-abc/docs/large.bin": b"x" * 5000,
        }
    )

    data = plugin_archive.normalize_archive(raw, strip_top_level=True, subdir="plugins/p")

    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        assert archive.namelist() == [".claude-plugin/plugin.json"]


def test_normalize_ignores_unsafe_paths_outside_the_selected_subdirectory():
    raw = _zip_bytes(
        {
            "repo-abc/plugins/p/.claude-plugin/plugin.json": b"{}",
            "repo-abc/other/../../escape.txt": b"x",
        }
    )

    data = plugin_archive.normalize_archive(raw, strip_top_level=True, subdir="plugins/p")

    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        assert archive.namelist() == [".claude-plugin/plugin.json"]


@pytest.mark.parametrize("name", ["../escape", "a/../../escape", "/etc/passwd"])
def test_stored_archive_paths_cannot_leave_the_archive_directory(archive_dir, name):
    with pytest.raises(plugin_archive.ArchiveBuildError, match="not a valid plugin name"):
        plugin_archive.store_archive(name, b"x")

    assert not any(archive_dir.parent.glob("escape.zip"))


def test_store_archive_leaves_no_temp_file_when_the_write_fails(archive_dir, monkeypatch):
    """A failed publish must not litter the archive dir with half-written .tmp files."""

    def _boom(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(plugin_archive.os, "replace", _boom)

    with pytest.raises(OSError, match="disk full"):
        plugin_archive.store_archive("s3-skill", b"data")

    assert list(archive_dir.iterdir()) == []


def test_normalize_rejects_a_repository_archive_without_a_single_root():
    raw = _zip_bytes({"one/.claude-plugin/plugin.json": b"{}", "two/README.md": b"x"})

    with pytest.raises(plugin_archive.ArchiveBuildError, match="single top-level directory"):
        plugin_archive.normalize_archive(raw, strip_top_level=True)


def test_normalize_rejects_a_source_that_is_not_a_zip():
    with pytest.raises(plugin_archive.ArchiveBuildError, match="not a valid ZIP archive"):
        plugin_archive.normalize_archive(b"this is not a zip", strip_top_level=False)


def test_normalize_rejects_an_oversized_rebuilt_archive(monkeypatch):
    """The post-rebuild size check is the last line of defence against a compression bomb."""
    monkeypatch.setattr(plugin_archive, "MAX_ARCHIVE_BYTES", 100)
    monkeypatch.setattr(plugin_archive, "_write_zip", lambda _members: b"x" * 101)

    with pytest.raises(plugin_archive.ArchiveBuildError, match="archive is larger than"):
        plugin_archive.normalize_archive(_zip_bytes({".claude-plugin/plugin.json": b"{}"}), strip_top_level=False)


@pytest.mark.asyncio
async def test_plugin_archive_get_maps_an_ssrf_rejection_to_a_build_error(monkeypatch):
    async def _blocked(*_args, **_kwargs):
        raise plugin_archive.SSRFError("private IP")

    monkeypatch.setattr(plugin_archive, "async_safe_get", _blocked)

    with pytest.raises(plugin_archive.ArchiveBuildError, match=r"refusing to fetch .*: private IP"):
        await plugin_archive._get("http://169.254.169.254/plugin.zip")


@pytest.mark.asyncio
async def test_plugin_archive_get_maps_a_transport_failure_to_a_build_error(monkeypatch):
    async def _fails(*_args, **_kwargs):
        raise httpx.ConnectTimeout("timed out")

    monkeypatch.setattr(plugin_archive, "async_safe_get", _fails)

    with pytest.raises(plugin_archive.ArchiveBuildError, match=r"could not fetch .*: timed out"):
        await plugin_archive._get("https://example.com/plugin.zip")


@pytest.mark.asyncio
async def test_plugin_archive_get_rejects_a_non_200_response(monkeypatch):
    async def _not_found(*_args, **_kwargs):
        return httpx.Response(404, content=b"nope")

    monkeypatch.setattr(plugin_archive, "async_safe_get", _not_found)

    with pytest.raises(plugin_archive.ArchiveBuildError, match="returned HTTP 404"):
        await plugin_archive._get("https://example.com/plugin.zip")


@pytest.mark.asyncio
async def test_plugin_archive_get_rejects_a_body_over_the_download_limit(monkeypatch):
    """A client that ignores max_response_bytes still cannot hand us an oversized body."""
    monkeypatch.setattr(plugin_archive, "MAX_DOWNLOAD_BYTES", 4)

    async def _too_big(*_args, **_kwargs):
        return httpx.Response(200, content=b"much too long")

    monkeypatch.setattr(plugin_archive, "async_safe_get", _too_big)

    with pytest.raises(plugin_archive.ArchiveBuildError, match=r"is larger than the .* plugin limit"):
        await plugin_archive._get("https://example.com/plugin.zip")


@pytest.mark.asyncio
async def test_build_archive_rejects_a_malformed_github_repo(monkeypatch):
    monkeypatch.setattr(plugin_archive, "build_archive", _REAL_BUILD_ARCHIVE)

    with pytest.raises(plugin_archive.ArchiveBuildError, match="not a valid GitHub repository"):
        await _REAL_BUILD_ARCHIVE({"source": "github", "repo": "not-a-repo"})


@pytest.mark.asyncio
async def test_build_archive_rejects_a_malformed_git_ref(monkeypatch):
    monkeypatch.setattr(plugin_archive, "build_archive", _REAL_BUILD_ARCHIVE)

    with pytest.raises(plugin_archive.ArchiveBuildError, match="not a valid git ref"):
        await _REAL_BUILD_ARCHIVE({"source": "github", "repo": "org/repo", "ref": "main; rm -rf /"})


@pytest.mark.asyncio
async def test_build_archive_from_a_github_source_skips_ref_resolution_for_a_sha(monkeypatch):
    """A 40-hex ref is already immutable, so no commits API round trip is needed."""
    monkeypatch.setattr(plugin_archive, "build_archive", _REAL_BUILD_ARCHIVE)
    commit = "d" * 40
    raw = _zip_bytes({"repo-d/.claude-plugin/plugin.json": b"{}"})
    get = AsyncMock(return_value=raw)
    monkeypatch.setattr(plugin_archive, "_get", get)

    built = await _REAL_BUILD_ARCHIVE({"source": "github", "repo": "org/repo", "ref": commit})

    assert built.ref == commit
    assert [call.args[0] for call in get.await_args_list] == [f"https://codeload.github.com/org/repo/zip/{commit}"]


@pytest.mark.asyncio
async def test_build_archive_rejects_a_ref_that_does_not_resolve_to_a_commit(monkeypatch):
    monkeypatch.setattr(plugin_archive, "build_archive", _REAL_BUILD_ARCHIVE)
    monkeypatch.setattr(plugin_archive, "_get", AsyncMock(return_value=b"Not Found"))

    with pytest.raises(plugin_archive.ArchiveBuildError, match="could not resolve 'no-such-branch'"):
        await _REAL_BUILD_ARCHIVE({"source": "github", "repo": "org/repo", "ref": "no-such-branch"})


@pytest.mark.asyncio
async def test_build_archive_from_an_archive_source_returns_the_normalized_zip(monkeypatch):
    monkeypatch.setattr(plugin_archive, "build_archive", _REAL_BUILD_ARCHIVE)
    raw = _zip_bytes({"my-plugin/.claude-plugin/plugin.json": b"{}"})
    monkeypatch.setattr(plugin_archive, "_get", AsyncMock(return_value=raw))

    built = await _REAL_BUILD_ARCHIVE(
        {
            "source": "archive",
            "url": "https://example.com/plugin.zip",
            "sha256": plugin_archive.sha256_hex(raw),
        }
    )

    assert built.ref is None
    assert built.sha256 == plugin_archive.sha256_hex(built.data)
    with zipfile.ZipFile(io.BytesIO(built.data)) as archive:
        assert archive.namelist() == [".claude-plugin/plugin.json"]


@pytest.mark.asyncio
async def test_archive_download_serves_the_zip_another_worker_wrote_while_we_waited(archive_dir, monkeypatch):
    """The second read under the lock wins the race, so only one worker refetches."""
    build = _patch_build(monkeypatch)
    await register_plugin(
        request=RegisterPluginRequest(name="s3-skill", source=_ARCHIVE_SOURCE), user_api_key_dict=_USER
    )
    build.reset_mock()
    monkeypatch.setattr(plugin_archive, "read_stored_archive", MagicMock(side_effect=[None, _PLUGIN_ZIP]))

    response = await get_plugin_archive("s3-skill", request=_marketplace_request())

    assert response.body == _PLUGIN_ZIP
    build.assert_not_awaited()


@pytest.mark.asyncio
async def test_archive_refetch_failure_is_reported_as_a_bad_gateway(archive_dir, monkeypatch):
    _patch_build(monkeypatch)
    await register_plugin(
        request=RegisterPluginRequest(name="s3-skill", source=_ARCHIVE_SOURCE), user_api_key_dict=_USER
    )
    (archive_dir / "s3-skill.zip").unlink()
    monkeypatch.setattr(
        plugin_archive, "build_archive", AsyncMock(side_effect=plugin_archive.ArchiveBuildError("upstream is gone"))
    )

    with pytest.raises(HTTPException) as exc:
        await get_plugin_archive("s3-skill", request=_marketplace_request())

    assert exc.value.status_code == 502
    assert exc.value.detail["error"] == ("Could not rebuild the archive for plugin 's3-skill': upstream is gone")


@pytest.mark.asyncio
async def test_archive_download_of_a_plugin_with_no_source_in_its_manifest_is_404(archive_dir, monkeypatch):
    _patch_build(monkeypatch)
    await register_plugin(
        request=RegisterPluginRequest(name="s3-skill", source=_ARCHIVE_SOURCE), user_api_key_dict=_USER
    )
    table = litellm.proxy.proxy_server.prisma_client.db.litellm_claudecodeplugintable
    manifest = await _read_stored_manifest("s3-skill")
    await table.update(
        where={"name": "s3-skill"},
        data={"manifest_json": json.dumps({k: v for k, v in manifest.items() if k != "source"})},
    )

    with pytest.raises(HTTPException) as exc:
        await get_plugin_archive("s3-skill", request=_marketplace_request())

    assert exc.value.status_code == 404
    assert exc.value.detail["error"] == "Plugin 's3-skill' has no hosted archive"


@pytest.mark.asyncio
async def test_update_to_a_hosted_archive_publishes_the_new_zip(archive_dir, monkeypatch):
    _patch_build(monkeypatch)
    await register_plugin(
        request=RegisterPluginRequest(name="s3-skill", source=_ARCHIVE_SOURCE, serve_archive=False),
        user_api_key_dict=_USER,
    )
    assert not (archive_dir / "s3-skill.zip").exists()

    response = await update_plugin(
        plugin_name="s3-skill",
        request=UpdatePluginRequest(source=_ARCHIVE_SOURCE, serve_archive=True),
        user_api_key_dict=_USER,
    )

    assert response.plugin.serve_archive is True
    assert (archive_dir / "s3-skill.zip").read_bytes() == _PLUGIN_ZIP
