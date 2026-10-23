"""
Unit tests for claude_code_marketplace.py source validation.

Covers the git-subdir and archive source types added alongside the existing github and url types.
"""

import base64
import hashlib
import io
import json
import zipfile
from contextlib import asynccontextmanager
from typing import Final, cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI, HTTPException, Request
from fastapi.testclient import TestClient
from pydantic import ValidationError
from starlette.types import Message, Receive, Scope, Send

import litellm
from litellm.proxy import proxy_server
from litellm.proxy._types import LiteLLM_ObjectPermissionTable, ProxyException, UserAPIKeyAuth
from litellm.proxy.anthropic_endpoints.claude_code_endpoints import claude_code_marketplace
from litellm.proxy.anthropic_endpoints.claude_code_endpoints.claude_code_marketplace import (
    delete_plugin,
    disable_plugin,
    download_plugin_archive,
    enable_plugin,
    get_marketplace,
    get_plugin,
    list_plugins,
    register_plugin,
    update_plugin,
    upload_plugin_archive,
)
from litellm.proxy.auth import auth_checks
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.proxy_server import LitellmUserRoles
from litellm.types.proxy.claude_code_endpoints import (
    InstallationPreference,
    MarketplaceResponse,
    RegisterPluginRequest,
    UpdatePluginRequest,
)
from tests._master_key import MASTER_KEY


def _make_mock_prisma():
    """Stateful prisma mock that supports find_unique, find_many, create, and update."""
    store: dict[str, MagicMock] = {}
    archive_store: dict[str, MagicMock] = {}

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
        record.archive_sha256 = data.get("archive_sha256")
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

    archive_table = MagicMock()

    async def _find_archive(where: dict[str, str]):
        return archive_store.get(where.get("plugin_id"))

    async def _upsert_archive(where: dict[str, str], data: dict[str, object]):
        archive_data = data["update"] if where["plugin_id"] in archive_store else data["create"]
        record = MagicMock()
        record.plugin_id = archive_data["plugin_id"]
        record.content = archive_data["content"]
        record.size_bytes = archive_data["size_bytes"]
        record.updated_at = archive_data["updated_at"]
        record.model_dump = MagicMock(return_value={"content": base64.b64encode(record.content.decode()).decode()})
        archive_store[where["plugin_id"]] = record
        return record

    async def _delete_archive(where: dict[str, str]):
        archive_store.pop(where["plugin_id"], None)

    archive_table.find_unique = AsyncMock(side_effect=_find_archive)
    archive_table.upsert = AsyncMock(side_effect=_upsert_archive)
    archive_table.delete_many = AsyncMock(side_effect=_delete_archive)
    mock_client.db.litellm_claudecodepluginarchivetable = archive_table

    @asynccontextmanager
    async def _transaction():
        yield mock_client.db

    mock_client.db.tx = _transaction
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


@pytest.fixture(autouse=True)
def _patch_proxy_globals(monkeypatch):
    """Scope prisma_client/master_key mutations to each test via monkeypatch."""
    monkeypatch.setattr(litellm.proxy.proxy_server, "prisma_client", _make_mock_prisma())
    monkeypatch.setattr(litellm.proxy.proxy_server, "master_key", MASTER_KEY)


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
async def test_list_plugins_returns_500_for_invalid_manifest_json():
    await register_plugin(
        request=RegisterPluginRequest(name="invalid-manifest", source=_GIT_SUBDIR_SOURCE, version="1.0.0"),
        user_api_key_dict=_USER,
    )
    table: Final = litellm.proxy.proxy_server.prisma_client.db.litellm_claudecodeplugintable
    await table.update(where={"name": "invalid-manifest"}, data={"manifest_json": "{"})

    with pytest.raises(HTTPException) as exc_info:
        await list_plugins(user_api_key_dict=_USER)

    assert exc_info.value.status_code == 500


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

    header_request: Final = MagicMock(headers={"Authorization": "Bearer sk-granted"})
    header_response: Final = await get_marketplace(request=header_request)
    assert sorted(plugin["name"] for plugin in json.loads(header_response.body)["plugins"]) == [
        "private-skill",
        "public-skill",
    ]

    with pytest.raises(ProxyException) as exc_info:
        await get_marketplace(request=MagicMock(), key="sk-bogus")
    assert exc_info.value.code == "401"
    with pytest.raises(ProxyException) as header_error:
        await get_marketplace(request=MagicMock(headers={"Authorization": "Bearer sk-bogus"}))
    assert header_error.value.code == "401"


def _plugin_archive(manifest_path: str = ".claude-plugin/plugin.json") -> bytes:
    archive_bytes: Final = io.BytesIO()
    with zipfile.ZipFile(archive_bytes, "w") as archive:
        archive.writestr(manifest_path, '{"name":"hosted-plugin"}')
    return archive_bytes.getvalue()


def _archive_request(
    content: bytes,
    *,
    content_type: str = "application/zip",
    content_length: str | None = None,
    receive: Receive | None = None,
) -> Request:
    content_headers: Final = ((b"content-type", content_type.encode()),)
    length_headers: Final = () if content_length is None else ((b"content-length", content_length.encode()),)
    headers: Final = (*content_headers, *length_headers)

    async def receive_body() -> Message:
        return {"type": "http.request", "body": content, "more_body": False}

    scope: Final[Scope] = {
        "type": "http",
        "http_version": "1.1",
        "method": "PUT",
        "scheme": "http",
        "path": "/claude-code/plugins/hosted-plugin/archive",
        "raw_path": b"/claude-code/plugins/hosted-plugin/archive",
        "query_string": b"",
        "headers": list(headers),
        "server": ("testserver", 80),
        "client": ("127.0.0.1", 1234),
    }
    return Request(scope, receive or receive_body)


def _register_request(
    name: str,
    source: dict[str, str],
    installation_preference: InstallationPreference | None = None,
) -> RegisterPluginRequest:
    return RegisterPluginRequest(
        name=name,
        source=source,
        version="1.0.0",
        description=None,
        author=None,
        homepage=None,
        keywords=None,
        category=None,
        domain=None,
        namespace=None,
        installation_preference=installation_preference,
    )


def _update_request(source: dict[str, str]) -> UpdatePluginRequest:
    return UpdatePluginRequest(
        source=source,
        version=None,
        description=None,
        author=None,
        homepage=None,
        keywords=None,
        category=None,
        domain=None,
        namespace=None,
        installation_preference=None,
    )


async def _register_hosted_plugin(name: str = "hosted-plugin") -> None:
    await register_plugin(
        request=_register_request(name, {"source": "hosted"}),
        user_api_key_dict=_USER,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "manifest_path",
    [".claude-plugin/plugin.json", "bundle/.claude-plugin/plugin.json"],
)
async def test_hosted_plugin_archive_upload_and_download(manifest_path: str):
    await _register_hosted_plugin()
    content: Final = _plugin_archive(manifest_path)
    result: Final = await upload_plugin_archive(
        plugin_name="hosted-plugin",
        request=_archive_request(content),
        user_api_key_dict=_USER,
    )
    assert result.sha256 == hashlib.sha256(content).hexdigest()

    downloaded: Final = await download_plugin_archive(
        plugin_name="hosted-plugin",
        request=MagicMock(headers={}),
    )
    assert downloaded.body == content
    assert downloaded.media_type == "application/zip"
    assert downloaded.headers["content-disposition"] == 'attachment; filename="hosted-plugin.zip"'
    listed: Final = await list_plugins(user_api_key_dict=_USER)
    stored_plugin: Final = await get_plugin(plugin_name="hosted-plugin", user_api_key_dict=_USER)
    assert next(plugin for plugin in listed.plugins if plugin.name == "hosted-plugin").archive_sha256 == result.sha256
    assert stored_plugin["archive_sha256"] == result.sha256


@pytest.mark.asyncio
@pytest.mark.parametrize("content_type", ["application/zip", "application/octet-stream"])
async def test_upload_plugin_archive_accepts_raw_body_route(
    monkeypatch: pytest.MonkeyPatch,
    content_type: str,
):
    await _register_hosted_plugin()
    prisma_client: Final = litellm.proxy.proxy_server.prisma_client
    monkeypatch.setattr(claude_code_marketplace, "get_prisma_client", AsyncMock(return_value=prisma_client))

    app: Final = FastAPI()
    app.include_router(claude_code_marketplace.router)
    app.dependency_overrides[user_api_key_auth] = lambda: _USER
    content: Final = _plugin_archive()

    with TestClient(app) as client:
        response = client.put(
            "/claude-code/plugins/hosted-plugin/archive",
            content=content,
            headers={"Content-Type": content_type},
        )

    assert response.status_code == 200
    assert response.json()["sha256"] == hashlib.sha256(content).hexdigest()


@pytest.mark.asyncio
async def test_upload_plugin_archive_rejects_unauthenticated_request_before_storage(monkeypatch: pytest.MonkeyPatch):
    await _register_hosted_plugin()
    prisma_client: Final = litellm.proxy.proxy_server.prisma_client
    monkeypatch.setattr(claude_code_marketplace, "get_prisma_client", AsyncMock(return_value=prisma_client))
    app: Final = FastAPI()
    app.include_router(claude_code_marketplace.router)

    def reject_unauthenticated_request() -> UserAPIKeyAuth:
        raise HTTPException(status_code=401, detail="Unauthorized")

    app.dependency_overrides[user_api_key_auth] = reject_unauthenticated_request
    content: Final = _plugin_archive()
    receive_spy: Final = MagicMock()

    async def track_body_reads(scope: Scope, receive: Receive, send: Send) -> None:
        async def tracked_receive() -> Message:
            message: Final = await receive()
            if message["type"] == "http.request" and message.get("body"):
                receive_spy(message)
            return message

        await app(scope, tracked_receive, send)

    with TestClient(track_body_reads) as client:
        response = client.put(
            "/claude-code/plugins/hosted-plugin/archive",
            content=content,
            headers={"Content-Type": "application/zip"},
        )

    archive: Final = await prisma_client.db.litellm_claudecodepluginarchivetable.find_unique(
        where={"plugin_id": "test-id"}
    )
    assert response.status_code == 401
    assert receive_spy.call_count == 0
    assert archive is None


@pytest.mark.asyncio
async def test_private_hosted_archive_uses_inference_key_visibility(monkeypatch: pytest.MonkeyPatch):
    await _register_hosted_plugin("private-plugin")
    await disable_plugin(plugin_name="private-plugin", user_api_key_dict=_USER)
    content: Final = _plugin_archive()
    await upload_plugin_archive(
        plugin_name="private-plugin",
        request=_archive_request(content),
        user_api_key_dict=_USER,
    )
    keys: dict[str, UserAPIKeyAuth] = {"sk-granted": _granted_user(["private-plugin"]), "sk-plain": _NON_ADMIN_USER}

    async def _fake_auth(request: Request, api_key: str) -> UserAPIKeyAuth:
        token = api_key.removeprefix("Bearer ")
        if token not in keys:
            raise ProxyException(message="invalid key", type="auth_error", param="key", code=401)
        return keys[token]

    monkeypatch.setattr(claude_code_marketplace, "user_api_key_auth", _fake_auth)
    hidden_request = MagicMock(headers={"Authorization": "Bearer sk-plain"})
    with pytest.raises(HTTPException) as hidden_error:
        await download_plugin_archive("private-plugin", hidden_request)
    assert hidden_error.value.status_code == 404

    granted_request = MagicMock(headers={"Authorization": "Bearer sk-granted"})
    visible_archive: Final = await download_plugin_archive("private-plugin", granted_request)
    assert visible_archive.body == content


@pytest.mark.asyncio
async def test_access_group_grant_exposes_disabled_hosted_archive_and_marketplace(monkeypatch: pytest.MonkeyPatch):
    await _register_hosted_plugin("group-only-plugin")
    await disable_plugin(plugin_name="group-only-plugin", user_api_key_dict=_USER)
    content: Final = _plugin_archive()
    await upload_plugin_archive(
        plugin_name="group-only-plugin",
        request=_archive_request(content),
        user_api_key_dict=_USER,
    )

    group_key: Final = UserAPIKeyAuth(
        api_key="sk-group-only",
        token="sk-group-only",
        user_role=LitellmUserRoles.INTERNAL_USER,
        access_group_ids=["plugin-access-group"],
    )
    access_group: Final = MagicMock()
    access_group.access_group_id = "plugin-access-group"
    access_group.access_skill_names = ["group-only-plugin"]
    access_group.assigned_team_ids = []
    access_group.assigned_key_ids = [group_key.token]
    keys: Final[dict[str, UserAPIKeyAuth]] = {
        "sk-group-only": group_key,
        "sk-no-group": _NON_ADMIN_USER,
    }

    async def _fake_auth(request: Request, api_key: str) -> UserAPIKeyAuth:
        assert request.headers.get("Authorization") == api_key
        return keys[api_key.removeprefix("Bearer ")]

    monkeypatch.setattr(claude_code_marketplace, "user_api_key_auth", _fake_auth)
    monkeypatch.setattr(auth_checks, "get_access_object", AsyncMock(return_value=access_group))
    monkeypatch.setattr(auth_checks, "get_skill_names_from_access_groups", AsyncMock(return_value=frozenset()))
    monkeypatch.setattr(proxy_server, "user_api_key_cache", MagicMock())
    monkeypatch.setenv("PROXY_BASE_URL", "https://marketplace.example.com")

    grouped_request: Final = cast(Request, MagicMock(headers={"Authorization": "Bearer sk-group-only"}))
    marketplace_response: Final = await get_marketplace(request=grouped_request)
    marketplace_plugins: Final = json.loads(bytes(marketplace_response.body))["plugins"]
    assert marketplace_plugins == [
        {
            "name": "group-only-plugin",
            "source": {
                "source": "archive",
                "url": "https://marketplace.example.com/claude-code/plugins/group-only-plugin/archive.zip",
                "sha256": hashlib.sha256(content).hexdigest(),
            },
            "version": "1.0.0",
        }
    ]

    archive: Final = await download_plugin_archive("group-only-plugin", grouped_request)
    assert archive.status_code == 200
    assert archive.body == content

    without_group_request: Final = cast(Request, MagicMock(headers={"Authorization": "Bearer sk-no-group"}))
    with pytest.raises(HTTPException) as hidden_error:
        await download_plugin_archive("group-only-plugin", without_group_request)
    assert hidden_error.value.status_code == 404
    assert hidden_error.value.detail == {"error": "Plugin not found"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "source",
    [{"source": "hosted", "url": "https://example.com/archive.zip"}, {"source": "hosted", "sha256": "abc"}],
)
async def test_register_rejects_hosted_source_extra_keys(source: dict[str, str]):
    with pytest.raises(HTTPException) as exc_info:
        await register_plugin(
            request=_register_request("hosted-plugin", source),
            user_api_key_dict=_USER,
        )

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "source",
    [{"source": "hosted", "url": "https://example.com/archive.zip"}, {"source": "hosted", "sha256": "abc"}],
)
async def test_update_rejects_hosted_source_extra_keys(source: dict[str, str]):
    await register_plugin(
        request=_register_request("existing-plugin", _GIT_SUBDIR_SOURCE),
        user_api_key_dict=_USER,
    )

    with pytest.raises(HTTPException) as exc_info:
        await update_plugin(
            plugin_name="existing-plugin",
            request=_update_request(source),
            user_api_key_dict=_USER,
        )

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_hosted_marketplace_uses_request_origin_and_digest(monkeypatch: pytest.MonkeyPatch):
    await _register_hosted_plugin()
    content: Final = _plugin_archive()
    await upload_plugin_archive(
        plugin_name="hosted-plugin",
        request=_archive_request(content),
        user_api_key_dict=_USER,
    )
    monkeypatch.setenv("PROXY_BASE_URL", "https://proxy.example.com/root")

    response: Final = await get_marketplace(request=MagicMock(headers={}))
    hosted_entry: Final = next(
        plugin for plugin in json.loads(bytes(response.body))["plugins"] if plugin["name"] == "hosted-plugin"
    )
    assert hosted_entry["source"] == {
        "source": "archive",
        "url": "https://proxy.example.com/root/claude-code/plugins/hosted-plugin/archive.zip",
        "sha256": hashlib.sha256(content).hexdigest(),
    }


@pytest.mark.asyncio
async def test_hosted_marketplace_uses_request_origin_without_configured_proxy_url(monkeypatch: pytest.MonkeyPatch):
    await _register_hosted_plugin()
    content: Final = _plugin_archive()
    await upload_plugin_archive(
        plugin_name="hosted-plugin",
        request=_archive_request(content),
        user_api_key_dict=_USER,
    )
    monkeypatch.delenv("PROXY_BASE_URL", raising=False)
    request: Final = Request(
        {
            "type": "http",
            "http_version": "1.1",
            "method": "GET",
            "scheme": "https",
            "path": "/claude-code/marketplace.json",
            "raw_path": b"/claude-code/marketplace.json",
            "query_string": b"",
            "headers": [],
            "server": ("marketplace.example.com", 443),
            "client": ("127.0.0.1", 1234),
        }
    )

    response: Final = await get_marketplace(request=request)
    hosted_entry: Final = next(
        plugin for plugin in json.loads(bytes(response.body))["plugins"] if plugin["name"] == "hosted-plugin"
    )
    assert hosted_entry["source"]["url"] == (
        "https://marketplace.example.com/claude-code/plugins/hosted-plugin/archive.zip"
    )


@pytest.mark.asyncio
async def test_hosted_plugin_without_archive_is_omitted_from_marketplace(caplog: pytest.LogCaptureFixture):
    await _register_hosted_plugin()

    response: Final = await get_marketplace(request=MagicMock(headers={}))

    assert "hosted-plugin" not in {plugin["name"] for plugin in json.loads(bytes(response.body))["plugins"]}
    assert "has no archive" in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "plugin_name",
    ["missing-plugin", "non-hosted-plugin", "hosted-plugin"],
)
async def test_archive_download_hides_missing_non_hosted_and_archive_less_plugins(plugin_name: str):
    await register_plugin(
        request=_register_request("non-hosted-plugin", _GIT_SUBDIR_SOURCE),
        user_api_key_dict=_USER,
    )
    await _register_hosted_plugin()

    with pytest.raises(HTTPException) as exc_info:
        await download_plugin_archive(plugin_name=plugin_name, request=MagicMock(headers={}))

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == {"error": "Plugin not found"}


@pytest.mark.asyncio
async def test_upload_archive_requires_admin_and_rejects_invalid_zip():
    await _register_hosted_plugin()
    with pytest.raises(HTTPException) as auth_error:
        await upload_plugin_archive("hosted-plugin", _archive_request(b"not a zip"), _NON_ADMIN_USER)
    assert auth_error.value.status_code == 403

    with pytest.raises(HTTPException) as zip_error:
        await upload_plugin_archive(
            "hosted-plugin",
            _archive_request(b"not a zip"),
            _USER,
        )
    assert zip_error.value.status_code == 400


@pytest.mark.asyncio
async def test_upload_archive_enforces_stream_size_limit(monkeypatch: pytest.MonkeyPatch):
    await _register_hosted_plugin()
    monkeypatch.setattr(claude_code_marketplace, "CLAUDE_CODE_PLUGIN_ARCHIVE_MAX_BYTES", 4)

    with pytest.raises(HTTPException) as exc_info:
        await upload_plugin_archive(
            "hosted-plugin",
            _archive_request(b"12345"),
            _USER,
        )

    assert exc_info.value.status_code == 413
    archive: Final = await litellm.proxy.proxy_server.prisma_client.db.litellm_claudecodepluginarchivetable.find_unique(
        where={"plugin_id": "test-id"}
    )
    assert archive is None


@pytest.mark.asyncio
async def test_upload_archive_rejects_over_limit_content_length_without_reading_body(
    monkeypatch: pytest.MonkeyPatch,
):
    await _register_hosted_plugin()
    monkeypatch.setattr(claude_code_marketplace, "CLAUDE_CODE_PLUGIN_ARCHIVE_MAX_BYTES", 4)
    receive: Final = AsyncMock(return_value={"type": "http.request", "body": _plugin_archive(), "more_body": False})
    request: Final = _archive_request(
        _plugin_archive(),
        content_length="5",
        receive=cast(Receive, receive),
    )

    with pytest.raises(HTTPException) as exc_info:
        await upload_plugin_archive("hosted-plugin", request, _USER)

    archive: Final = await litellm.proxy.proxy_server.prisma_client.db.litellm_claudecodepluginarchivetable.find_unique(
        where={"plugin_id": "test-id"}
    )
    assert exc_info.value.status_code == 413
    assert receive.await_count == 0
    assert archive is None


@pytest.mark.asyncio
async def test_upload_archive_rejects_unsupported_content_type():
    await _register_hosted_plugin()

    with pytest.raises(HTTPException) as exc_info:
        await upload_plugin_archive(
            "hosted-plugin",
            _archive_request(_plugin_archive(), content_type="application/json"),
            _USER,
        )

    assert exc_info.value.status_code == 415


@pytest.mark.asyncio
async def test_hosted_plugin_accepts_auto_install_preference():
    response: Final = await register_plugin(
        request=_register_request("hosted-auto-install", {"source": "hosted"}, "auto_install"),
        user_api_key_dict=_USER,
    )
    assert response.plugin.source == {"source": "hosted"}
    assert response.plugin.installation_preference == "auto_install"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "manifest_path",
    ["../.claude-plugin/plugin.json", "/.claude-plugin/plugin.json", "folder/extra/.claude-plugin/plugin.json"],
)
async def test_upload_archive_rejects_unsafe_or_deep_manifest_paths(manifest_path: str):
    await _register_hosted_plugin()

    with pytest.raises(HTTPException) as exc_info:
        await upload_plugin_archive(
            "hosted-plugin",
            _archive_request(_plugin_archive(manifest_path)),
            _USER,
        )

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_update_away_from_hosted_source_clears_archive():
    await _register_hosted_plugin()
    await upload_plugin_archive(
        "hosted-plugin",
        _archive_request(_plugin_archive()),
        _USER,
    )

    await update_plugin(
        plugin_name="hosted-plugin",
        request=_update_request(_GIT_SUBDIR_SOURCE),
        user_api_key_dict=_USER,
    )

    plugin: Final = await litellm.proxy.proxy_server.prisma_client.db.litellm_claudecodeplugintable.find_unique(
        where={"name": "hosted-plugin"}
    )
    archive: Final = await litellm.proxy.proxy_server.prisma_client.db.litellm_claudecodepluginarchivetable.find_unique(
        where={"plugin_id": plugin.id}
    )
    assert plugin.archive_sha256 is None
    assert archive is None


@pytest.mark.asyncio
@pytest.mark.parametrize("preference", ["available", "auto_install", "required"])
async def test_get_marketplace_emits_installation_preference(preference):
    await register_plugin(
        request=RegisterPluginRequest(
            name="s3-skill",
            source=_ARCHIVE_SOURCE,
            installation_preference=preference,
        ),
        user_api_key_dict=_USER,
    )

    marketplace = json.loads((await get_marketplace(request=MagicMock())).body)

    assert marketplace["plugins"] == [
        {"name": "s3-skill", "source": _ARCHIVE_SOURCE, "version": "1.0.0", "installationPreference": preference}
    ]


@pytest.mark.asyncio
async def test_get_marketplace_preserves_non_hosted_entry_key_order_and_content():
    await register_plugin(
        request=RegisterPluginRequest(
            name="marketplace-shape",
            source=_ARCHIVE_SOURCE,
            version="2.3.4",
            description="Marketplace compatibility",
            author={"name": "Ada Lovelace", "email": "ada@example.com"},
            homepage="https://example.com/plugin",
            keywords=["analysis", "tools"],
            category="development",
            installation_preference="available",
        ),
        user_api_key_dict=_USER,
    )

    request = Request(
        {
            "type": "http",
            "method": "GET",
            "scheme": "https",
            "server": ("gateway.example", 443),
            "client": ("127.0.0.1", 12345),
            "headers": [],
            "path": "/claude-code/marketplace.json",
            "query_string": b"",
        }
    )
    marketplace = json.loads((await get_marketplace(request=request)).body)
    entry = marketplace["plugins"][0]
    expected = {
        "name": "marketplace-shape",
        "source": _ARCHIVE_SOURCE,
        "installationPreference": "available",
        "version": "2.3.4",
        "description": "Marketplace compatibility",
        "author": {"name": "Ada Lovelace", "email": "ada@example.com"},
        "homepage": "https://example.com/plugin",
        "keywords": ["analysis", "tools"],
        "category": "development",
    }

    assert tuple(entry.items()) == tuple(expected.items())


@pytest.mark.parametrize(
    "build_request",
    [
        lambda preference: RegisterPluginRequest(
            name="s3-skill", source=_ARCHIVE_SOURCE, installation_preference=preference
        ),
        lambda preference: UpdatePluginRequest(source=_ARCHIVE_SOURCE, installation_preference=preference),
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
        request=RegisterPluginRequest(name=name, source=_ARCHIVE_SOURCE),
        user_api_key_dict=_USER,
    )

    response = await update_plugin(
        plugin_name=name,
        request=UpdatePluginRequest(
            source=_ARCHIVE_SOURCE,
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
        request=RegisterPluginRequest(name=name, source=_ARCHIVE_SOURCE, installation_preference="auto_install"),
        user_api_key_dict=_USER,
    )

    await update_plugin(
        plugin_name=name,
        request=UpdatePluginRequest(source=_ARCHIVE_SOURCE),
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
        {"name": "s3-skill", "source": _ARCHIVE_SOURCE, "installationPreference": "auto_install"}
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
            request=RegisterPluginRequest(name="no-op-plugin", source=source, installation_preference=preference),
            user_api_key_dict=_USER,
        )
    await register_plugin(request=RegisterPluginRequest(name="no-op-plugin", source=source), user_api_key_dict=_USER)
    with pytest.raises(HTTPException) as update_error:
        await update_plugin(
            plugin_name="no-op-plugin",
            request=UpdatePluginRequest(source=source, installation_preference=preference),
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
            name="listed-plugin", source=_UNPINNED_ARCHIVE_SOURCE, installation_preference="available"
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
    request = build_request({"source": _ARCHIVE_SOURCE, null_key: None, value_key: "auto_install"})

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
        request=RegisterPluginRequest(name="s3-skill", source=_ARCHIVE_SOURCE),
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
        request=RegisterPluginRequest(name="unpinned-skill", source=source),
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
        request=UpdatePluginRequest(source=_ARCHIVE_SOURCE),
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
