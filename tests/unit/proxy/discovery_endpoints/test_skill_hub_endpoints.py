from __future__ import annotations

import asyncio
import hashlib
import io
import json
import stat
import threading
import zipfile
import zlib
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Final
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from prisma import models as prisma_models
from prisma.errors import ForeignKeyViolationError, UniqueViolationError

import litellm
from litellm.caching.caching import DualCache
from litellm.proxy import proxy_server
from litellm.proxy._types import LiteLLM_ObjectPermissionTable, UserAPIKeyAuth
from litellm.proxy.deps.repositories import get_plugin_upload_repository, get_skill_submission_repository
from litellm.proxy.discovery_endpoints.agent_skills_archive import (
    MAX_ARCHIVE_ENTRIES,
    MAX_ARCHIVE_UNPACKED_BYTES,
)
from litellm.proxy.discovery_endpoints.skill_hub_endpoints import import_http_client
from litellm.proxy.hooks.parallel_request_limiter_v3 import PROXY_MaxParallelRequestsHandler_v3
from litellm.proxy.utils import InternalUsageCache
from litellm.repositories.plugin_upload_repository import PluginUploadRepository
from litellm.repositories.skill_submission_repository import SkillSubmissionRepository
from tests._master_key import MASTER_KEY
from tests.unit.proxy.skill_submission_rows import SubmissionRows

NAME: Final = "portable-notes"
MANIFEST: Final = (
    b"---\nname: portable-notes\ndescription: Summarize synthetic notes.\n---\n\nRead references/guide.md.\n"
)
FILES: Final = {
    "SKILL.md": MANIFEST,
    "references/guide.md": b"Use only the supplied notes.\n",
    "assets/example.bin": b"\x00\x01\xff",
    "scripts/check.sh": b"#!/bin/sh\nprintf 'synthetic notes\\n'\n",
}
HUB: Final = "/skill-hub/skills"
INDEX: Final = "/.well-known/agent-skills/index.json"
GRANTED_KEY: Final = "sk-hub-granted"
PLAIN_KEY: Final = "sk-hub-plain"
MEMBER_KEY: Final = "sk-hub-member"
OTHER_KEY: Final = "sk-hub-other"
VIEWER_KEY: Final = "sk-hub-viewer"


def _zip(files: Mapping[str, bytes], prefix: str = "") -> bytes:
    buffer: Final = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for path, content in files.items():
            entry: Final = zipfile.ZipInfo(f"{prefix}{path}", (2024, 1, 1, 0, 0, 0))
            entry.create_system = 3
            entry.external_attr = (stat.S_IFREG | (0o755 if path.endswith(".sh") else 0o644)) << 16
            archive.writestr(entry, content, compress_type=zipfile.ZIP_DEFLATED)
    return buffer.getvalue()


def _headers(key: str | None) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}"} if key is not None else {}


@dataclass(frozen=True, slots=True)
class HubClient:
    client: httpx.AsyncClient
    plugins: MagicMock
    archives: MagicMock
    downloads: MagicMock
    anonymous_counter: AsyncMock
    upstream: MagicMock
    clock: MagicMock
    database: MagicMock


@pytest.fixture
async def hub(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[HubClient]:
    prisma: Final = MagicMock()
    prisma.writer_db = prisma.db
    prisma.db.execute_raw = AsyncMock(return_value=1)
    prisma.get_data = AsyncMock(return_value=None)
    prisma.db.litellm_skillstable.find_many = AsyncMock(return_value=[])
    prisma.db.litellm_usertable.find_unique = AsyncMock(return_value=None)
    plugins: Final = prisma.db.litellm_claudecodeplugintable
    archives: Final = prisma.db.litellm_claudecodepluginupload
    async def create_row(data: Mapping[str, object]) -> prisma_models.LiteLLM_ClaudeCodePluginTable:
        return prisma_models.LiteLLM_ClaudeCodePluginTable.model_validate({"id": "hub-id", "enabled": True, "anonymous_download_count": 0, **data})

    plugins.create = AsyncMock(side_effect=create_row)
    downloads: Final = prisma.db.litellm_claudecodeplugindownload
    downloads.create = AsyncMock()
    plugins.find_unique = AsyncMock(return_value=None)
    plugins.find_many = AsyncMock(return_value=[])

    async def update_row(where: Mapping[str, str], data: Mapping[str, object]) -> prisma_models.LiteLLM_ClaudeCodePluginTable:
        current: Final = plugins.find_unique.return_value.model_copy(update=dict(data))
        plugins.find_unique.return_value = current
        plugins.find_many.return_value = [current]
        return current

    plugins.update = AsyncMock(side_effect=update_row)
    plugins.delete = AsyncMock()
    archives.find_first = AsyncMock(return_value=None)
    archives.find_unique = AsyncMock(return_value=None)
    archives.create = AsyncMock()
    archives.upsert = AsyncMock()
    archives.delete_many = AsyncMock()
    prisma.db.tx.return_value.__aenter__.return_value = prisma.db
    submissions: Final = SubmissionRows()
    prisma.db.litellm_skillsubmission = submissions
    prisma.db.query_raw = AsyncMock(return_value=[])
    prisma.db.litellm_claudecodepluginversion.create = AsyncMock()
    prisma.db.litellm_claudecodepluginversion.count = AsyncMock(return_value=0)
    prisma.db.litellm_claudecodepluginversion.find_first = AsyncMock(return_value=None)
    upstream: Final = MagicMock(return_value=httpx.Response(404))
    clock: Final = MagicMock(return_value=datetime.now())
    limiter: Final = PROXY_MaxParallelRequestsHandler_v3(InternalUsageCache(DualCache()), time_provider=clock)
    monkeypatch.setattr(proxy_server.proxy_logging_obj, "proxy_hook_mapping", {"parallel_request_limiter": limiter})
    monkeypatch.setattr(proxy_server, "prisma_client", prisma)
    monkeypatch.setattr(proxy_server, "master_key", MASTER_KEY)
    monkeypatch.setattr(proxy_server, "general_settings", {})
    monkeypatch.setattr(proxy_server, "user_api_key_cache", DualCache())
    monkeypatch.setattr(litellm, "public_skills_index", True)
    monkeypatch.setenv("PROXY_BASE_URL", "http://test")
    monkeypatch.delenv("SERVER_ROOT_PATH", raising=False)
    for key, skills in ((GRANTED_KEY, [NAME]), (PLAIN_KEY, [])):
        proxy_server.user_api_key_cache.set_cache(
            key=proxy_server.hash_token(key),
            value=UserAPIKeyAuth(
                token=proxy_server.hash_token(key),
                user_role=proxy_server.LitellmUserRoles.INTERNAL_USER,
                object_permission=LiteLLM_ObjectPermissionTable(object_permission_id=key, skills=skills),
            ),
        )
    for key, role in (
        (MEMBER_KEY, proxy_server.LitellmUserRoles.INTERNAL_USER),
        (OTHER_KEY, proxy_server.LitellmUserRoles.INTERNAL_USER),
        (VIEWER_KEY, proxy_server.LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY),
    ):
        proxy_server.user_api_key_cache.set_cache(
            key=proxy_server.hash_token(key),
            value=UserAPIKeyAuth(token=proxy_server.hash_token(key), user_role=role, user_id=key),
        )
    async with (
        httpx.AsyncClient(transport=httpx.MockTransport(upstream), trust_env=False) as source_client,
        httpx.AsyncClient(transport=httpx.ASGITransport(app=proxy_server.app), base_url="http://test") as client,
    ):
        monkeypatch.setattr(proxy_server.app, "dependency_overrides", {
            get_plugin_upload_repository: lambda: PluginUploadRepository(prisma),
            get_skill_submission_repository: lambda: SkillSubmissionRepository(prisma),
            import_http_client: lambda: source_client,
        })
        yield HubClient(client, plugins, archives, downloads, prisma.db.execute_raw, upstream, clock, prisma.db)


async def _upload(
    hub: HubClient, content: bytes, key: str | None = MASTER_KEY, version: str | None = "2.0.0",
) -> httpx.Response:
    return await hub.client.post(
        f"{HUB}/upload",
        headers=_headers(key),
        data={"version": version} if version is not None else {},
        files={"file": ("notes.zip", content, "application/zip")},
    )


def _stored_upload(hub: HubClient, enabled: bool = True) -> prisma_models.LiteLLM_ClaudeCodePluginTable:
    data: Final = hub.plugins.create.await_args.kwargs["data"]
    record: Final = prisma_models.LiteLLM_ClaudeCodePluginTable(
        id="hub-id", name=data["name"], version=data["version"], description=data["description"], anonymous_download_count=0,
        manifest_json=data["manifest_json"], enabled=enabled, updated_at=data["updated_at"],
    )
    hub.plugins.find_unique.return_value = record
    hub.plugins.find_many.return_value = [record]
    hub.archives.find_first.return_value = prisma_models.LiteLLM_ClaudeCodePluginUpload(
        plugin_name=record.name, content=hub.archives.create.await_args.kwargs["data"]["content"],
    )
    hub.archives.find_unique.return_value = hub.archives.find_first.return_value
    return record


def _assert_portable(response: httpx.Response) -> None:
    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "application/zip"
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        assert {name: archive.read(name) for name in archive.namelist()} == FILES
        assert stat.S_IMODE(archive.getinfo("scripts/check.sh").external_attr >> 16) == 0o755


@pytest.mark.parametrize("route", (f"{HUB}?limit=1", "/claude-code/plugins"))
@pytest.mark.parametrize("key", (MASTER_KEY, PLAIN_KEY))
async def test_catalog_counts_are_batched_for_the_page(
    hub: HubClient, key: str, route: str,
) -> None:
    assert (await _upload(hub, _zip(FILES))).status_code == 200
    record: Final = _stored_upload(hub)
    hub.plugins.find_many.return_value = [record, record.model_copy(update={"id": "other", "name": "zzz-other"})]
    hub.database.query_raw.reset_mock()
    hub.database.query_raw.return_value = [{"id": record.id, "download_count": 7}]
    response: Final = await hub.client.get(route, headers=_headers(key))
    assert response.status_code == 200, response.text
    items: Final = response.json()["plugins" if route == "/claude-code/plugins" else "skills"]
    expected: Final = {record.id: 7, "other": 0} if route == "/claude-code/plugins" else {record.id: 7}
    assert {item["id"]: item["download_count"] for item in items} == expected
    assert hub.database.query_raw.await_count == 1
    assert set(hub.database.query_raw.await_args.args[1]) == set(expected)
    assert (await hub.client.get(f"{HUB}?offset=2", headers=_headers(key))).json()["skills"] == []
    assert hub.database.query_raw.await_count == 1


@pytest.mark.parametrize("operation", ("upload", "import"))
@pytest.mark.parametrize("public_base, root_path", (
    ("http://test", ""), ("https://public.example", ""), ("https://public.example", "/gateway"),
))
async def test_publication_round_trips_both_formats_without_inflating_on_the_request_thread(
    hub: HubClient, monkeypatch: pytest.MonkeyPatch, operation: str, public_base: str, root_path: str,
) -> None:
    monkeypatch.delenv("PROXY_BASE_URL")
    hub.upstream.return_value = httpx.Response(200, content=_zip(FILES, "repo-revision/"))
    response: Final = (
        await _upload(hub, _zip(FILES)) if operation == "upload" else await hub.client.post(
            f"{HUB}/import", headers=_headers(MASTER_KEY),
            json={"url": f"https://skills.sh/synthetic-author/notes-library/{NAME}"},
        )
    )
    assert response.status_code == 200, response.text
    assert response.json()["version"] == ("2.0.0" if operation == "upload" else "1.0.0")
    record: Final = _stored_upload(hub)
    manifest: Final = json.loads(record.manifest_json)
    monkeypatch.setattr(proxy_server.app, "root_path", root_path)
    monkeypatch.setenv("SERVER_ROOT_PATH", root_path)
    serving_base: Final = public_base + root_path
    hub.client.base_url = httpx.URL(serving_base)
    details: Final = await hub.client.get(f"{HUB}/{NAME}", headers=_headers(PLAIN_KEY))
    listed: Final = await hub.client.get(HUB, headers=_headers(PLAIN_KEY))
    assert listed.json()["skills"] == [details.json()]
    assert "has_agent_skill_metadata" not in listed.text + details.text
    assert details.json()["download_url"] == f"{serving_base}{HUB}/{NAME}/archive"
    assert details.json()["content_url"] == f"{serving_base}{HUB}/{NAME}/content"
    catalog: Final = await hub.client.get("/claude-code/plugins", headers=_headers(PLAIN_KEY))
    assert catalog.status_code == 200 and "has_agent_skill_metadata" not in catalog.text
    hub.downloads.create.assert_not_awaited()
    hub.anonymous_counter.assert_not_awaited()
    request_thread: Final = threading.get_ident()
    decompress: Final = zlib.decompressobj

    def off_thread_decompressor(wbits: int) -> zlib.Decompress:
        assert threading.get_ident() != request_thread
        return decompress(wbits)

    monkeypatch.setattr(zlib, "decompressobj", off_thread_decompressor)
    portable: Final = await hub.client.get(details.json()["download_url"])
    content: Final = await hub.client.get(details.json()["content_url"])
    assert content.status_code == 200 and content.content == MANIFEST
    digest: Final = "sha256:" + hashlib.sha256(portable.content).hexdigest()
    assert manifest["agent_skill"]["digest"] == digest
    entry: Final = (await hub.client.get(INDEX)).json()["skills"][0]
    assert entry["digest"] == digest
    assert entry["url"].startswith(f"{serving_base}{HUB}/{NAME}/archive?")
    assert (await hub.client.get(entry["url"])).content == portable.content
    assert (await hub.client.get(f"{HUB}/{NAME}/archive?digest=sha256:stale")).status_code == 404
    legacy_path: Final = httpx.URL(manifest["source"]["url"]).path
    legacy: Final = await hub.client.get(legacy_path, headers=_headers(GRANTED_KEY))
    assert legacy.status_code == 200
    assert hashlib.sha256(legacy.content).hexdigest() == manifest["source"]["sha256"]
    assert (await hub.client.get(legacy_path.replace(manifest["source"]["sha256"], "stale"))).status_code == 404
    assert hub.anonymous_counter.await_count == 2
    assert hub.anonymous_counter.await_args.args[1:] == (record.id,)
    hub.downloads.create.assert_awaited_once()
    monkeypatch.setattr(zlib, "decompressobj", decompress)
    _assert_portable(portable)
    with zipfile.ZipFile(io.BytesIO(legacy.content)) as archive:
        assert {path: archive.read(path) for path in archive.namelist()} == {
            ".claude-plugin/plugin.json": json.dumps({"name": NAME}).encode(),
            **{f"skills/{NAME}/{path}": content for path, content in FILES.items()},
        }


@pytest.mark.parametrize("operation", ("upload", "import"))
@pytest.mark.parametrize("boundary", ("entries", "bytes", "file"))
@pytest.mark.parametrize("fits", (True, False))
async def test_publication_reserves_wrapper_capacity_before_persisting(
    hub: HubClient, operation: str, boundary: str, fits: bool,
) -> None:
    file_limit: Final = 10 * 1024 * 1024
    padding_size: Final = (
        MAX_ARCHIVE_UNPACKED_BYTES - len(MANIFEST) - len(json.dumps({"name": NAME}).encode()) * int(fits)
        if boundary == "bytes" else file_limit + int(not fits)
    )
    files: Final = (
        {"SKILL.md": MANIFEST, **{
            f"references/{index}.txt": b"x" for index in range(MAX_ARCHIVE_ENTRIES - 1 - int(fits))
        }} if boundary == "entries" else {
            "SKILL.md": MANIFEST,
            **({
                f"assets/{index}.bin": b"x" * min(file_limit, padding_size - index * file_limit)
                for index in range(5)
            } if boundary == "bytes" else {"assets/padding.bin": b"x" * padding_size}),
        }
    )
    content: Final = _zip(files, prefix="repo-revision/" if operation == "import" else "")
    hub.upstream.return_value = httpx.Response(200, content=content)
    response: Final = (
        await _upload(hub, content) if operation == "upload" else await hub.client.post(
            f"{HUB}/import", headers=_headers(MASTER_KEY),
            json={"url": "https://github.com/synthetic-author/notes-library"},
        )
    )
    assert response.status_code == (200 if fits else 422), response.text
    if not fits:
        hub.plugins.create.assert_not_awaited()
        return
    _stored_upload(hub)
    downloaded: Final = await hub.client.get(response.json()["download_url"])
    assert downloaded.status_code == 200, downloaded.text
    with zipfile.ZipFile(io.BytesIO(downloaded.content)) as archive:
        actual: Final = {
            path: (archive.getinfo(path).file_size, hashlib.sha256(archive.read(path)).hexdigest())
            for path in archive.namelist()
        }
    expected: Final = {path: (len(content), hashlib.sha256(content).hexdigest()) for path, content in files.items()}
    assert actual == expected


@pytest.mark.parametrize("enabled, public_index, auth_required, key, status", (
    (True, True, False, None, 200), (False, True, False, None, 404),
    (True, False, False, None, 401), (True, True, True, None, 401),
    (False, True, False, GRANTED_KEY, 200), (False, True, False, PLAIN_KEY, 404),
    (True, True, False, "", 401), (True, True, False, "sk-invalid", 401),
))
async def test_download_policy_requires_publication_or_a_valid_grant(
    hub: HubClient, monkeypatch: pytest.MonkeyPatch, enabled: bool,
    public_index: bool, auth_required: bool, key: str | None, status: int,
) -> None:
    assert (await _upload(hub, _zip(FILES))).status_code == 200
    _stored_upload(hub, enabled=enabled)
    monkeypatch.setattr(litellm, "public_skills_index", public_index)
    monkeypatch.setattr(proxy_server, "general_settings", {"claude_code_marketplace_auth_required": auth_required})
    for suffix in ("archive", "content"):
        response: Final = await hub.client.get(f"{HUB}/{NAME}/{suffix}", headers=_headers(key))
        assert response.status_code == status, response.text
    assert hub.downloads.create.await_count == int(status == 200 and key is not None)
    assert hub.anonymous_counter.await_count == int(status == 200 and key is None)
    if status != 200:
        hub.archives.find_first.assert_not_awaited()
    index: Final = await hub.client.get(INDEX)
    assert index.status_code == (200 if public_index else 404)
    if public_index:
        assert [entry["name"] for entry in index.json()["skills"]] == ([NAME] if enabled and not auth_required else [])


@pytest.mark.parametrize("key", (None, GRANTED_KEY))
async def test_deleted_skill_is_not_served_when_download_cannot_be_recorded(hub: HubClient, key: str | None) -> None:
    assert (await _upload(hub, _zip(FILES))).status_code == 200
    record: Final = _stored_upload(hub)
    legacy_url: Final = json.loads(record.manifest_json)["source"]["url"]
    hub.anonymous_counter.return_value = 0
    hub.downloads.create.side_effect = ForeignKeyViolationError(data={})
    for path in (f"{HUB}/{NAME}/archive", legacy_url):
        response: Final = await hub.client.get(path, headers=_headers(key))
        assert response.status_code == 404, response.text
        assert "Skill archive not found" in response.text
    assert hub.anonymous_counter.await_count == (2 if key is None else 0)
    assert hub.downloads.create.await_count == (0 if key is None else 2)


async def test_search_paginates_visible_matches(hub: HubClient) -> None:
    assert (await _upload(hub, _zip(FILES))).status_code == 200
    record: Final = _stored_upload(hub)
    hub.plugins.find_many.return_value = [
        record.model_copy(update={"name": "private-notes", "enabled": False}), record,
        record.model_copy(update={"name": "other-notes"}),
        record.model_copy(update={"name": "unmatched", "description": "Different topic"}),
    ]
    for offset, name in enumerate(("other-notes", NAME)):
        response: Final = await hub.client.get(
            HUB, headers=_headers(PLAIN_KEY), params={"query": "SYNTHETIC", "limit": 1, "offset": offset},
        )
        assert response.status_code == 200
        assert [item["name"] for item in response.json()["skills"]] == [name]
        assert response.json()["count"] == 2
        assert response.json()["next_offset"] == (1 if offset == 0 else None)


async def test_metadata_publication_and_deletion_preserve_files_and_grants(hub: HubClient) -> None:
    assert (await _upload(hub, _zip(FILES))).status_code == 200
    original: Final = _stored_upload(hub)
    manifest: Final = json.loads(original.manifest_json)

    for route, metadata in (
        (f"/claude-code/plugins/{NAME}", {"source": manifest["source"], "description": "Legacy edit"}),
        (f"{HUB}/{NAME}", {"description": "Neutral edit", "version": "3.0.0", "keywords": ["notes"]}),
        (f"{HUB}/{NAME}", {}),
    ):
        response: Final = await hub.client.put(route, headers=_headers(MASTER_KEY), json=metadata)
        assert response.status_code == 200, response.text
        assert json.loads(hub.plugins.find_unique.return_value.manifest_json)["agent_skill"] == manifest["agent_skill"]
        details: Final = await hub.client.get(f"{HUB}/{NAME}", headers=_headers(PLAIN_KEY))
        assert details.json()["description"] == metadata.get("description")
        _assert_portable(await hub.client.get(details.json()["download_url"]))
    for action, public in (("unpublish", False), ("publish", True)):
        changed: Final = await hub.client.post(f"{HUB}/{NAME}/{action}", headers=_headers(MASTER_KEY))
        assert changed.status_code == 200
        assert changed.json()["visibility"] == ("public" if public else "private")
        assert (await hub.client.get(f"{HUB}/{NAME}/archive")).status_code == (200 if public else 404)
        _assert_portable(await hub.client.get(f"{HUB}/{NAME}/archive", headers=_headers(GRANTED_KEY)))
        assert (await hub.client.get(f"{HUB}/{NAME}", headers=_headers(PLAIN_KEY))).status_code == (200 if public else 404)
        assert bool((await hub.client.get(INDEX)).json()["skills"]) == public
    assert (await hub.client.delete(f"{HUB}/{NAME}", headers=_headers(MASTER_KEY))).status_code == 204
    hub.plugins.delete.assert_awaited_once_with(where={"name": NAME})


@pytest.mark.parametrize("method, suffix", (("PUT", ""), ("POST", "/publish"), ("POST", "/unpublish"), ("DELETE", "")))
async def test_granted_readers_cannot_manage_skills(hub: HubClient, method: str, suffix: str) -> None:
    response: Final = await hub.client.request(
        method, f"{HUB}/{NAME}{suffix}", headers=_headers(GRANTED_KEY), json={} if method == "PUT" else None,
    )
    assert response.status_code == 403
    hub.plugins.update.assert_not_awaited()
    hub.plugins.delete.assert_not_awaited()


@pytest.mark.parametrize("content, key, status", (
    (b"not a zip", MASTER_KEY, 422),
    (_zip(FILES), PLAIN_KEY, 403), (_zip(FILES), None, 401),
), ids=("invalid-zip", "forbidden", "unauthenticated"))
async def test_rejected_uploads_do_not_write(hub: HubClient, content: bytes, key: str | None, status: int) -> None:
    assert (await _upload(hub, content, key)).status_code == status
    hub.plugins.create.assert_not_awaited()


async def test_stored_content_survives_duplicate_upload_and_is_hidden_after_source_replacement(hub: HubClient) -> None:
    assert (await _upload(hub, _zip(FILES))).status_code == 200
    original: Final = _stored_upload(hub)
    hub.plugins.create.side_effect = UniqueViolationError({"message": "duplicate", "error_code": "P2002"})
    response: Final = await _upload(hub, _zip({**FILES, "references/guide.md": b"replacement"}))
    assert response.status_code == 409
    _assert_portable(await hub.client.get(f"{HUB}/{NAME}/archive"))
    hub.plugins.update.assert_not_awaited()
    hub.archives.find_first.reset_mock()
    manifest: Final = json.loads(original.manifest_json)
    external: Final = {
        **manifest["source"],
        "url": str(httpx.URL(manifest["source"]["url"]).copy_with(scheme="https", host="external.example")),
    }
    replaced: Final = await hub.client.put(
        f"/claude-code/plugins/{NAME}", headers=_headers(MASTER_KEY),
        json={"source": external, "agent_skill": manifest["agent_skill"]},
    )
    assert replaced.status_code == 200, replaced.text
    assert "agent_skill" not in json.loads(hub.plugins.find_unique.return_value.manifest_json)
    details: Final = await hub.client.get(f"{HUB}/{NAME}", headers=_headers(MASTER_KEY))
    assert details.json()["source"] == external and details.json()["download_url"] is None
    for suffix in ("archive", "content"):
        assert (await hub.client.get(f"{HUB}/{NAME}/{suffix}")).status_code == 404
    assert (await hub.client.get(httpx.URL(manifest["source"]["url"]).path)).status_code == 404
    hub.archives.find_first.assert_not_awaited()
    assert (await hub.client.get(INDEX)).json()["skills"] == []


@pytest.mark.parametrize("user_id, role, admitted", (
    ("publisher", proxy_server.LitellmUserRoles.PROXY_ADMIN, 10),
    (None, proxy_server.LitellmUserRoles.PROXY_ADMIN, 20),
    ("publisher", proxy_server.LitellmUserRoles.INTERNAL_USER, 10),
), ids=("admin-user", "admin-key", "member-user"))
async def test_publication_limit_is_shared_by_user_and_route_and_resets(
    hub: HubClient, monkeypatch: pytest.MonkeyPatch, user_id: str | None, role: str, admitted: int,
) -> None:
    for key in (PLAIN_KEY, GRANTED_KEY):
        proxy_server.user_api_key_cache.set_cache(
            key=proxy_server.hash_token(key),
            value=UserAPIKeyAuth(
                token=proxy_server.hash_token(key), user_id=user_id,
                user_role=role,
            ),
        )
    async def attempt(index: int) -> httpx.Response:
        key: Final = PLAIN_KEY if index % 2 else GRANTED_KEY
        if index % 3 == 0:
            return await _upload(hub, b"invalid", key)
        if index % 3 == 1:
            return await hub.client.post(
                f"{HUB}/import", headers=_headers(key), json={"url": "https://example.com/skill"},
            )
        return await hub.client.put(
            f"{HUB}/{NAME}/upload", headers=_headers(key), files={"file": ("invalid.zip", b"invalid")},
        )

    responses: Final = await asyncio.gather(*(attempt(index) for index in range(20)))
    assert sum(response.status_code in (400, 404, 422) for response in responses) == admitted
    assert sum(response.status_code == 429 for response in responses) == 20 - admitted
    assert all(response.headers["retry-after"] == "60" for response in responses if response.status_code == 429)
    hub.plugins.create.assert_not_awaited()
    hub.upstream.assert_not_called()
    assert (await _upload(hub, _zip(FILES))).status_code == 200
    _stored_upload(hub)
    for _ in range(11):
        assert (await hub.client.get(INDEX)).status_code == 200
        assert (await hub.client.get(f"{HUB}/{NAME}/archive", headers=_headers(PLAIN_KEY))).status_code == 200
    hub.clock.return_value += timedelta(seconds=60)
    assert (await _upload(hub, b"invalid", GRANTED_KEY)).status_code == 422
    monkeypatch.setattr(proxy_server.proxy_logging_obj, "proxy_hook_mapping", {})
    assert (await _upload(hub, b"invalid")).status_code == 503


@pytest.mark.asyncio
async def test_neutral_metadata_publication_pairs_version_and_preserved_skill_snapshot(
    hub: HubClient, monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: Final = await _upload(hub, _zip(FILES), version=None)
    assert created.status_code == 200, created.text
    assert created.json()["version"] == "1.0.0"
    original: Final = _stored_upload(hub)
    hub.plugins.find_unique.return_value = original.model_copy(update={
        "manifest_json": json.dumps({**json.loads(original.manifest_json), "author": "legacy author"}),
    })
    assert (await hub.client.get(f"{HUB}/{NAME}", headers=_headers(MASTER_KEY))).status_code == 200
    monkeypatch.delenv("PROXY_BASE_URL")
    hub.client.base_url = httpx.URL("https://public.example")
    first: Final = hub.database.litellm_claudecodepluginversion.create.await_args.kwargs["data"]
    hub.database.litellm_claudecodepluginversion.count.return_value = 1

    async def replace_row(
        where: Mapping[str, str], data: Mapping[str, object],
    ) -> prisma_models.LiteLLM_ClaudeCodePluginTable:
        current: Final = original.model_copy(update=dict(data))
        hub.plugins.find_unique.return_value = current
        return current

    hub.plugins.update.side_effect = replace_row
    updated: Final = await hub.client.put(
        f"{HUB}/{NAME}", headers=_headers(MASTER_KEY),
        json={"description": "Registry description", "domain": "notes", "namespace": "team", "keywords": ["notes"]},
    )
    assert updated.status_code == 200, updated.text
    assert updated.json()["version"] == "1.0.1"
    assert updated.json()["download_url"] == f"https://public.example{HUB}/{NAME}/archive"
    latest: Final = hub.database.litellm_claudecodepluginversion.create.await_args.kwargs["data"]
    manifest: Final = json.loads(latest["manifest_json"])
    assert latest["version"] == manifest["version"] == updated.json()["version"]
    assert latest["manifest_json"] == hub.plugins.find_unique.return_value.manifest_json
    assert manifest["agent_skill"] == json.loads(original.manifest_json)["agent_skill"]
    assert (manifest["domain"], manifest["namespace"], manifest["keywords"]) == ("notes", "team", ["notes"])
    assert first["version"] == "1.0.0"
    assert latest["content"].decode() == first["content"].decode()
    _assert_portable(await hub.client.get(updated.json()["download_url"]))


@pytest.mark.parametrize("key,status", ((MASTER_KEY, 200), (GRANTED_KEY, 200), (PLAIN_KEY, 404)))
@pytest.mark.asyncio
async def test_neutral_history_list_detail_and_portable_download_share_current_visibility(
    hub: HubClient, key: str, status: int,
) -> None:
    assert (await _upload(hub, _zip(FILES), version=None)).status_code == 200
    original: Final = _stored_upload(hub, enabled=False)
    data: Final = hub.database.litellm_claudecodepluginversion.create.await_args.kwargs["data"]
    version: Final = prisma_models.LiteLLM_ClaudeCodePluginVersion.model_validate({"id": "retained", **data})
    hub.plugins.find_unique.return_value = original.model_copy(update={"version": "3.0.0"})
    hub.database.litellm_claudecodepluginversion.find_first.return_value = version
    hub.database.query_raw.return_value = [{
        "id": version.id, "version": version.version, "description": version.description,
        "manifest": json.loads(version.manifest_json), "published_at": version.published_at,
        "published_by": version.published_by, "archive_available": version.archive_available,
    }]
    listed: Final = await hub.client.get(f"{HUB}/{NAME}/versions", headers=_headers(key))
    detail: Final = await hub.client.get(f"{HUB}/{NAME}/versions/retained", headers=_headers(key))
    downloaded: Final = await hub.client.get(f"{HUB}/{NAME}/versions/retained/archive", headers=_headers(key))
    assert listed.status_code == detail.status_code == downloaded.status_code == status
    assert hub.downloads.create.await_count == int(status == 200)
    if status == 200:
        event: Final = hub.downloads.create.await_args.kwargs["data"]
        assert event["version"] == version.version != "3.0.0"
        assert event["digest"] == hashlib.sha256(downloaded.content).hexdigest()
        assert listed.json() == {"versions": [detail.json()], "next_offset": None}
        assert detail.json()["manifest"]["agent_skill"] == json.loads(original.manifest_json)["agent_skill"]
        assert detail.json()["download_url"] == f"http://test{HUB}/{NAME}/versions/retained/archive"
        _assert_portable(downloaded)
        assert downloaded.headers["etag"] == f'"sha256:{hashlib.sha256(downloaded.content).hexdigest()}"'
        hub.database.query_raw.return_value = []
        assert (await hub.client.get(f"{HUB}/{NAME}/versions/missing", headers=_headers(key))).status_code == 404
        hub.database.litellm_claudecodepluginversion.find_first.return_value = None
        assert (await hub.client.get(
            f"{HUB}/{NAME}/versions/missing/archive", headers=_headers(key),
        )).status_code == 404


@pytest.mark.asyncio
async def test_neutral_history_labels_external_sources_and_bounds_pagination(hub: HubClient) -> None:
    assert (await _upload(hub, _zip(FILES))).status_code == 200
    _stored_upload(hub)
    row: Final = {
        "id": "reference", "version": "3.0.0", "description": None,
        "manifest": {"source": {"source": "github", "repo": "example/skill"}},
        "published_at": None, "published_by": None, "archive_available": False,
    }
    hub.database.query_raw.return_value = [row, {**row, "id": "older"}]
    listed: Final = await hub.client.get(f"{HUB}/{NAME}/versions?limit=1", headers=_headers(MASTER_KEY))
    assert listed.status_code == 200, listed.text
    assert listed.json()["next_offset"] == 1
    assert len(listed.json()["versions"]) == 1
    assert listed.json()["versions"][0]["content_kind"] == "source_reference"
    assert listed.json()["versions"][0]["download_url"] is None
    assert (await hub.client.get(
        f"{HUB}/{NAME}/versions?limit=101", headers=_headers(MASTER_KEY),
    )).status_code == 422


@pytest.fixture
async def pending_hub(hub: HubClient) -> HubClient:
    hub.database.query_raw.side_effect = hub.database.litellm_skillsubmission.query_raw
    return hub


@pytest.mark.parametrize("operation,approve", (("upload", True), ("import", True), ("upload", False)),
                         ids=("approve-upload", "approve-import", "reject-upload"))
async def test_member_archive_stays_private_until_admin_review(
    pending_hub: HubClient, operation: str, approve: bool,
) -> None:
    hub: Final = pending_hub
    hub.upstream.return_value = httpx.Response(200, content=_zip(FILES, "repository/"))
    pending: Final = (
        await _upload(hub, _zip(FILES), MEMBER_KEY, version=None) if operation == "upload"
        else await hub.client.post(
            f"{HUB}/import", headers=_headers(MEMBER_KEY),
            json={"url": f"https://skills.sh/synthetic-author/notes-library/{NAME}"},
        )
    )
    assert pending.status_code == 200, pending.text
    receipt: Final = pending.json()
    assert receipt["status"] == "pending_review" and receipt["proposal"]["version"] is None
    assert receipt["proposal"]["agent_skill"]["name"] == NAME
    hub.plugins.create.assert_not_awaited()
    hub.database.litellm_claudecodepluginversion.create.assert_not_awaited()
    path: Final = f"/skill-hub/submissions/{receipt['submission_id']}"
    assert receipt["proposal"]["source"]["url"] == f"http://test{path}/archive"
    own: Final = await hub.client.get(path, headers=_headers(MEMBER_KEY))
    assert own.json()["current_skill"] is None
    assert own.json()["submission"]["proposal"] == receipt["proposal"]
    for suffix in ("", "/archive"):
        assert (await hub.client.get(path + suffix, headers=_headers(OTHER_KEY))).status_code == 404
    _assert_portable(await hub.client.get(path + "/archive", headers=_headers(MEMBER_KEY)))
    assert (await hub.client.get(path, headers=_headers(VIEWER_KEY))).status_code == 200
    denied: Final = await hub.client.put(path + "/approve", headers=_headers(VIEWER_KEY))
    assert denied.status_code == 403, denied.text
    assert (await hub.client.get(f"{HUB}/{NAME}/archive")).status_code == 404
    if not approve:
        rejected: Final = await hub.client.put(
            path + "/reject", headers=_headers(MASTER_KEY), json={"reason": "Needs revision"},
        )
        assert rejected.status_code == 200, rejected.text
        assert rejected.json()["skill"] is None and rejected.json()["action"] == "rejected"
        assert rejected.json()["submission"]["rejection_reason"] == "Needs revision"
        assert rejected.json()["submission"]["proposal"] == receipt["proposal"]
        hub.plugins.create.assert_not_awaited()
        return
    approved: Final = await hub.client.put(path + "/approve", headers=_headers(MASTER_KEY))
    assert approved.status_code == 200, approved.text
    assert approved.json()["action"] == "approved"
    assert approved.json()["skill"]["version"] == "1.0.0"
    assert approved.json()["submission"]["proposal"] == receipt["proposal"]
    _stored_upload(hub)
    _assert_portable(await hub.client.get(approved.json()["skill"]["download_url"]))
    assert (await hub.client.put(path + "/reject", headers=_headers(MASTER_KEY))).status_code == 409
    listed: Final = await hub.client.get("/skill-hub/submissions?status=approved", headers=_headers(MEMBER_KEY))
    assert (listed.json()["count"], listed.json()["has_more"]) == (1, False)


@pytest.mark.parametrize("replacement", (False, True), ids=("metadata", "archive"))
async def test_member_updates_preserve_manifest_and_original_editor_base(pending_hub: HubClient, replacement: bool) -> None:
    hub: Final = pending_hub
    assert (await _upload(hub, _zip(FILES))).status_code == 200
    original: Final = _stored_upload(hub)
    manifest: Final = {
        **json.loads(original.manifest_json), "domain": "Tools", "namespace": "research",
        "keywords": ["notes"], "extension": {"retained": True}, "author": "legacy author",
    }
    hub.plugins.find_unique.return_value = original.model_copy(update={"manifest_json": json.dumps(manifest)})
    before: Final = hub.database.litellm_claudecodepluginversion.create.await_count
    updated_files: Final = {**FILES, "references/guide.md": b"Updated supporting instructions.\n"}
    expected: Final = original.updated_at.isoformat()
    response: Final = (
        await hub.client.put(
            f"{HUB}/{NAME}/upload", headers=_headers(MEMBER_KEY),
            data={"expected_updated_at": expected}, files={"file": ("new.zip", _zip(updated_files), "application/zip")},
        ) if replacement else await hub.client.put(
            f"{HUB}/{NAME}", headers=_headers(MEMBER_KEY),
            json={"description": "Reviewed description", "expected_updated_at": expected,
                  "domain": "Tools", "namespace": "research", "keywords": ["notes"]},
        )
    )
    assert response.status_code == 200, response.text
    path: Final = f"/skill-hub/submissions/{response.json()['submission_id']}"
    detail: Final = (await hub.client.get(path, headers=_headers(MEMBER_KEY))).json()
    assert detail["submission"]["base_updated_at"] == original.updated_at.isoformat().replace("+00:00", "Z")
    assert detail["submission"]["proposal"]["domain"] == "Tools"
    assert detail["submission"]["proposal"]["keywords"] == ["notes"]
    assert detail["current_skill"]["namespace"] == "research"
    for version in ("1.0.0", original.version):
        invalid: Final = await hub.client.put(
            path + ("/upload" if replacement else ""), headers=_headers(MEMBER_KEY),
            **({"data": {"expected_revision": "0", "version": version},
                "files": {"file": ("new.zip", _zip(updated_files), "application/zip")}}
               if replacement else {"json": {"expected_revision": 0, "version": version}}),
        )
        assert invalid.status_code == 400, invalid.text
        assert (await hub.client.get(path, headers=_headers(MEMBER_KEY))).json() == detail
    assert hub.database.litellm_claudecodepluginversion.create.await_count == before
    hub.plugins.update.assert_not_awaited()
    approved: Final = await hub.client.put(path + "/approve", headers=_headers(MASTER_KEY))
    assert approved.status_code == 200, approved.text
    stored: Final = json.loads(hub.plugins.find_unique.return_value.manifest_json)
    assert (stored["extension"], stored["author"]) == (manifest["extension"], manifest["author"])
    assert stored["agent_skill"]["digest"] == response.json()["proposal"]["agent_skill"]["digest"]
    assert approved.json()["skill"]["version"] == "2.0.1"
    assert approved.json()["submission"]["proposal"]["version"] is None
    wrapper: Final = hub.archives.upsert.await_args.kwargs["data"]["update"]["content"].decode()
    with zipfile.ZipFile(io.BytesIO(wrapper)) as archive:
        assert archive.read(f"skills/{NAME}/references/guide.md") == (
            updated_files["references/guide.md"] if replacement else FILES["references/guide.md"]
        )
    if not replacement:
        assert stored["agent_skill"] == manifest["agent_skill"]


@pytest.mark.parametrize("failure,status", (("stale", 409), ("wrong-name", 422), ("viewer", 403)),
                         ids=("stale", "wrong-name", "viewer"))
async def test_replacement_rejections_do_not_stage_or_change_files(pending_hub: HubClient, failure: str, status: int) -> None:
    hub: Final = pending_hub
    assert (await _upload(hub, _zip(FILES))).status_code == 200
    _stored_upload(hub)
    response: Final = await hub.client.put(
        f"{HUB}/{NAME}/upload", headers=_headers(VIEWER_KEY if failure == "viewer" else MEMBER_KEY),
        data={"expected_updated_at": "2020-01-01T00:00:00Z"} if failure == "stale" else {},
        files={"file": ("new.zip", _zip({"SKILL.md": MANIFEST.replace(NAME.encode(), b"different-name")}), "application/zip")},
    )
    assert response.status_code == status, response.text
    assert hub.database.litellm_skillsubmission.rows == ()
    hub.plugins.update.assert_not_awaited()
    hub.archives.upsert.assert_not_awaited()


@pytest.mark.parametrize("key", (MEMBER_KEY, OTHER_KEY, VIEWER_KEY))
@pytest.mark.parametrize("method", ("GET", "POST"))
async def test_skill_scan_requires_full_admin(hub: HubClient, key: str, method: str) -> None:
    response = await hub.client.request(method, "/skill-hub/submissions/absent/scan", headers=_headers(key))
    assert response.status_code == 403, response.text


async def test_unconfigured_scanning_preserves_manual_approval(pending_hub: HubClient) -> None:
    hub = pending_hub
    created = await _upload(hub, _zip(FILES), MEMBER_KEY, version=None)
    path = f"/skill-hub/submissions/{created.json()['submission_id']}"
    state = await hub.client.get(path + "/scan", headers=_headers(MASTER_KEY))
    assert state.status_code == 200, state.text
    assert state.json() == {"configured": False, "scan": None}
    unavailable = await hub.client.post(path + "/scan", headers=_headers(MASTER_KEY))
    assert unavailable.status_code == 503, unavailable.text
    approved = await hub.client.put(path + "/approve", headers=_headers(MASTER_KEY))
    assert approved.status_code == 200, approved.text
    assert approved.json()["submission"]["status"] == "approved"


@pytest.mark.parametrize("scanner_status", (200, 503), ids=("report", "error"))
async def test_configured_scan_persists_result_without_changing_manual_approval(
    pending_hub: HubClient, monkeypatch: pytest.MonkeyPatch, scanner_status: int,
) -> None:
    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
    from litellm.proxy.discovery_endpoints import skill_scanning

    hub: Final = pending_hub
    report: Final = {
        "scanner_version": "test-version", "complete": True, "scan_mode": "static-only",
        "risk_score": 0, "findings": [], "limitations": [],
    }

    def respond(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer scanner-only-token"
        return httpx.Response(scanner_status, json=report)

    handler: Final = AsyncHTTPHandler(transport=httpx.MockTransport(respond))

    def scanner_client(provider: str, params: Mapping[str, object]) -> AsyncHTTPHandler:
        return handler

    monkeypatch.setattr(skill_scanning, "get_async_httpx_client", scanner_client)
    monkeypatch.setattr(proxy_server, "general_settings", {"skill_scanner": {
        "endpoint": "https://scanner.example/scan", "api_key_env": "SCANNER_TEST_KEY",
    }})
    monkeypatch.setenv("SCANNER_TEST_KEY", "scanner-only-token")
    created: Final = await _upload(hub, _zip(FILES), MEMBER_KEY, version=None)
    path: Final = f"/skill-hub/submissions/{created.json()['submission_id']}"
    before: Final = (await hub.client.get(path, headers=_headers(MASTER_KEY))).json()
    async with handler.client:
        started: Final = await hub.client.post(path + "/scan", headers=_headers(MASTER_KEY))
    assert started.status_code == 200, started.text
    completed: Final = (await hub.client.get(path + "/scan", headers=_headers(MASTER_KEY))).json()["scan"]
    assert completed["id"] == started.json()["scan"]["id"]
    expected: Final = ("completed", report) if scanner_status == 200 else ("failed", None)
    assert (completed["status"], completed["report"]) == expected
    assert completed["error"] is None if scanner_status == 200 else "HTTP 503" in completed["error"]
    monkeypatch.setattr(proxy_server, "general_settings", {"skill_scanner": {"endpoint": "invalid"}})
    retained: Final = await hub.client.get(path + "/scan", headers=_headers(MASTER_KEY))
    assert retained.status_code == 200 and retained.json() == {"configured": False, "scan": completed}
    unavailable: Final = await hub.client.post(path + "/scan", headers=_headers(MASTER_KEY))
    assert unavailable.status_code == 503 and "configuration is invalid" in unavailable.text
    assert (await hub.client.get(path, headers=_headers(MASTER_KEY))).json() == before
    approved: Final = await hub.client.put(path + "/approve", headers=_headers(MASTER_KEY))
    assert approved.status_code == 200 and approved.json()["submission"]["status"] == "approved"


async def test_catalog_filters_metadata_before_pagination_and_preserves_tag_facets(hub: HubClient) -> None:
    hub.plugins.find_many.return_value = [
        prisma_models.LiteLLM_ClaudeCodePluginTable(
            id=name, name=name, enabled=True, anonymous_download_count=0, created_by="author@example.test",
            created_at=datetime(2026, 1, 2), updated_at=datetime(2026, 2, 3),
            manifest_json=json.dumps({
                "source": {"source": "github", "repo": "example/skills"},
                "keywords": tags,
                "_litellm_published_by": "publisher@example.test",
                **({"_litellm_review": {"approved_by": "reviewer@example.test", "security_scan_passed": scanned}} if approved else {}),
            }),
        )
        for name, tags, approved, scanned in (
            ("a-other", ["support"], False, False),
            ("b-review", ["engineering"], True, False),
            ("c-review", ["engineering"], True, True),
        )
    ]
    response = await hub.client.get(HUB, headers=_headers(MASTER_KEY), params={
        "query": " REVIEWER ", "tag": "ENGINEERING", "limit": 1, "offset": 1,
    })
    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 2
    assert body["tags"] == ["engineering", "support"]
    assert [skill["name"] for skill in body["skills"]] == ["c-review"]
    skill = body["skills"][0]
    assert skill["created_by"] == "author@example.test"
    assert skill["published_by"] == "publisher@example.test"
    assert skill["approved_by"] == "reviewer@example.test"
    assert skill["created_at"].startswith("2026-01-02")
    assert skill["updated_at"].startswith("2026-02-03")
    assert skill["security_scan_passed"] is True
    scanned = await hub.client.get(HUB, headers=_headers(MASTER_KEY), params={"security_scan_passed": "true"})
    assert [skill["name"] for skill in scanned.json()["skills"]] == ["c-review"]
    unscanned = await hub.client.get(HUB, headers=_headers(MASTER_KEY), params={"security_scan_passed": "false"})
    assert [skill["name"] for skill in unscanned.json()["skills"]] == ["a-other", "b-review"]
    assert unscanned.json()["skills"][0]["approved_by"] is None


@pytest.mark.parametrize("has_issues", (False, True))
async def test_admin_approval_does_not_override_scan_result(pending_hub: HubClient, has_issues: bool) -> None:
    hub: Final = pending_hub
    pending: Final = await _upload(hub, _zip(FILES), MEMBER_KEY, version=None)
    receipt: Final = pending.json()
    submission: Final = hub.database.litellm_skillsubmission.rows[0]
    digest: Final = hashlib.sha256(submission.archive_content.decode()).hexdigest()
    await hub.database.litellm_skillsubmission.update(
        where={"id": submission.id}, data={"scan_json": json.dumps({
            "id": "scan", "requested_by": "admin", "started_at": "2026-01-01T00:00:00Z",
            "expires_at": "2026-01-01T00:03:00Z", "status": "completed", "archive_sha256": digest,
            "report": {"scanner_version": "test", "scan_mode": "static-only", "complete": True,
                       "risk_score": 0, "findings": [{"rule_id": "rule", "severity": "LOW",
                       "message": "Issue", "file": "SKILL.md"}] if has_issues else []},
        })},
    )
    approved: Final = await hub.client.put(
        f"/skill-hub/submissions/{receipt['submission_id']}/approve", headers=_headers(MASTER_KEY),
    )
    assert approved.status_code == 200
    skill: Final = approved.json()["skill"]
    assert skill["security_scan_passed"] is not has_issues
    assert skill["created_by"] == MEMBER_KEY
    assert skill["approved_by"] == approved.json()["submission"]["reviewed_by"]
    assert skill["published_by"] == skill["approved_by"]
    _stored_upload(hub)
    listed: Final = (await hub.client.get(HUB, headers=_headers(MASTER_KEY))).json()["skills"][0]
    assert listed["security_scan_passed"] is not has_issues
    assert listed["published_by"] == skill["published_by"]


@pytest.mark.parametrize("field,order,expected", [
    ("name", "asc", "a"), ("name", "desc", "c"),
    ("download_count", "desc", "a"), ("download_count", "asc", "b"),
    ("updated_at", "desc", "c"), ("updated_at", "asc", "b"),
])
async def test_catalog_sorts_all_matches_before_pagination(
    hub: HubClient, field: str, order: str, expected: str,
) -> None:
    hub.plugins.find_many.return_value = [
        prisma_models.LiteLLM_ClaudeCodePluginTable(
            id=name, name=name, enabled=True, anonymous_download_count=0, updated_at=datetime(2026, 1, day),
            manifest_json=json.dumps({"source": {"source": "github", "repo": "example/skills"}}),
        ) for name, day in (("a", 2), ("b", 1), ("c", 3))
    ]
    hub.database.query_raw.return_value = [
        {"id": name, "download_count": count} for name, count in (("a", 8), ("b", 0), ("c", 3))
    ]
    response = await hub.client.get(HUB, headers=_headers(PLAIN_KEY), params={
        "sort_by": field, "sort_order": order, "limit": 1, "offset": 0,
    })
    assert response.status_code == 200, response.text
    assert [skill["name"] for skill in response.json()["skills"]] == [expected]


async def _restorable_version(hub: HubClient) -> prisma_models.LiteLLM_ClaudeCodePluginVersion:
    assert (await _upload(hub, _zip(FILES), version="1.0.0")).status_code == 200
    original: Final = _stored_upload(hub, enabled=False)
    history: Final = hub.database.litellm_claudecodepluginversion
    retained: Final = prisma_models.LiteLLM_ClaudeCodePluginVersion.model_validate({
        "id": "retained", **history.create.await_args.kwargs["data"],
    })
    history.count.return_value = 1
    history.find_first.return_value = retained
    hub.database.query_raw.return_value = [{
        "id": retained.id, "version": retained.version, "description": retained.description,
        "manifest": json.loads(retained.manifest_json), "published_at": retained.published_at,
        "published_by": retained.published_by, "archive_available": True,
    }]
    hub.plugins.find_unique.return_value = original.model_copy(update={
        "version": "2.0.0", "description": "Current description",
        "updated_at": datetime(2026, 1, 2),
    })
    return retained


def _restore_body() -> dict[str, object]:
    return {"expected_skill_id": "hub-id", "expected_updated_at": "2026-01-02T00:00:00"}


@pytest.mark.asyncio
async def test_restore_republishes_retained_bytes_and_metadata_with_new_version(hub: HubClient) -> None:
    retained: Final = await _restorable_version(hub)
    response: Final = await hub.client.post(
        f"{HUB}/{NAME}/versions/retained/restore", headers=_headers(MASTER_KEY), json=_restore_body(),
    )
    assert response.status_code == 200, response.text
    assert response.json()["version"] == "2.0.1"
    assert response.json()["description"] == retained.description
    assert response.json()["id"] == "hub-id"
    assert response.json()["visibility"] == "private"
    assert response.json()["approved_by"] is None
    assert response.json()["security_scan_passed"] is False
    latest: Final = hub.database.litellm_claudecodepluginversion.create.await_args.kwargs["data"]
    assert latest["content"] == retained.content
    assert latest["version"] == "2.0.1"
    assert json.loads(latest["manifest_json"])["agent_skill"] == json.loads(retained.manifest_json)["agent_skill"]
    assert retained.version == "1.0.0"
    assert latest["published_by"] == response.json()["published_by"]


@pytest.mark.parametrize("scan_passed", (True, False))
@pytest.mark.asyncio
async def test_restore_carries_retained_scan_and_credits_restoring_admin(hub: HubClient, scan_passed: bool) -> None:
    retained: Final = await _restorable_version(hub)
    manifest: Final = {
        **json.loads(retained.manifest_json),
        "_litellm_review": {"approved_by": "previous-admin", "security_scan_passed": scan_passed},
    }
    hub.database.query_raw.return_value = [
        {**row, "manifest": manifest} for row in hub.database.query_raw.return_value
    ]
    hub.database.litellm_claudecodepluginversion.find_first.return_value = retained.model_copy(update={
        "manifest_json": json.dumps(manifest),
    })
    response: Final = await hub.client.post(
        f"{HUB}/{NAME}/versions/retained/restore", headers=_headers(MASTER_KEY), json=_restore_body(),
    )
    assert response.status_code == 200, response.text
    assert response.json()["security_scan_passed"] is scan_passed
    assert response.json()["approved_by"] == response.json()["published_by"] == "default_user_id"
    latest: Final = hub.database.litellm_claudecodepluginversion.create.await_args.kwargs["data"]
    assert json.loads(latest["manifest_json"])["_litellm_review"] == {
        "approved_by": "default_user_id", "security_scan_passed": scan_passed,
    }
    assert latest["published_by"] == "default_user_id"


@pytest.mark.parametrize("key,status", (
    (GRANTED_KEY, 403), (PLAIN_KEY, 403), (MEMBER_KEY, 403), (VIEWER_KEY, 403), (None, 401),
))
@pytest.mark.asyncio
async def test_restore_rejects_nonadmins_even_with_skill_grants(hub: HubClient, key: str | None, status: int) -> None:
    await _restorable_version(hub)
    hub.plugins.update.reset_mock()
    response: Final = await hub.client.post(
        f"{HUB}/{NAME}/versions/retained/restore", headers=_headers(key), json=_restore_body(),
    )
    assert response.status_code == status, response.text
    hub.plugins.update.assert_not_awaited()


@pytest.mark.parametrize("change,status", (
    ({"expected_skill_id": "deleted-skill"}, 409),
    ({"expected_updated_at": "2026-01-01T00:00:00"}, 409),
    ({"expected_updated_at": None}, 409),
    ({"version": "invalid"}, 400),
    ({"version": "1.0.0"}, 400),
    ({"_litellm_review": {"approved_by": "forged", "security_scan_passed": True}}, 422),
))
@pytest.mark.asyncio
async def test_restore_rejects_stale_state_and_invalid_requests(
    hub: HubClient, change: Mapping[str, object], status: int,
) -> None:
    await _restorable_version(hub)
    hub.plugins.update.reset_mock()
    response: Final = await hub.client.post(
        f"{HUB}/{NAME}/versions/retained/restore", headers=_headers(MASTER_KEY),
        json={**_restore_body(), **change},
    )
    assert response.status_code == status, response.text
    hub.plugins.update.assert_not_awaited()


@pytest.mark.parametrize("missing,status", (("version", 404), ("content", 409), ("source_reference", 409)))
@pytest.mark.asyncio
async def test_restore_requires_a_retained_verified_archive(hub: HubClient, missing: str, status: int) -> None:
    await _restorable_version(hub)
    if missing == "version":
        hub.database.query_raw.return_value = []
    elif missing == "source_reference":
        hub.database.query_raw.return_value = [
            {**row, "archive_available": False} for row in hub.database.query_raw.return_value
        ]
    else:
        hub.database.litellm_claudecodepluginversion.find_first.return_value = None
    hub.plugins.update.reset_mock()
    response: Final = await hub.client.post(
        f"{HUB}/{NAME}/versions/retained/restore", headers=_headers(MASTER_KEY), json=_restore_body(),
    )
    assert response.status_code == status, response.text
    hub.plugins.update.assert_not_awaited()


@pytest.mark.asyncio
async def test_restore_checks_the_snapshot_again_under_the_publication_lock(hub: HubClient) -> None:
    await _restorable_version(hub)
    current: Final = hub.plugins.find_unique.return_value
    hub.plugins.find_unique.side_effect = [
        current, current.model_copy(update={"updated_at": datetime(2026, 1, 3)}),
    ]
    hub.plugins.update.reset_mock()
    response: Final = await hub.client.post(
        f"{HUB}/{NAME}/versions/retained/restore", headers=_headers(MASTER_KEY), json=_restore_body(),
    )
    assert response.status_code == 409, response.text
    hub.plugins.update.assert_not_awaited()


@pytest.mark.parametrize("key,expected", [(OTHER_KEY, 404), (MASTER_KEY, 404), (VIEWER_KEY, 403), (PLAIN_KEY, 403)])
async def test_only_original_submitter_can_edit_or_withdraw(pending_hub: HubClient, key: str, expected: int) -> None:
    hub: Final = pending_hub
    created: Final = await _upload(hub, _zip(FILES), MEMBER_KEY)
    path: Final = f"/skill-hub/submissions/{created.json()['submission_id']}"
    for suffix in ("", "/upload", "/withdraw"):
        response: Final = await hub.client.put(
            path + suffix, headers=_headers(key),
            **({"data": {"expected_revision": "0"}, "files": {"file": ("edit.zip", _zip(FILES))}}
               if suffix == "/upload" else {"json": {"expected_revision": 0}}),
        )
        assert response.status_code == expected, response.text
    own: Final = (await hub.client.get(path, headers=_headers(MEMBER_KEY))).json()["submission"]
    assert (own["revision"], own["status"], own["can_edit"]) == (0, "pending_review", True)
    admin: Final = (await hub.client.get(path, headers=_headers(MASTER_KEY))).json()["submission"]
    assert admin["can_edit"] is False
    hub.plugins.create.assert_not_awaited()


async def test_pending_edits_preserve_owner_and_files_until_fresh_review(pending_hub: HubClient) -> None:
    hub: Final = pending_hub
    created: Final = await _upload(hub, _zip(FILES), MEMBER_KEY)
    path: Final = f"/skill-hub/submissions/{created.json()['submission_id']}"
    before: Final = (await hub.client.get(path, headers=_headers(MEMBER_KEY))).json()["submission"]
    edited: Final = await hub.client.put(path, headers=_headers(MEMBER_KEY), json={
        "expected_revision": 0, "description": "Revised metadata", "version": "3.0.0", "keywords": ["review"],
    })
    assert edited.status_code == 200, edited.text
    result: Final = edited.json()
    assert (result["revision"], result["submitted_by"], result["created_at"], result["base_updated_at"]) == (
        1, before["submitted_by"], before["created_at"], before["base_updated_at"],
    )
    assert result["proposal"]["description"] == "Revised metadata"
    _assert_portable(await hub.client.get(path + "/archive", headers=_headers(MEMBER_KEY)))
    for body in ({}, {"expected_revision": 0}):
        stale: Final = await hub.client.put(path + "/approve", headers=_headers(MASTER_KEY), json=body or None)
        assert stale.status_code == 409, stale.text
    assert (await hub.client.put(path, headers=_headers(MEMBER_KEY), json={"expected_revision": 0})).status_code == 409
    hub.plugins.create.assert_not_awaited()
    approved: Final = await hub.client.put(path + "/approve", headers=_headers(MASTER_KEY), json={"expected_revision": 1})
    assert approved.status_code == 200, approved.text
    assert approved.json()["skill"]["description"] == "Revised metadata"
    assert approved.json()["skill"]["version"] == "3.0.0"
    for suffix in ("", "/withdraw"):
        assert (await hub.client.put(path + suffix, headers=_headers(MEMBER_KEY), json={"expected_revision": 1})).status_code == 409


async def test_withdrawn_submission_cannot_be_edited_reviewed_or_published(pending_hub: HubClient) -> None:
    hub: Final = pending_hub
    created: Final = await _upload(hub, _zip(FILES), MEMBER_KEY)
    path: Final = f"/skill-hub/submissions/{created.json()['submission_id']}"
    withdrawn: Final = await hub.client.put(path + "/withdraw", headers=_headers(MEMBER_KEY), json={"expected_revision": 0})
    assert withdrawn.status_code == 200, withdrawn.text
    assert (withdrawn.json()["status"], withdrawn.json()["can_edit"], withdrawn.json()["reviewed_by"]) == ("withdrawn", False, None)
    assert withdrawn.json()["has_archive"] is False
    assert (await hub.client.get(path + "/archive", headers=_headers(MEMBER_KEY))).status_code == 404
    for suffix, key in (("", MEMBER_KEY), ("/withdraw", MEMBER_KEY), ("/approve", MASTER_KEY), ("/reject", MASTER_KEY)):
        response: Final = await hub.client.put(path + suffix, headers=_headers(key), json={"expected_revision": 0})
        assert response.status_code == 409, response.text
    listed: Final = await hub.client.get("/skill-hub/submissions?status=withdrawn", headers=_headers(MEMBER_KEY))
    assert listed.json()["count"] == 1
    hub.plugins.create.assert_not_awaited()


async def test_pending_archive_edit_revalidates_and_clears_scan(pending_hub: HubClient) -> None:
    hub: Final = pending_hub
    created: Final = await _upload(hub, _zip(FILES), MEMBER_KEY)
    identifier: Final = created.json()["submission_id"]
    path: Final = f"/skill-hub/submissions/{identifier}"
    await hub.database.litellm_skillsubmission.update(where={"id": identifier}, data={"scan_json": "old scan"})
    invalid: Final = await hub.client.put(path + "/upload", headers=_headers(MEMBER_KEY),
        data={"expected_revision": "0"}, files={"file": ("invalid.zip", b"invalid")})
    assert invalid.status_code == 422, invalid.text
    updated_files: Final = {**FILES, "references/guide.md": b"Revised pending instructions.\n"}
    edited: Final = await hub.client.put(path + "/upload", headers=_headers(MEMBER_KEY),
        data={"expected_revision": "0", "version": "2.1.0"}, files={"file": ("edit.zip", _zip(updated_files))})
    assert edited.status_code == 200, edited.text
    assert edited.json()["revision"] == 1
    stored: Final = await hub.database.litellm_skillsubmission.find_unique(where={"id": identifier})
    assert stored.scan_json is None
    archive: Final = await hub.client.get(path + "/archive", headers=_headers(MEMBER_KEY))
    with zipfile.ZipFile(io.BytesIO(archive.content)) as content:
        assert content.read("references/guide.md") == updated_files["references/guide.md"]
    hub.plugins.create.assert_not_awaited()


@pytest.mark.parametrize("published_version", (None, "legacy"))
@pytest.mark.parametrize("replacement", (False, True), ids=("metadata", "archive"))
async def test_pending_legacy_update_keeps_explicit_version_requirement(
    pending_hub: HubClient, published_version: str | None, replacement: bool,
) -> None:
    hub: Final = pending_hub
    assert (await _upload(hub, _zip(FILES))).status_code == 200
    original: Final = _stored_upload(hub)
    hub.plugins.find_unique.return_value = original.model_copy(update={"version": published_version})
    created: Final = await hub.client.put(
        f"{HUB}/{NAME}", headers=_headers(MEMBER_KEY), json={"version": "3.0.0"},
    )
    assert created.status_code == 200, created.text
    path: Final = f"/skill-hub/submissions/{created.json()['submission_id']}"
    before: Final = (await hub.client.get(path, headers=_headers(MEMBER_KEY))).json()
    edited: Final = await hub.client.put(
        path + ("/upload" if replacement else ""), headers=_headers(MEMBER_KEY),
        **({"data": {"expected_revision": "0"}, "files": {"file": ("new.zip", _zip(FILES))}}
           if replacement else {"json": {"expected_revision": 0}}),
    )
    assert edited.status_code == 409, edited.text
    assert "explicit semantic version" in edited.json()["detail"]
    assert (await hub.client.get(path, headers=_headers(MEMBER_KEY))).json() == before
    hub.plugins.update.assert_not_awaited()
