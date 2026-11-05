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
from prisma.errors import UniqueViolationError

import litellm
from litellm.caching.caching import DualCache
from litellm.proxy import proxy_server
from litellm.proxy._types import LiteLLM_ObjectPermissionTable, UserAPIKeyAuth
from litellm.proxy.deps.repositories import get_plugin_upload_repository
from litellm.proxy.discovery_endpoints.agent_skills_archive import (
    MAX_ARCHIVE_ENTRIES,
    MAX_ARCHIVE_UNPACKED_BYTES,
)
from litellm.proxy.discovery_endpoints.skill_hub_endpoints import import_http_client
from litellm.proxy.hooks.parallel_request_limiter_v3 import PROXY_MaxParallelRequestsHandler_v3
from litellm.proxy.utils import InternalUsageCache
from litellm.repositories.plugin_upload_repository import PluginUploadRepository
from tests._master_key import MASTER_KEY

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
    upstream: MagicMock
    clock: MagicMock


@pytest.fixture
async def hub(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[HubClient]:
    prisma: Final = MagicMock()
    prisma.writer_db = prisma.db
    prisma.get_data = AsyncMock(return_value=None)
    prisma.db.litellm_skillstable.find_many = AsyncMock(return_value=[])
    prisma.db.litellm_usertable.find_unique = AsyncMock(return_value=None)
    plugins: Final = prisma.db.litellm_claudecodeplugintable
    archives: Final = prisma.db.litellm_claudecodepluginupload
    plugins.create = AsyncMock(return_value=prisma_models.LiteLLM_ClaudeCodePluginTable(
        id="hub-id", name=NAME, enabled=True,
    ))
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
    async with (
        httpx.AsyncClient(transport=httpx.MockTransport(upstream), trust_env=False) as source_client,
        httpx.AsyncClient(transport=httpx.ASGITransport(app=proxy_server.app), base_url="http://test") as client,
    ):
        monkeypatch.setattr(proxy_server.app, "dependency_overrides", {
            get_plugin_upload_repository: lambda: PluginUploadRepository(prisma),
            import_http_client: lambda: source_client,
        })
        yield HubClient(client, plugins, archives, upstream, clock)


async def _upload(hub: HubClient, content: bytes, key: str | None = MASTER_KEY) -> httpx.Response:
    return await hub.client.post(
        f"{HUB}/upload",
        headers=_headers(key),
        data={"version": "2.0.0"},
        files={"file": ("notes.zip", content, "application/zip")},
    )


def _stored_upload(hub: HubClient, enabled: bool = True) -> prisma_models.LiteLLM_ClaudeCodePluginTable:
    data: Final = hub.plugins.create.await_args.kwargs["data"]
    record: Final = prisma_models.LiteLLM_ClaudeCodePluginTable(
        id="hub-id", name=data["name"], version=data["version"], description=data["description"],
        manifest_json=data["manifest_json"], enabled=enabled,
    )
    hub.plugins.find_unique.return_value = record
    hub.plugins.find_many.return_value = [record]
    hub.archives.find_first.return_value = prisma_models.LiteLLM_ClaudeCodePluginUpload(
        plugin_name=record.name, content=data["uploaded_archive"]["create"]["content"],
    )
    return record


def _assert_portable(response: httpx.Response) -> None:
    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "application/zip"
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        assert {name: archive.read(name) for name in archive.namelist()} == FILES
        assert stat.S_IMODE(archive.getinfo("scripts/check.sh").external_attr >> 16) == 0o755


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
    legacy: Final = await hub.client.get(legacy_path)
    assert legacy.status_code == 200
    assert hashlib.sha256(legacy.content).hexdigest() == manifest["source"]["sha256"]
    assert (await hub.client.get(legacy_path.replace(manifest["source"]["sha256"], "stale"))).status_code == 404
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
    if status != 200:
        hub.archives.find_first.assert_not_awaited()
    index: Final = await hub.client.get(INDEX)
    assert index.status_code == (200 if public_index else 404)
    if public_index:
        assert [entry["name"] for entry in index.json()["skills"]] == ([NAME] if enabled and not auth_required else [])


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


@pytest.mark.parametrize("user_id, admitted", (("publisher", 10), (None, 20)))
async def test_publication_limit_is_shared_by_user_and_route_and_resets(
    hub: HubClient, monkeypatch: pytest.MonkeyPatch, user_id: str | None, admitted: int,
) -> None:
    for key in (PLAIN_KEY, GRANTED_KEY):
        proxy_server.user_api_key_cache.set_cache(
            key=proxy_server.hash_token(key),
            value=UserAPIKeyAuth(
                token=proxy_server.hash_token(key), user_id=user_id,
                user_role=proxy_server.LitellmUserRoles.PROXY_ADMIN,
            ),
        )
    responses: Final = await asyncio.gather(*(
        _upload(hub, b"invalid", PLAIN_KEY) if index % 2 else hub.client.post(
            f"{HUB}/import", headers=_headers(GRANTED_KEY), json={"url": "https://example.com/skill"},
        ) for index in range(20)
    ))
    assert sum(response.status_code in (400, 422) for response in responses) == admitted
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
