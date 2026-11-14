import hashlib
import io
import json
import zipfile
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Final
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from prisma import Base64
from prisma import models as prisma_models

import litellm
from litellm.models.skills import LiteLLM_SkillsTable
from litellm.proxy.anthropic_endpoints.claude_code_endpoints.skill_upload_archive import build_uploaded_skill_archive
from litellm.proxy.deps.repositories import get_plugin_upload_repository
from litellm.proxy.discovery_endpoints.agent_skills_archive import build_agent_skill_archive
from litellm.proxy.discovery_endpoints.agent_skills_endpoints import (
    MAX_CACHED_ARCHIVE_BYTES,
    MAX_INDEXED_SKILLS,
    _ARCHIVE_CACHE,
    router,
    stored_skill,
    stored_skills,
)
from litellm.proxy.discovery_endpoints.skill_hub_endpoints import router as hub_router
from litellm.repositories.plugin_upload_repository import PluginUploadRepository
from litellm.types.proxy.discovery_endpoints.agent_skills_endpoints import (
    AGENT_SKILLS_DISCOVERY_SCHEMA_URL,
    MAX_SKILL_NAME_LENGTH,
)

WELL_KNOWN_PATHS = ("/.well-known/agent-skills/index.json", "/.well-known/skills/index.json")

MANIFEST = b"""---
name: pdf-summarizer
description: Summarize a PDF into an executive brief.
---

Read the PDF, then write the brief.
"""


def zip_bytes(files: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return buffer.getvalue()


def skill(
    skill_id: str,
    *,
    display_title: str | None = "PDF Summarizer",
    description: str | None = None,
    files: dict[str, bytes] | None = None,
    updated_at: datetime | None = None,
) -> LiteLLM_SkillsTable:
    return LiteLLM_SkillsTable(
        skill_id=skill_id,
        display_title=display_title,
        description=description,
        file_content=zip_bytes(files if files is not None else {"pdf-summarizer/SKILL.md": MANIFEST}),
        updated_at=updated_at,
    )


def client_for(*skills: LiteLLM_SkillsTable, uploads: PluginUploadRepository | None = None) -> TestClient:
    app = FastAPI()
    app.include_router(router)
    app.include_router(hub_router)

    def _skills() -> tuple[LiteLLM_SkillsTable, ...]:
        return skills

    def _skill(skill_id: str) -> LiteLLM_SkillsTable | None:
        return next((candidate for candidate in skills if candidate.skill_id == skill_id), None)

    app.dependency_overrides[stored_skills] = _skills
    app.dependency_overrides[stored_skill] = _skill
    app.dependency_overrides[get_plugin_upload_repository] = lambda: uploads
    return TestClient(app)


def hub_record(
    name: str,
) -> tuple[prisma_models.LiteLLM_ClaudeCodePluginTable, prisma_models.LiteLLM_ClaudeCodePluginUpload]:
    manifest: Final = MANIFEST.replace(b"pdf-summarizer", name.encode())
    portable: Final = build_agent_skill_archive(zip_bytes({"SKILL.md": manifest}))
    assert portable is not None
    wrapper: Final = build_uploaded_skill_archive(portable.content, name)
    assert wrapper is not None
    digest: Final = hashlib.sha256(wrapper).hexdigest()
    return (
        prisma_models.LiteLLM_ClaudeCodePluginTable(
            id=f"hub-{name}", name=name, enabled=True, anonymous_download_count=0, manifest_json=json.dumps({
                "source": {
                    "source": "archive", "sha256": digest,
                    "url": f"http://testserver/claude-code/plugins/{name}/archive/{digest}.zip",
                },
                "agent_skill": {"name": name, "description": portable.declared_description, "digest": portable.digest},
            }),
        ),
        prisma_models.LiteLLM_ClaudeCodePluginUpload(plugin_name=name, content=Base64.encode(wrapper)),
    )


def hub_repository(*names: str) -> PluginUploadRepository:
    records: Final = tuple(hub_record(name) for name in names)
    prisma: Final = MagicMock()
    prisma.writer_db.execute_raw = AsyncMock(return_value=1)
    plugins: Final = prisma.writer_db.litellm_claudecodeplugintable
    plugins.find_many = AsyncMock(return_value=[record for record, _ in records])

    async def metadata(where: Mapping[str, str]) -> prisma_models.LiteLLM_ClaudeCodePluginTable | None:
        return next((record for record, _ in records if record.name == where["name"]), None)

    async def content(where: Mapping[str, object]) -> prisma_models.LiteLLM_ClaudeCodePluginUpload | None:
        return next((archive for _, archive in records if archive.plugin_name == where["plugin_name"]), None)

    plugins.find_unique = AsyncMock(side_effect=metadata)
    prisma.writer_db.litellm_claudecodepluginupload.find_first = AsyncMock(side_effect=content)
    return PluginUploadRepository(prisma)


@pytest.fixture
def index_enabled(monkeypatch):
    monkeypatch.setattr(litellm, "public_skills_index", True)


def test_discovery_is_absent_until_public_skills_index_is_enabled(monkeypatch):
    monkeypatch.setattr(litellm, "public_skills_index", False)
    client = client_for(skill("litellm_skill_1"))

    for path in WELL_KNOWN_PATHS:
        assert client.get(path).status_code == 404
    assert client.get("/v1/skills/litellm_skill_1/archive").status_code == 404


@pytest.mark.parametrize("path", WELL_KNOWN_PATHS)
def test_index_publishes_each_stored_skill_in_the_v0_2_0_shape(index_enabled, path):
    client = client_for(skill("litellm_skill_1"))

    body = client.get(path).json()

    assert body["$schema"] == AGENT_SKILLS_DISCOVERY_SCHEMA_URL
    assert len(body["skills"]) == 1
    entry = body["skills"][0]
    assert entry["name"] == "pdf-summarizer"
    assert entry["type"] == "archive"
    assert entry["description"] == "Summarize a PDF into an executive brief."
    assert entry["url"].endswith("/v1/skills/litellm_skill_1/archive")
    assert entry["digest"].startswith("sha256:")


def test_index_digest_matches_the_bytes_the_archive_route_serves(index_enabled):
    client = client_for(skill("litellm_skill_1"))

    entry = client.get(WELL_KNOWN_PATHS[0]).json()["skills"][0]
    downloaded = client.get(entry["url"])

    assert downloaded.status_code == 200
    assert downloaded.headers["content-type"] == "application/zip"
    assert entry["digest"] == f"sha256:{hashlib.sha256(downloaded.content).hexdigest()}"


def test_install_name_falls_back_to_the_manifest_name_without_a_display_title(index_enabled):
    client = client_for(skill("litellm_skill_1", display_title=None))

    assert client.get(WELL_KNOWN_PATHS[0]).json()["skills"][0]["name"] == "pdf-summarizer"


@pytest.mark.parametrize(
    "manifest, stored_description, expected",
    [
        (MANIFEST, "registry copy", "Summarize a PDF into an executive brief."),
        (b"no frontmatter here", "registry copy", "registry copy"),
        (b"no frontmatter here", None, "PDF Summarizer"),
    ],
)
def test_description_prefers_the_manifest_then_the_registry_then_the_title(
    index_enabled, manifest, stored_description, expected
):
    client = client_for(
        skill(
            "litellm_skill_1",
            description=stored_description,
            files={"pdf-summarizer/SKILL.md": manifest},
        )
    )

    assert client.get(WELL_KNOWN_PATHS[0]).json()["skills"][0]["description"] == expected


def test_skills_sharing_a_title_get_distinct_install_names(index_enabled):
    client = client_for(
        skill("litellm_skill_2", files={"pdf-summarizer/SKILL.md": b"second"}),
        skill("litellm_skill_1", files={"pdf-summarizer/SKILL.md": b"first"}),
    )

    names = [entry["name"] for entry in client.get(WELL_KNOWN_PATHS[0]).json()["skills"]]

    assert names == ["pdf-summarizer", "pdf-summarizer-2"]


@pytest.mark.parametrize("hosted_names, hub_names, aliases", (
    (("foo",), ("foo", "foo-2"), ("foo-3", "foo-2")),
    (("a" * 64,), ("a" * 64, "a" * 62 + "-2"), ("a" * 62 + "-3", "a" * 62 + "-2")),
    (("foo", "foo"), ("foo", "foo-2"), ("foo-3", "foo-2-2")),
    (("foo", "foo", "foo-2"), ("foo", "foo-2", "foo-3"), ("foo-4", "foo-2-2", "foo-3-2")),
    (("bar", "bar", "foo", "foo"), ("bar", "bar-2", "foo", "foo-2"), ("bar-3", "bar-2-2", "foo-3", "foo-2-2")),
))
def test_hub_collisions_keep_every_hosted_entry_and_only_alias_hub_install_names(
    index_enabled: None, hosted_names: tuple[str, ...], hub_names: tuple[str, ...], aliases: tuple[str, ...],
) -> None:
    hosted: Final = tuple(
        skill(f"hosted-{position}", files={"s/SKILL.md": MANIFEST.replace(b"pdf-summarizer", name.encode())})
        for position, name in enumerate(hosted_names)
    )
    before: Final = client_for(*reversed(hosted)).get(WELL_KNOWN_PATHS[0]).json()["skills"]
    client: Final = client_for(*reversed(hosted), uploads=hub_repository(*hub_names))
    response: Final = client.get(WELL_KNOWN_PATHS[0])
    assert response.status_code == 200, response.text
    entries: Final = response.json()["skills"]
    assert entries[:len(hosted)] == before
    assert len({entry["name"] for entry in entries}) == len(hosted) + len(hub_names)
    assert all(len(entry["name"]) <= MAX_SKILL_NAME_LENGTH for entry in entries)
    assert tuple(entry["name"] for entry in entries[len(hosted):]) == aliases
    assert client.get(WELL_KNOWN_PATHS[1]).json() == response.json()
    assert client.get(WELL_KNOWN_PATHS[0]).json() == response.json()
    for entry, declared_name in zip(entries, hosted_names + hub_names, strict=True):
        downloaded: Final = client.get(entry["url"])
        assert downloaded.status_code == 200, downloaded.text
        assert entry["digest"] == f"sha256:{hashlib.sha256(downloaded.content).hexdigest()}"
        with zipfile.ZipFile(io.BytesIO(downloaded.content)) as archive:
            assert archive.read("SKILL.md") == MANIFEST.replace(b"pdf-summarizer", declared_name.encode())


@pytest.mark.parametrize("hosted_count", (MAX_INDEXED_SKILLS - 1, MAX_INDEXED_SKILLS + 1))
def test_combined_index_cap_keeps_hosted_entries_before_hub_entries(index_enabled: None, hosted_count: int) -> None:
    hosted: Final = tuple(skill(f"hosted-{position}") for position in range(hosted_count))
    client: Final = client_for(*hosted, uploads=hub_repository("first", "second"))
    response: Final = client.get(WELL_KNOWN_PATHS[0])
    assert response.status_code == 200, response.text
    entries: Final = response.json()["skills"]
    retained: Final = min(hosted_count, MAX_INDEXED_SKILLS)
    assert len(entries) == MAX_INDEXED_SKILLS
    assert len({entry["name"] for entry in entries}) == MAX_INDEXED_SKILLS
    assert [entry["url"] for entry in entries[:retained]] == [
        f"http://testserver/v1/skills/{item.skill_id}/archive" for item in reversed(hosted[:MAX_INDEXED_SKILLS])
    ]
    assert [entry["name"] for entry in entries[retained:]] == (["first"] if retained < MAX_INDEXED_SKILLS else [])


def test_uploads_without_a_root_manifest_are_left_out_of_the_index(index_enabled):
    client = client_for(
        skill("litellm_skill_1"),
        skill("litellm_skill_2", files={"pdf-summarizer/reference.md": b"no manifest"}),
    )

    body = client.get(WELL_KNOWN_PATHS[0]).json()

    assert [entry["url"].split("/")[-2] for entry in body["skills"]] == ["litellm_skill_1"]
    assert client.get("/v1/skills/litellm_skill_2/archive").status_code == 404


def test_archive_route_404s_for_a_skill_that_does_not_exist(index_enabled):
    client = client_for(skill("litellm_skill_1"))

    assert client.get("/v1/skills/litellm_skill_missing/archive").status_code == 404


def test_a_stored_skill_is_repacked_once_per_version(index_enabled):
    stamp = datetime(2026, 9, 6, 9, 0, tzinfo=timezone.utc)
    first = client_for(skill("litellm_skill_cached", files={"s/SKILL.md": MANIFEST}, updated_at=stamp))
    published = first.get(WELL_KNOWN_PATHS[0]).json()["skills"][0]["digest"]

    unchanged_row = client_for(
        skill("litellm_skill_cached", files={"s/SKILL.md": MANIFEST, "s/extra.md": b"rewritten"}, updated_at=stamp)
    )

    assert unchanged_row.get(WELL_KNOWN_PATHS[0]).json()["skills"][0]["digest"] == published
    assert hashlib.sha256(unchanged_row.get("/v1/skills/litellm_skill_cached/archive").content).hexdigest() == (
        published.removeprefix("sha256:")
    )


def test_retained_manifest_bytes_count_toward_the_archive_cache_limit(
    index_enabled: None,
) -> None:
    stamp: Final = datetime(2026, 9, 6, 9, 0, tzinfo=timezone.utc)
    name: Final = "litellm_skill_large_manifest"
    manifest: Final = MANIFEST + b"x" * MAX_CACHED_ARCHIVE_BYTES
    client: Final = client_for(skill(name, files={"s/SKILL.md": manifest}, updated_at=stamp))
    response: Final = client.get(f"/v1/skills/{name}/archive")
    assert response.status_code == 200
    assert len(response.content) < MAX_CACHED_ARCHIVE_BYTES
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        stored: Final = archive.read("SKILL.md")
    actual: Final = (len(stored), hashlib.sha256(stored).hexdigest())
    expected: Final = (len(manifest), hashlib.sha256(manifest).hexdigest())
    assert actual == expected
    assert _ARCHIVE_CACHE.get_cache(f"{name}:{stamp.isoformat()}") is None


def test_a_skill_edited_since_the_last_read_is_republished(index_enabled):
    stamp = datetime(2026, 9, 6, 9, 0, tzinfo=timezone.utc)
    before = client_for(skill("litellm_skill_edited", files={"s/SKILL.md": MANIFEST}, updated_at=stamp))
    published = before.get(WELL_KNOWN_PATHS[0]).json()["skills"][0]["digest"]

    after = client_for(
        skill(
            "litellm_skill_edited",
            files={"s/SKILL.md": MANIFEST, "s/extra.md": b"rewritten"},
            updated_at=datetime(2026, 9, 6, 10, 0, tzinfo=timezone.utc),
        )
    )
    republished = after.get(WELL_KNOWN_PATHS[0]).json()["skills"][0]["digest"]

    assert republished != published
    assert hashlib.sha256(after.get("/v1/skills/litellm_skill_edited/archive").content).hexdigest() == (
        republished.removeprefix("sha256:")
    )


def test_openapi_declares_the_archive_route_as_a_zip_download(index_enabled):
    schema = client_for(skill("litellm_skill_1")).get("/openapi.json").json()

    content = schema["paths"]["/v1/skills/{skill_id}/archive"]["get"]["responses"]["200"]["content"]

    assert "application/zip" in content
    assert "application/json" not in content
