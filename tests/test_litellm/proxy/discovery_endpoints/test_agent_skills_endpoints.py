import hashlib
import io
import zipfile

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import litellm
from litellm.models.skills import LiteLLM_SkillsTable
from litellm.proxy.discovery_endpoints.agent_skills_endpoints import (
    router,
    stored_skill,
    stored_skills,
)
from litellm.types.proxy.discovery_endpoints.agent_skills_endpoints import (
    AGENT_SKILLS_DISCOVERY_SCHEMA_URL,
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
) -> LiteLLM_SkillsTable:
    return LiteLLM_SkillsTable(
        skill_id=skill_id,
        display_title=display_title,
        description=description,
        file_content=zip_bytes(files if files is not None else {"pdf-summarizer/SKILL.md": MANIFEST}),
    )


def client_for(*skills: LiteLLM_SkillsTable) -> TestClient:
    app = FastAPI()
    app.include_router(router)

    def _skills() -> tuple[LiteLLM_SkillsTable, ...]:
        return skills

    def _skill(skill_id: str) -> LiteLLM_SkillsTable | None:
        return next((candidate for candidate in skills if candidate.skill_id == skill_id), None)

    app.dependency_overrides[stored_skills] = _skills
    app.dependency_overrides[stored_skill] = _skill
    return TestClient(app)


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
