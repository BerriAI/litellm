"""
Unit tests for claude_code_marketplace_sync.py.

Covers source-ref host detection, syncing an external marketplace.json into
LiteLLM_ClaudeCodePluginTable (including relative-source rewriting and the
GitHub-only skills/ directory-scan fallback), error classification, and the
idempotent/soft-disable upsert semantics.
"""

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from types import SimpleNamespace

import httpx
import pytest

from litellm.proxy.anthropic_endpoints.claude_code_endpoints import (
    claude_code_marketplace_sync as sync_module,
)
from litellm.proxy.anthropic_endpoints.claude_code_endpoints.claude_code_marketplace_sync import (
    MarketplaceSyncError,
    _parse_source_ref,
    resolve_and_sync,
)


class _FakeTable:
    def __init__(self):
        self._rows: dict = {}

    @staticmethod
    def _matches(record, where):
        return all(getattr(record, k, None) == v for k, v in where.items())

    async def find_unique(self, where):
        for record in self._rows.values():
            if self._matches(record, where):
                return record
        return None

    async def find_many(self, where=None):
        if not where:
            return list(self._rows.values())
        return [r for r in self._rows.values() if self._matches(r, where)]

    async def count(self, where=None):
        return len(await self.find_many(where))

    async def create(self, data):
        record_data = dict(data)
        record_data.setdefault("id", str(uuid.uuid4()))
        record = SimpleNamespace(**record_data)
        self._rows[record.id] = record
        return record

    async def update(self, where, data):
        record = await self.find_unique(where)
        if record is None:
            raise ValueError(f"no record matching {where}")
        for k, v in data.items():
            setattr(record, k, v)
        return record

    async def update_many(self, where, data):
        matched = await self.find_many(where)
        for record in matched:
            for k, v in data.items():
                setattr(record, k, v)
        return len(matched)

    async def upsert(self, where, data):
        existing = await self.find_unique(where)
        if existing is not None:
            for k, v in data["update"].items():
                setattr(existing, k, v)
            return existing
        return await self.create(data["create"])

    async def delete(self, where):
        record = await self.find_unique(where)
        if record is not None:
            del self._rows[record.id]
        return record


def _make_fake_prisma_client():
    client = SimpleNamespace()
    client.db = SimpleNamespace(
        litellm_skillmarketplacetable=_FakeTable(),
        litellm_claudecodeplugintable=_FakeTable(),
    )
    return client


class _FakeClock:
    def __init__(self):
        self._counter = 0

    def now(self, tz=None):
        self._counter += 1
        return datetime(2024, 1, 1, tzinfo=tz) + timedelta(seconds=self._counter)


_ANTHROPIC_SKILLS_MANIFEST = {
    "name": "anthropic-agent-skills",
    "owner": {"name": "Keith Lazuka", "email": "klazuka@anthropic.com"},
    "metadata": {"description": "Anthropic example skills", "version": "1.0.0"},
    "plugins": [
        {
            "name": "document-skills",
            "description": (
                "Collection of document processing suite including Excel, Word, PowerPoint, and PDF capabilities"
            ),
            "source": "./",
            "strict": False,
            "skills": ["./skills/xlsx", "./skills/docx", "./skills/pptx", "./skills/pdf"],
        },
        {
            "name": "example-skills",
            "description": "Collection of example skills demonstrating various capabilities",
            "source": "./",
            "strict": False,
            "skills": ["./skills/algorithmic-art"],
        },
        {
            "name": "claude-api",
            "description": "Claude API and SDK documentation skill",
            "source": "./",
            "strict": False,
            "skills": ["./skills/claude-api"],
        },
    ],
}

_FIND_SKILLS_SKILL_MD = """---
name: find-skills
description: Helps users discover and install agent skills across GitHub repositories.
---

# Find Skills

Body content describing how to discover and install skills.
"""

_ANTHROPIC_MANIFEST_URL = "https://raw.githubusercontent.com/anthropics/skills/main/.claude-plugin/marketplace.json"


async def _create_marketplace(client, *, name, source_ref, branch=None):
    return await client.db.litellm_skillmarketplacetable.create(
        data={
            "name": name,
            "source_type": "claude_marketplace_json",
            "source_ref": source_ref,
            "branch": branch,
            "sync_status": "pending",
        }
    )


@pytest.mark.parametrize(
    "raw,expected_host,expected_repo",
    [
        ("anthropics/skills", "github", "anthropics/skills"),
        ("https://github.com/anthropics/skills", "github", "anthropics/skills"),
        ("https://gitlab.com/foo/bar", "gitlab", "foo/bar"),
        ("https://bitbucket.org/foo/bar", "bitbucket", "foo/bar"),
        ("https://example.com/marketplace.json", "url", "https://example.com/marketplace.json"),
    ],
)
def test_parse_source_ref(raw, expected_host, expected_repo):
    resolved = _parse_source_ref(raw)
    assert resolved.host == expected_host
    assert resolved.repo_or_url == expected_repo


def test_marketplace_sync_error_str_includes_reason_and_detail():
    err = MarketplaceSyncError(reason="unreachable", detail="boom")
    assert str(err) == "unreachable: boom"
    assert err.reason == "unreachable"
    assert err.detail == "boom"


@pytest.mark.asyncio
async def test_resolve_and_sync_rewrites_relative_sources(monkeypatch):
    """Regression test: a '.claude-plugin/marketplace.json' plugin entry whose
    ``source`` is the relative shorthand './' must be rewritten into a resolvable
    github reference, never left as the bare './' string Claude Code can't install."""
    client = _make_fake_prisma_client()
    marketplace = await _create_marketplace(client, name="anthropic-agent-skills", source_ref="anthropics/skills")

    async def _get(http_client, url, **kwargs):
        assert url == _ANTHROPIC_MANIFEST_URL
        return httpx.Response(200, json=_ANTHROPIC_SKILLS_MANIFEST)

    monkeypatch.setattr(sync_module, "async_safe_get", _get)

    result = await resolve_and_sync(client, marketplace)

    assert result.status == "success"
    assert result.plugin_count == 3

    plugins = await client.db.litellm_claudecodeplugintable.find_many(where={"marketplace_id": marketplace.id})
    plugins_by_name = {p.name: p for p in plugins}
    assert set(plugins_by_name) == {
        "anthropic-agent-skills--document-skills",
        "anthropic-agent-skills--example-skills",
        "anthropic-agent-skills--claude-api",
    }
    for plugin in plugins_by_name.values():
        assert plugin.enabled is False
        source = json.loads(plugin.manifest_json)["source"]
        assert source == {"source": "github", "repo": "anthropics/skills"}
        assert source != "./"

    refreshed_marketplace = await client.db.litellm_skillmarketplacetable.find_unique(where={"id": marketplace.id})
    assert refreshed_marketplace.sync_status == "success"


@pytest.mark.asyncio
async def test_resolve_and_sync_falls_back_to_skills_directory_scan(monkeypatch):
    """A repo with no marketplace.json (404) falls back to scanning skills/*/SKILL.md
    frontmatter (github-only), pulling name/description from the frontmatter."""
    client = _make_fake_prisma_client()
    marketplace = await _create_marketplace(client, name="vercel-skills", source_ref="vercel-labs/skills")

    manifest_url = "https://raw.githubusercontent.com/vercel-labs/skills/main/.claude-plugin/marketplace.json"
    plugin_json_url = "https://raw.githubusercontent.com/vercel-labs/skills/main/.claude-plugin/plugin.json"
    contents_url = "https://api.github.com/repos/vercel-labs/skills/contents/skills?ref=main"
    skill_md_url = "https://raw.githubusercontent.com/vercel-labs/skills/main/skills/find-skills/SKILL.md"

    async def _get(http_client, url, **kwargs):
        if url == manifest_url:
            return httpx.Response(404)
        if url == plugin_json_url:
            return httpx.Response(404)
        if url == contents_url:
            return httpx.Response(
                200,
                json=[{"name": "find-skills", "path": "skills/find-skills", "type": "dir"}],
            )
        if url == skill_md_url:
            return httpx.Response(200, text=_FIND_SKILLS_SKILL_MD)
        raise AssertionError(f"unexpected url requested: {url}")

    monkeypatch.setattr(sync_module, "async_safe_get", _get)

    result = await resolve_and_sync(client, marketplace)

    assert result.status == "success"
    assert result.plugin_count == 1

    plugins = await client.db.litellm_claudecodeplugintable.find_many(where={"marketplace_id": marketplace.id})
    assert len(plugins) == 1
    plugin = plugins[0]
    assert plugin.name == "vercel-skills--find-skills"
    assert plugin.description == "Helps users discover and install agent skills across GitHub repositories."
    source = json.loads(plugin.manifest_json)["source"]
    assert source == {
        "source": "git-subdir",
        "url": "https://github.com/vercel-labs/skills.git",
        "path": "skills/find-skills",
    }


@pytest.mark.asyncio
async def test_resolve_and_sync_retries_transient_failures_then_succeeds(monkeypatch):
    """Regression test: a request that fails with a connection error on its
    first attempt but succeeds on retry must not be treated as a permanent
    failure - this is the exact shape of the flakiness a single unlucky
    request among many concurrent fetches used to cause. Uses the contents
    listing (a single call site) so the retry count is unambiguous, unlike
    a SKILL.md URL which is hit once for discovery and again for content."""
    client = _make_fake_prisma_client()
    marketplace = await _create_marketplace(client, name="vercel-skills", source_ref="vercel-labs/skills")

    manifest_url = "https://raw.githubusercontent.com/vercel-labs/skills/main/.claude-plugin/marketplace.json"
    plugin_json_url = "https://raw.githubusercontent.com/vercel-labs/skills/main/.claude-plugin/plugin.json"
    contents_url = "https://api.github.com/repos/vercel-labs/skills/contents/skills?ref=main"
    skill_md_url = "https://raw.githubusercontent.com/vercel-labs/skills/main/skills/find-skills/SKILL.md"

    contents_attempts = {"count": 0}

    async def _get(http_client, url, **kwargs):
        if url == manifest_url:
            return httpx.Response(404)
        if url == plugin_json_url:
            return httpx.Response(404)
        if url == contents_url:
            contents_attempts["count"] += 1
            if contents_attempts["count"] == 1:
                raise httpx.ConnectError("connection reset", request=httpx.Request("GET", url))
            return httpx.Response(
                200,
                json=[{"name": "find-skills", "path": "skills/find-skills", "type": "dir"}],
            )
        if url == skill_md_url:
            return httpx.Response(200, text=_FIND_SKILLS_SKILL_MD)
        raise AssertionError(f"unexpected url requested: {url}")

    monkeypatch.setattr(sync_module, "async_safe_get", _get)
    _real_sleep = asyncio.sleep
    monkeypatch.setattr(sync_module.asyncio, "sleep", lambda *_a, **_kw: _real_sleep(0))

    result = await resolve_and_sync(client, marketplace)

    assert result.status == "success"
    assert result.plugin_count == 1
    assert result.skipped_count == 0
    assert contents_attempts["count"] == 2, "expected exactly one retry after the first transient failure"


@pytest.mark.asyncio
async def test_resolve_and_sync_skips_skill_that_fails_after_retries_exhausted(monkeypatch):
    """Regression test: a skill whose SKILL.md is confirmed to exist (discovery
    succeeds) but whose content fetch keeps failing even after retries must be
    skipped (and counted via skipped_count), not abort the whole sync - the
    other, healthy skill must still import successfully."""
    client = _make_fake_prisma_client()
    marketplace = await _create_marketplace(client, name="vercel-skills", source_ref="vercel-labs/skills")

    manifest_url = "https://raw.githubusercontent.com/vercel-labs/skills/main/.claude-plugin/marketplace.json"
    plugin_json_url = "https://raw.githubusercontent.com/vercel-labs/skills/main/.claude-plugin/plugin.json"
    contents_url = "https://api.github.com/repos/vercel-labs/skills/contents/skills?ref=main"
    healthy_skill_md_url = "https://raw.githubusercontent.com/vercel-labs/skills/main/skills/find-skills/SKILL.md"
    flaky_skill_md_url = "https://raw.githubusercontent.com/vercel-labs/skills/main/skills/flaky-skill/SKILL.md"

    # First call to flaky_skill_md_url is the discovery-phase existence check
    # (must succeed so this candidate is confirmed as a real skill); every
    # call after that is the content-fetch phase, which keeps failing.
    flaky_attempts = {"count": 0}

    async def _get(http_client, url, **kwargs):
        if url == manifest_url:
            return httpx.Response(404)
        if url == plugin_json_url:
            return httpx.Response(404)
        if url == contents_url:
            return httpx.Response(
                200,
                json=[
                    {"name": "find-skills", "path": "skills/find-skills", "type": "dir"},
                    {"name": "flaky-skill", "path": "skills/flaky-skill", "type": "dir"},
                ],
            )
        if url == healthy_skill_md_url:
            return httpx.Response(200, text=_FIND_SKILLS_SKILL_MD)
        if url == flaky_skill_md_url:
            flaky_attempts["count"] += 1
            if flaky_attempts["count"] == 1:
                return httpx.Response(200, text=_FIND_SKILLS_SKILL_MD)
            raise httpx.ConnectError("connection reset", request=httpx.Request("GET", url))
        raise AssertionError(f"unexpected url requested: {url}")

    monkeypatch.setattr(sync_module, "async_safe_get", _get)
    _real_sleep = asyncio.sleep
    monkeypatch.setattr(sync_module.asyncio, "sleep", lambda *_a, **_kw: _real_sleep(0))

    result = await resolve_and_sync(client, marketplace)

    assert result.status == "success"
    assert result.plugin_count == 1
    assert result.skipped_count == 1

    plugins = await client.db.litellm_claudecodeplugintable.find_many(where={"marketplace_id": marketplace.id})
    assert len(plugins) == 1
    assert plugins[0].name == "vercel-skills--find-skills"

    refreshed_marketplace = await client.db.litellm_skillmarketplacetable.find_unique(where={"id": marketplace.id})
    assert refreshed_marketplace.sync_status == "success"
    assert refreshed_marketplace.skipped_count == 1


@pytest.mark.asyncio
async def test_resolve_and_sync_reads_plugin_json_explicit_skill_list(monkeypatch):
    """Regression test: a repo with no marketplace.json but a single-plugin
    ``.claude-plugin/plugin.json`` (explicit ``skills: [...]`` path list, e.g.
    inference-sh/skills) must resolve every listed SKILL.md - not fall through
    to the skills/ directory-scan convention, which this repo doesn't use."""
    client = _make_fake_prisma_client()
    marketplace = await _create_marketplace(client, name="inference-sh", source_ref="inference-sh/skills")

    manifest_url = "https://raw.githubusercontent.com/inference-sh/skills/main/.claude-plugin/marketplace.json"
    plugin_json_url = "https://raw.githubusercontent.com/inference-sh/skills/main/.claude-plugin/plugin.json"
    skill_md_url = "https://raw.githubusercontent.com/inference-sh/skills/main/guides/prompt-engineering/SKILL.md"

    async def _get(http_client, url, **kwargs):
        if url == manifest_url:
            return httpx.Response(404)
        if url == plugin_json_url:
            return httpx.Response(
                200,
                json={
                    "name": "inference-sh",
                    "description": "AI agent skills via inference.sh",
                    "skills": ["./guides/prompt-engineering/SKILL.md"],
                },
            )
        if url == skill_md_url:
            return httpx.Response(
                200,
                text="---\nname: prompt-engineering\ndescription: Write better prompts.\n---\nBody.",
            )
        raise AssertionError(f"unexpected url requested: {url}")

    monkeypatch.setattr(sync_module, "async_safe_get", _get)

    result = await resolve_and_sync(client, marketplace)

    assert result.status == "success"
    assert result.plugin_count == 1

    plugins = await client.db.litellm_claudecodeplugintable.find_many(where={"marketplace_id": marketplace.id})
    assert len(plugins) == 1
    assert plugins[0].name == "inference-sh--prompt-engineering"
    source = json.loads(plugins[0].manifest_json)["source"]
    assert source == {
        "source": "git-subdir",
        "url": "https://github.com/inference-sh/skills.git",
        "path": "guides/prompt-engineering",
    }


@pytest.mark.asyncio
async def test_resolve_and_sync_falls_back_to_root_directory_scan(monkeypatch):
    """Regression test: a repo with no marketplace.json, no plugin.json, and no
    skills/ folder at all (e.g. gstack, whose skills sit directly at repo root
    as <name>/SKILL.md) must still be discovered via a root-level scan."""
    client = _make_fake_prisma_client()
    marketplace = await _create_marketplace(client, name="gstack", source_ref="garrytan/gstack")

    manifest_url = "https://raw.githubusercontent.com/garrytan/gstack/main/.claude-plugin/marketplace.json"
    plugin_json_url = "https://raw.githubusercontent.com/garrytan/gstack/main/.claude-plugin/plugin.json"
    skills_contents_url = "https://api.github.com/repos/garrytan/gstack/contents/skills?ref=main"
    root_contents_url = "https://api.github.com/repos/garrytan/gstack/contents/?ref=main"
    investigate_skill_md_url = "https://raw.githubusercontent.com/garrytan/gstack/main/investigate/SKILL.md"
    bin_skill_md_url = "https://raw.githubusercontent.com/garrytan/gstack/main/bin/SKILL.md"

    async def _get(http_client, url, **kwargs):
        if url in (manifest_url, plugin_json_url, skills_contents_url, bin_skill_md_url):
            return httpx.Response(404)
        if url == root_contents_url:
            return httpx.Response(
                200,
                json=[
                    {"name": "investigate", "path": "investigate", "type": "dir"},
                    {"name": "bin", "path": "bin", "type": "dir"},
                    {"name": "README.md", "path": "README.md", "type": "file"},
                ],
            )
        if url == investigate_skill_md_url:
            return httpx.Response(
                200,
                text="---\nname: investigate\ndescription: Systematic debugging.\n---\nBody.",
            )
        raise AssertionError(f"unexpected url requested: {url}")

    monkeypatch.setattr(sync_module, "async_safe_get", _get)

    result = await resolve_and_sync(client, marketplace)

    assert result.status == "success"
    assert result.plugin_count == 1

    plugins = await client.db.litellm_claudecodeplugintable.find_many(where={"marketplace_id": marketplace.id})
    assert len(plugins) == 1
    assert plugins[0].name == "gstack--investigate"
    source = json.loads(plugins[0].manifest_json)["source"]
    assert source == {
        "source": "git-subdir",
        "url": "https://github.com/garrytan/gstack.git",
        "path": "investigate",
    }


@pytest.mark.asyncio
async def test_resolve_and_sync_connection_failure_is_unreachable(monkeypatch):
    client = _make_fake_prisma_client()
    marketplace = await _create_marketplace(client, name="flaky-marketplace", source_ref="org/flaky-repo")

    async def _get(http_client, url, **kwargs):
        raise httpx.ConnectError("connection refused", request=httpx.Request("GET", url))

    monkeypatch.setattr(sync_module, "async_safe_get", _get)

    result = await resolve_and_sync(client, marketplace)

    assert result.status == "error"
    refreshed = await client.db.litellm_skillmarketplacetable.find_unique(where={"id": marketplace.id})
    assert refreshed.sync_status == "error"
    assert refreshed.sync_error.startswith("unreachable:")


@pytest.mark.asyncio
async def test_resolve_and_sync_non_200_is_http_error(monkeypatch):
    client = _make_fake_prisma_client()
    marketplace = await _create_marketplace(client, name="broken-marketplace", source_ref="org/broken-repo")

    async def _get(http_client, url, **kwargs):
        return httpx.Response(500)

    monkeypatch.setattr(sync_module, "async_safe_get", _get)

    result = await resolve_and_sync(client, marketplace)

    assert result.status == "error"
    refreshed = await client.db.litellm_skillmarketplacetable.find_unique(where={"id": marketplace.id})
    assert refreshed.sync_status == "error"
    assert refreshed.sync_error.startswith("http_error:")


@pytest.mark.asyncio
async def test_resolve_and_sync_malformed_json_is_invalid_json(monkeypatch):
    client = _make_fake_prisma_client()
    marketplace = await _create_marketplace(client, name="badjson-marketplace", source_ref="org/badjson-repo")

    async def _get(http_client, url, **kwargs):
        return httpx.Response(200, text="{not valid json")

    monkeypatch.setattr(sync_module, "async_safe_get", _get)

    result = await resolve_and_sync(client, marketplace)

    assert result.status == "error"
    refreshed = await client.db.litellm_skillmarketplacetable.find_unique(where={"id": marketplace.id})
    assert refreshed.sync_status == "error"
    assert refreshed.sync_error.startswith("invalid_json:")


@pytest.mark.asyncio
async def test_resolve_and_sync_schema_violation_is_invalid_schema(monkeypatch):
    client = _make_fake_prisma_client()
    marketplace = await _create_marketplace(client, name="badschema-marketplace", source_ref="org/badschema-repo")

    async def _get(http_client, url, **kwargs):
        # Valid JSON, but missing the required top-level "name" field.
        return httpx.Response(200, json={"foo": "bar"})

    monkeypatch.setattr(sync_module, "async_safe_get", _get)

    result = await resolve_and_sync(client, marketplace)

    assert result.status == "error"
    refreshed = await client.db.litellm_skillmarketplacetable.find_unique(where={"id": marketplace.id})
    assert refreshed.sync_status == "error"
    assert refreshed.sync_error.startswith("invalid_schema:")


@pytest.mark.asyncio
async def test_resolve_and_sync_non_github_404_does_not_attempt_dir_scan(monkeypatch):
    """Regression test: the skills/ directory-scan fallback is GitHub-only. A
    gitlab/bitbucket/plain-url source that 404s on its manifest path must fail
    the sync outright and must never hit the GitHub contents API."""
    client = _make_fake_prisma_client()
    marketplace = await _create_marketplace(client, name="gitlab-marketplace", source_ref="https://gitlab.com/foo/bar")

    manifest_url = "https://gitlab.com/foo/bar/-/raw/main/.claude-plugin/marketplace.json"
    calls = []

    async def _get(http_client, url, **kwargs):
        calls.append(url)
        return httpx.Response(404)

    monkeypatch.setattr(sync_module, "async_safe_get", _get)

    result = await resolve_and_sync(client, marketplace)

    assert result.status == "error"
    refreshed = await client.db.litellm_skillmarketplacetable.find_unique(where={"id": marketplace.id})
    assert refreshed.sync_status == "error"
    assert calls == [manifest_url]
    assert all("api.github.com" not in url for url in calls)


@pytest.mark.asyncio
async def test_resolve_and_sync_is_idempotent(monkeypatch):
    monkeypatch.setattr(sync_module, "datetime", _FakeClock())

    client = _make_fake_prisma_client()
    marketplace = await _create_marketplace(client, name="anthropic-agent-skills", source_ref="anthropics/skills")

    async def _get(http_client, url, **kwargs):
        return httpx.Response(200, json=_ANTHROPIC_SKILLS_MANIFEST)

    monkeypatch.setattr(sync_module, "async_safe_get", _get)

    await resolve_and_sync(client, marketplace)
    first_pass = await client.db.litellm_claudecodeplugintable.find_many(where={"marketplace_id": marketplace.id})
    assert len(first_pass) == 3
    # Extract the value now - the stored record is a live, mutable object that
    # the second sync will update in place, so keeping a reference to the
    # record itself (rather than its updated_at value) would falsely "advance"
    # this too.
    first_updated_at = next(p for p in first_pass if p.name == "anthropic-agent-skills--claude-api").updated_at

    await resolve_and_sync(client, marketplace)
    second_pass = await client.db.litellm_claudecodeplugintable.find_many(where={"marketplace_id": marketplace.id})
    assert len(second_pass) == 3

    target_after = next(p for p in second_pass if p.name == "anthropic-agent-skills--claude-api")
    assert target_after.updated_at > first_updated_at


@pytest.mark.asyncio
async def test_resolve_and_sync_soft_disables_stale_plugin(monkeypatch):
    client = _make_fake_prisma_client()
    marketplace = await _create_marketplace(client, name="anthropic-agent-skills", source_ref="anthropics/skills")

    async def _get_full(http_client, url, **kwargs):
        return httpx.Response(200, json=_ANTHROPIC_SKILLS_MANIFEST)

    monkeypatch.setattr(sync_module, "async_safe_get", _get_full)
    await resolve_and_sync(client, marketplace)

    stale_name = "anthropic-agent-skills--claude-api"
    stale_plugin = await client.db.litellm_claudecodeplugintable.find_unique(where={"name": stale_name})
    assert stale_plugin is not None
    # Simulate an admin having manually enabled this skill before the next sync.
    stale_plugin.enabled = True

    reduced_manifest = {
        **_ANTHROPIC_SKILLS_MANIFEST,
        "plugins": [p for p in _ANTHROPIC_SKILLS_MANIFEST["plugins"] if p["name"] != "claude-api"],
    }

    async def _get_reduced(http_client, url, **kwargs):
        return httpx.Response(200, json=reduced_manifest)

    monkeypatch.setattr(sync_module, "async_safe_get", _get_reduced)
    result = await resolve_and_sync(client, marketplace)

    assert result.status == "success"
    assert result.plugin_count == 2

    refreshed_stale = await client.db.litellm_claudecodeplugintable.find_unique(where={"name": stale_name})
    assert refreshed_stale is not None
    assert refreshed_stale.enabled is False


@pytest.mark.asyncio
async def test_resolve_and_sync_refuses_to_overwrite_row_owned_by_another_marketplace(monkeypatch):
    """Regression test: `name` is only conventionally namespaced as
    "{marketplace}--{skill}", not schema-enforced, so a marketplace slug or
    skill name that collides with an existing row owned by a *different*
    marketplace (or a hand-registered plugin with no marketplace_id) must be
    skipped, never silently overwritten."""
    client = _make_fake_prisma_client()

    colliding_name = "anthropic-agent-skills--claude-api"
    hand_registered = await client.db.litellm_claudecodeplugintable.create(
        data={
            "name": colliding_name,
            "description": "trusted, hand-registered plugin",
            "manifest_json": json.dumps({"source": {"source": "github", "repo": "trusted/repo"}}),
            "files_json": "{}",
            "enabled": True,
            "marketplace_id": None,
            "created_at": datetime(2023, 1, 1),
            "updated_at": datetime(2023, 1, 1),
        }
    )

    marketplace = await _create_marketplace(client, name="anthropic-agent-skills", source_ref="anthropics/skills")

    async def _get(http_client, url, **kwargs):
        return httpx.Response(200, json=_ANTHROPIC_SKILLS_MANIFEST)

    monkeypatch.setattr(sync_module, "async_safe_get", _get)

    result = await resolve_and_sync(client, marketplace)

    assert result.status == "success"
    assert result.plugin_count == 2  # document-skills, example-skills - claude-api collided and was skipped
    assert result.skipped_count == 1

    untouched = await client.db.litellm_claudecodeplugintable.find_unique(where={"name": colliding_name})
    assert untouched.marketplace_id is None
    assert untouched.enabled is True
    assert json.loads(untouched.manifest_json)["source"] == {"source": "github", "repo": "trusted/repo"}
    assert untouched.id == hand_registered.id


@pytest.mark.asyncio
async def test_resolve_and_sync_unpublishes_skill_whose_source_changed(monkeypatch):
    """Regression test: a skill's git source must not be silently swapped by
    a re-sync of the marketplace that owns it - that would let a
    compromised/malicious upstream repoint a skill with no admin re-review.
    Unlike a plain "unpublish", get_marketplace() still serves a disabled row
    to anyone with a standing allowed_skills grant, so the sync must keep the
    previously-approved manifest (source) in place rather than overwrite it -
    demoting to enabled=False alone would still hand granted callers the new,
    unreviewed source."""
    client = _make_fake_prisma_client()
    marketplace = await _create_marketplace(client, name="anthropic-agent-skills", source_ref="anthropics/skills")

    async def _get(http_client, url, **kwargs):
        return httpx.Response(200, json=_ANTHROPIC_SKILLS_MANIFEST)

    monkeypatch.setattr(sync_module, "async_safe_get", _get)
    await resolve_and_sync(client, marketplace)

    published_name = "anthropic-agent-skills--claude-api"
    published = await client.db.litellm_claudecodeplugintable.find_unique(where={"name": published_name})
    published.enabled = True  # simulate an admin having reviewed and published it
    original_manifest_json = published.manifest_json
    original_source = json.loads(original_manifest_json)["source"]

    # A repointed `source` (repo root -> a subdirectory) is exactly what an
    # upstream marketplace repo owner controls and could change unilaterally.
    changed_manifest = {
        **_ANTHROPIC_SKILLS_MANIFEST,
        "plugins": [
            p if p["name"] != "claude-api" else {**p, "source": "./skills/claude-api"}
            for p in _ANTHROPIC_SKILLS_MANIFEST["plugins"]
        ],
    }

    async def _get_changed(http_client, url, **kwargs):
        return httpx.Response(200, json=changed_manifest)

    monkeypatch.setattr(sync_module, "async_safe_get", _get_changed)
    result = await resolve_and_sync(client, marketplace)

    assert result.status == "success"

    refreshed = await client.db.litellm_claudecodeplugintable.find_unique(where={"name": published_name})
    assert refreshed.enabled is False
    # The served manifest (and therefore source) is untouched - not just the
    # enabled flag - so a caller with a standing allowed_skills grant for
    # this row still only ever sees the previously-approved source.
    assert refreshed.manifest_json == original_manifest_json
    assert json.loads(refreshed.manifest_json)["source"] == original_source


@pytest.mark.asyncio
async def test_resolve_and_sync_leaves_published_skill_enabled_when_source_unchanged(monkeypatch):
    """A re-sync that resolves to the exact same source must not touch an
    already-published skill's enabled state."""
    client = _make_fake_prisma_client()
    marketplace = await _create_marketplace(client, name="anthropic-agent-skills", source_ref="anthropics/skills")

    async def _get(http_client, url, **kwargs):
        return httpx.Response(200, json=_ANTHROPIC_SKILLS_MANIFEST)

    monkeypatch.setattr(sync_module, "async_safe_get", _get)
    await resolve_and_sync(client, marketplace)

    published_name = "anthropic-agent-skills--claude-api"
    published = await client.db.litellm_claudecodeplugintable.find_unique(where={"name": published_name})
    published.enabled = True

    await resolve_and_sync(client, marketplace)

    refreshed = await client.db.litellm_claudecodeplugintable.find_unique(where={"name": published_name})
    assert refreshed.enabled is True
