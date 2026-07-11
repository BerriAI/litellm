"""
Sync logic for externally-hosted Claude Code marketplaces.

Given a ``LiteLLM_SkillMarketplaceTable`` row pointing at a git host (or a
direct URL), this module fetches the marketplace's plugin/skill catalog and
upserts the resolved entries into ``LiteLLM_ClaudeCodePluginTable`` so they
can be served through the existing marketplace/plugin endpoints.

Two upstream catalog shapes are supported:

- ``.claude-plugin/marketplace.json`` at the repo root (the Claude Code
  plugin marketplace spec).
- a bare ``skills/`` directory of ``SKILL.md`` files (one or two levels
  deep), for repos that don't publish a marketplace.json.
"""

import asyncio
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import (
    Any,  # noqa: TID251  # prisma_client crosses the untyped prisma ORM boundary; see MarketplaceRow
    Literal,
)
from urllib.parse import urlparse

import httpx
import yaml
from pydantic import BaseModel, Field, TypeAdapter, ValidationError

from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.url_utils import SSRFError, async_safe_get
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, get_async_httpx_client
from litellm.repositories.table_repositories import (
    ClaudeCodePluginRepository,
    SkillMarketplaceRepository,
)
from litellm.types.llms.custom_http import httpxSpecialProvider
from litellm.types.proxy.claude_code_endpoints import (
    GithubSource,
    GitSubdirSource,
    MarketplaceSourceType,
    PluginSourceConfig,
    UrlSource,
)

DEFAULT_SYNC_TIMEOUT_SECONDS = 10.0

# validate_url() (litellm_core_utils/url_utils.py) does a blocking
# socket.getaddrinfo() per request; discovery fans out one fetch per skill
# folder via asyncio.gather, and unbounded concurrency there serializes enough
# DNS lookups on the event loop to blow through DEFAULT_SYNC_TIMEOUT_SECONDS
# for repos with more than a handful of skills. Bounding concurrency here
# keeps each individual fetch fast without touching the shared SSRF utility.
_MAX_CONCURRENT_GITHUB_FETCHES = 6
_github_fetch_semaphore = asyncio.Semaphore(_MAX_CONCURRENT_GITHUB_FETCHES)

SourceHost = Literal["github", "gitlab", "bitbucket", "url"]
SyncErrorReason = Literal["unreachable", "http_error", "invalid_json", "invalid_schema"]


@dataclass(frozen=True, slots=True)
class MarketplaceRow:
    """The subset of ``LiteLLM_SkillMarketplaceTable`` this module needs.

    Callers construct this from the Prisma row before calling
    ``resolve_and_sync`` - keeps the untyped ORM boundary to a single
    conversion point instead of threading ``Any`` through every helper.
    """

    id: str
    name: str
    source_ref: str | None
    branch: str | None


@dataclass(frozen=True, slots=True)
class ResolvedSource:
    host: SourceHost
    repo_or_url: str
    branch: str = "main"


@dataclass(frozen=True, slots=True)
class MarketplaceSyncError(Exception):
    """Raised internally for any failure while syncing one marketplace.

    Always caught by ``resolve_and_sync`` and turned into a ``SyncResult`` -
    it never escapes this module.
    """

    reason: SyncErrorReason
    detail: str

    def __str__(self) -> str:
        return f"{self.reason}: {self.detail}"


@dataclass(frozen=True, slots=True)
class SyncResult:
    status: Literal["success", "error"]
    error: str | None
    plugin_count: int


@dataclass(frozen=True, slots=True)
class ResolvedPluginEntry:
    stored_name: str
    display_name: str
    description: str | None
    source: PluginSourceConfig


# --- source_ref parsing -----------------------------------------------------

_BARE_SHORTHAND_RE = re.compile(r"^[\w.-]+/[\w.-]+$")

_HOST_MARKERS: tuple[tuple[str, Literal["github", "gitlab", "bitbucket"]], ...] = (
    ("github.com", "github"),
    ("gitlab.com", "gitlab"),
    ("bitbucket.org", "bitbucket"),
)


def _extract_repo_path(raw: str) -> str:
    normalized = raw if "://" in raw else f"https://{raw}"
    path = urlparse(normalized).path.strip("/")
    path = path.removesuffix(".git")
    segments = path.split("/")
    return "/".join(segments[:2])


def _parse_source_ref(raw: str) -> ResolvedSource:
    stripped = raw.strip()
    if _BARE_SHORTHAND_RE.match(stripped):
        return ResolvedSource(host="github", repo_or_url=stripped)
    for marker, host in _HOST_MARKERS:
        if marker in stripped:
            return ResolvedSource(host=host, repo_or_url=_extract_repo_path(stripped))
    return ResolvedSource(host="url", repo_or_url=stripped)


def _build_manifest_url(resolved: ResolvedSource, branch: str) -> str:
    match resolved.host:
        case "github":
            return f"https://raw.githubusercontent.com/{resolved.repo_or_url}/{branch}/.claude-plugin/marketplace.json"
        case "gitlab":
            return f"https://gitlab.com/{resolved.repo_or_url}/-/raw/{branch}/.claude-plugin/marketplace.json"
        case "bitbucket":
            return f"https://bitbucket.org/{resolved.repo_or_url}/raw/{branch}/.claude-plugin/marketplace.json"
        case "url":
            return resolved.repo_or_url


def _git_clone_url(resolved: ResolvedSource) -> str:
    match resolved.host:
        case "github":
            return f"https://github.com/{resolved.repo_or_url}.git"
        case "gitlab":
            return f"https://gitlab.com/{resolved.repo_or_url}.git"
        case "bitbucket":
            return f"https://bitbucket.org/{resolved.repo_or_url}.git"
        case "url":
            return resolved.repo_or_url


# --- HTTP -----------------------------------------------------------------


async def _http_get(client: AsyncHTTPHandler, url: str, *, timeout: float) -> httpx.Response:
    try:
        # async_safe_get is SSRF-guarded (resolves + validates every redirect
        # hop) - required here because the target URL is admin-supplied.
        # Bounded by _github_fetch_semaphore: see its module-level comment.
        async with _github_fetch_semaphore:
            return await async_safe_get(client, url, headers={}, timeout=timeout)
    except SSRFError as exc:
        raise MarketplaceSyncError(reason="unreachable", detail=str(exc)) from exc
    except httpx.HTTPError as exc:
        raise MarketplaceSyncError(reason="unreachable", detail=str(exc)) from exc


def _parse_json_body(response: httpx.Response) -> object:
    try:
        return response.json()
    except json.JSONDecodeError as exc:
        raise MarketplaceSyncError(reason="invalid_json", detail=str(exc)) from exc


# --- marketplace.json parsing ----------------------------------------------


class _ExternalMarketplaceOwner(BaseModel):
    name: str
    email: str | None = None


class _ExternalMarketplacePluginEntry(BaseModel):
    name: str
    description: str | None = None
    source: str | dict[str, str]
    skills: list[str] | None = None


class _ExternalMarketplaceManifest(BaseModel):
    name: str
    owner: _ExternalMarketplaceOwner | None = None
    plugins: list[_ExternalMarketplacePluginEntry] = Field(default_factory=list)


def _parse_marketplace_manifest(response: httpx.Response) -> _ExternalMarketplaceManifest:
    body = _parse_json_body(response)
    try:
        return _ExternalMarketplaceManifest.model_validate(body)
    except ValidationError as exc:
        raise MarketplaceSyncError(reason="invalid_schema", detail=str(exc)) from exc


def _normalize_relative_path(raw: str) -> str | None:
    """Normalize a marketplace.json ``source`` relative path, rejecting traversal.

    Returns None (rather than raising) for anything unsafe so the caller can
    skip just that one plugin entry instead of failing the whole sync.
    """
    stripped = raw.strip()
    if stripped in ("", ".", "./"):
        return ""
    trimmed = stripped.removeprefix("./")
    trimmed = trimmed.strip("/")
    if not trimmed:
        return ""
    segments = trimmed.split("/")
    if any(segment in ("", "..") for segment in segments):
        return None
    return "/".join(segments)


def _resolve_relative_plugin_source(resolved: ResolvedSource, raw_path: str) -> PluginSourceConfig | None:
    if resolved.host == "url":
        # A raw marketplace.json URL isn't necessarily backed by a git repo we
        # can derive a clone URL from, so a relative plugin source can't be
        # resolved against it.
        return None

    normalized = _normalize_relative_path(raw_path)
    if normalized is None:
        return None
    if normalized == "":
        if resolved.host == "github":
            return GithubSource(repo=resolved.repo_or_url)
        return UrlSource(url=_git_clone_url(resolved))
    return GitSubdirSource(url=_git_clone_url(resolved), path=normalized)


def _resolve_manifest_entry_source(
    resolved: ResolvedSource, raw_source: str | dict[str, str]
) -> PluginSourceConfig | None:
    if isinstance(raw_source, str):
        return _resolve_relative_plugin_source(resolved, raw_source)
    try:
        return TypeAdapter(PluginSourceConfig).validate_python(raw_source)
    except ValidationError:
        return None


def _build_single_manifest_entry(
    marketplace_name: str,
    resolved: ResolvedSource,
    raw_entry: _ExternalMarketplacePluginEntry,
) -> ResolvedPluginEntry | None:
    plugin_source = _resolve_manifest_entry_source(resolved, raw_entry.source)
    if plugin_source is None:
        verbose_proxy_logger.warning(
            "skill-marketplace-sync: skipping plugin %r in marketplace %r, unresolvable source %r",
            raw_entry.name,
            marketplace_name,
            raw_entry.source,
        )
        return None
    return ResolvedPluginEntry(
        stored_name=f"{marketplace_name}--{raw_entry.name}",
        display_name=raw_entry.name,
        description=raw_entry.description,
        source=plugin_source,
    )


def _build_manifest_plugin_entries(
    marketplace_name: str,
    resolved: ResolvedSource,
    manifest: _ExternalMarketplaceManifest,
) -> tuple[ResolvedPluginEntry, ...]:
    resolved_entries = (
        _build_single_manifest_entry(marketplace_name, resolved, raw_entry) for raw_entry in manifest.plugins
    )
    return tuple(entry for entry in resolved_entries if entry is not None)


# --- skills/ directory discovery (GitHub only) ------------------------------


class _GithubContentsEntry(BaseModel):
    name: str
    path: str
    type: Literal["file", "dir", "symlink", "submodule"]


@dataclass(frozen=True, slots=True)
class _DiscoveredSkillDoc:
    skill_md_path: str
    plugin_source_path: str


class _SkillFrontmatter(BaseModel):
    name: str | None = None
    description: str | None = None


_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)$", re.DOTALL)


def _parse_skill_frontmatter(content: str) -> _SkillFrontmatter:
    match = _FRONTMATTER_RE.match(content)
    if not match:
        return _SkillFrontmatter()
    try:
        raw = yaml.safe_load(match.group(1))
    except yaml.YAMLError:
        return _SkillFrontmatter()
    if not isinstance(raw, dict):
        return _SkillFrontmatter()
    try:
        return _SkillFrontmatter.model_validate(raw)
    except ValidationError:
        return _SkillFrontmatter()


async def _list_github_contents(
    client: AsyncHTTPHandler, repo: str, path: str, branch: str, *, timeout: float
) -> tuple[_GithubContentsEntry, ...]:
    # NOTE: unauthenticated GitHub Contents API calls are capped at 60 req/hr
    # per source IP. No auth-token support yet - known v1 limitation.
    url = f"https://api.github.com/repos/{repo}/contents/{path}?ref={branch}"
    response = await _http_get(client, url, timeout=timeout)
    if response.status_code == 404:
        return ()
    if response.status_code >= 400:
        raise MarketplaceSyncError(
            reason="http_error",
            detail=f"GitHub contents API returned {response.status_code} for {url}",
        )
    body = _parse_json_body(response)
    try:
        return tuple(TypeAdapter(list[_GithubContentsEntry]).validate_python(body))
    except ValidationError as exc:
        raise MarketplaceSyncError(reason="invalid_schema", detail=str(exc)) from exc


async def _skill_md_exists(client: AsyncHTTPHandler, repo: str, branch: str, dir_path: str, *, timeout: float) -> bool:
    url = f"https://raw.githubusercontent.com/{repo}/{branch}/{dir_path}/SKILL.md"
    response = await _http_get(client, url, timeout=timeout)
    return response.status_code == 200


async def _check_nested_skill_doc(
    client: AsyncHTTPHandler,
    repo: str,
    branch: str,
    category_path: str,
    entry: _GithubContentsEntry,
    *,
    timeout: float,
) -> _DiscoveredSkillDoc | None:
    nested_path = f"{category_path}/{entry.name}"
    if await _skill_md_exists(client, repo, branch, nested_path, timeout=timeout):
        return _DiscoveredSkillDoc(skill_md_path=f"{nested_path}/SKILL.md", plugin_source_path=nested_path)
    return None


async def _discover_skill_docs_under(
    client: AsyncHTTPHandler,
    repo: str,
    branch: str,
    entry: _GithubContentsEntry,
    *,
    timeout: float,
) -> tuple[_DiscoveredSkillDoc, ...]:
    flat_path = f"skills/{entry.name}"
    if await _skill_md_exists(client, repo, branch, flat_path, timeout=timeout):
        return (_DiscoveredSkillDoc(skill_md_path=f"{flat_path}/SKILL.md", plugin_source_path=flat_path),)

    # Not a flat skill - try one level of catalog nesting (skills/<category>/<name>/SKILL.md).
    nested_entries = await _list_github_contents(client, repo, flat_path, branch, timeout=timeout)
    nested_docs = await asyncio.gather(
        *(
            _check_nested_skill_doc(client, repo, branch, flat_path, sub_entry, timeout=timeout)
            for sub_entry in nested_entries
            if sub_entry.type == "dir"
        )
    )
    return tuple(doc for doc in nested_docs if doc is not None)


async def _discover_github_skill_docs(
    client: AsyncHTTPHandler, repo: str, branch: str, *, timeout: float
) -> tuple[_DiscoveredSkillDoc, ...]:
    top_level = await _list_github_contents(client, repo, "skills", branch, timeout=timeout)
    grouped = await asyncio.gather(
        *(
            _discover_skill_docs_under(client, repo, branch, entry, timeout=timeout)
            for entry in top_level
            if entry.type == "dir"
        )
    )
    return tuple(doc for docs in grouped for doc in docs)


async def _fetch_skill_entry(
    client: AsyncHTTPHandler,
    repo: str,
    branch: str,
    marketplace_name: str,
    doc: _DiscoveredSkillDoc,
    *,
    timeout: float,
) -> ResolvedPluginEntry | None:
    url = f"https://raw.githubusercontent.com/{repo}/{branch}/{doc.skill_md_path}"
    response = await _http_get(client, url, timeout=timeout)
    if response.status_code != 200:
        return None

    frontmatter = _parse_skill_frontmatter(response.text)
    folder_name = doc.plugin_source_path.rsplit("/", 1)[-1]
    skill_name = frontmatter.name or folder_name
    return ResolvedPluginEntry(
        stored_name=f"{marketplace_name}--{skill_name}",
        display_name=skill_name,
        description=frontmatter.description,
        source=GitSubdirSource(url=f"https://github.com/{repo}.git", path=doc.plugin_source_path),
    )


# --- top-level fetch orchestration -----------------------------------------


async def _fetch_marketplace_entries(
    marketplace_row: MarketplaceRow,
) -> tuple[tuple[ResolvedPluginEntry, ...], MarketplaceSourceType]:
    if not marketplace_row.source_ref:
        raise MarketplaceSyncError(reason="invalid_schema", detail="marketplace has no source_ref to sync from")

    resolved = _parse_source_ref(marketplace_row.source_ref)
    branch = marketplace_row.branch or resolved.branch
    client = get_async_httpx_client(
        llm_provider=httpxSpecialProvider.PassThroughEndpoint,
        params={"timeout": DEFAULT_SYNC_TIMEOUT_SECONDS},
    )

    manifest_url = _build_manifest_url(resolved, branch)
    manifest_response = await _http_get(client, manifest_url, timeout=DEFAULT_SYNC_TIMEOUT_SECONDS)

    if manifest_response.status_code == 200:
        manifest = _parse_marketplace_manifest(manifest_response)
        entries = _build_manifest_plugin_entries(marketplace_row.name, resolved, manifest)
        return entries, "claude_marketplace_json"

    if manifest_response.status_code == 404 and resolved.host == "github":
        docs = await _discover_github_skill_docs(
            client, resolved.repo_or_url, branch, timeout=DEFAULT_SYNC_TIMEOUT_SECONDS
        )
        fetched = await asyncio.gather(
            *(
                _fetch_skill_entry(
                    client,
                    resolved.repo_or_url,
                    branch,
                    marketplace_row.name,
                    doc,
                    timeout=DEFAULT_SYNC_TIMEOUT_SECONDS,
                )
                for doc in docs
            )
        )
        entries = tuple(entry for entry in fetched if entry is not None)
        return entries, "skills_dir"

    if manifest_response.status_code == 404:
        raise MarketplaceSyncError(
            reason="http_error",
            detail=(
                f"no .claude-plugin/marketplace.json found for host={resolved.host!r} and "
                "directory-scan fallback is only supported for GitHub"
            ),
        )

    raise MarketplaceSyncError(
        reason="http_error",
        detail=f"manifest fetch returned HTTP {manifest_response.status_code} for {manifest_url}",
    )


# --- persistence -------------------------------------------------------------


def _build_plugin_manifest_json(entry: ResolvedPluginEntry) -> str:
    return json.dumps(
        {
            "name": entry.display_name,
            "source": entry.source.model_dump(),
            "description": entry.description,
        }
    )


async def _upsert_single_plugin(
    repository: ClaudeCodePluginRepository, marketplace_id: str, entry: ResolvedPluginEntry
) -> None:
    now = datetime.now(timezone.utc)
    manifest_json = _build_plugin_manifest_json(entry)
    await repository.table.upsert(
        where={"name": entry.stored_name},
        data={
            "create": {
                "name": entry.stored_name,
                "description": entry.description,
                "manifest_json": manifest_json,
                "files_json": "{}",
                "enabled": False,
                "marketplace_id": marketplace_id,
                "created_at": now,
                "updated_at": now,
            },
            "update": {
                "description": entry.description,
                "manifest_json": manifest_json,
                "marketplace_id": marketplace_id,
                "updated_at": now,
            },
        },
    )


# prisma_client has no importable type stubs in this codebase (generated at
# runtime, never imported by name) - every repository in
# litellm/repositories/ types it as Any for the same reason.
async def _upsert_plugin_entries(
    prisma_client: Any,  # noqa: ANN401  # see comment above
    marketplace_id: str,
    entries: tuple[ResolvedPluginEntry, ...],
) -> None:
    repository = ClaudeCodePluginRepository(prisma_client)
    await asyncio.gather(*(_upsert_single_plugin(repository, marketplace_id, entry) for entry in entries))


async def _soft_disable_stale_plugins(
    prisma_client: Any,  # noqa: ANN401  # prisma_client has no importable type stubs, see _upsert_plugin_entries
    marketplace_id: str,
    entries: tuple[ResolvedPluginEntry, ...],
) -> None:
    repository = ClaudeCodePluginRepository(prisma_client)
    existing = await repository.table.find_many(where={"marketplace_id": marketplace_id})
    fresh_names = frozenset(entry.stored_name for entry in entries)
    stale = tuple(row for row in existing if row.name not in fresh_names and row.enabled)
    await asyncio.gather(*(repository.table.update(where={"name": row.name}, data={"enabled": False}) for row in stale))


async def _record_sync_success(
    prisma_client: Any,  # noqa: ANN401  # prisma_client has no importable type stubs, see _upsert_plugin_entries
    marketplace_row: MarketplaceRow,
    source_type: MarketplaceSourceType,
) -> None:
    await SkillMarketplaceRepository(prisma_client).table.update(
        where={"id": marketplace_row.id},
        data={
            "sync_status": "success",
            "sync_error": None,
            "source_type": source_type,
            "last_synced_at": datetime.now(timezone.utc),
        },
    )


async def _record_sync_failure(
    prisma_client: Any,  # noqa: ANN401  # prisma_client has no importable type stubs, see _upsert_plugin_entries
    marketplace_row: MarketplaceRow,
    detail: str,
) -> None:
    await SkillMarketplaceRepository(prisma_client).table.update(
        where={"id": marketplace_row.id},
        data={
            "sync_status": "error",
            "sync_error": detail,
            "last_synced_at": datetime.now(timezone.utc),
        },
    )


async def resolve_and_sync(
    prisma_client: Any,  # noqa: ANN401  # prisma_client has no importable type stubs, see _upsert_plugin_entries
    marketplace_row: MarketplaceRow,
) -> SyncResult:
    """Fetch ``marketplace_row``'s upstream catalog and sync it into the plugin table.

    Never raises - any failure is recorded on the marketplace row
    (``sync_status="error"``) and reflected in the returned ``SyncResult``.
    """
    try:
        entries, source_type = await _fetch_marketplace_entries(marketplace_row)
    except MarketplaceSyncError as exc:
        verbose_proxy_logger.warning("skill-marketplace-sync: failed to sync %r: %s", marketplace_row.name, exc)
        await _record_sync_failure(prisma_client, marketplace_row, str(exc))
        return SyncResult(status="error", error=str(exc), plugin_count=0)

    await _upsert_plugin_entries(prisma_client, marketplace_row.id, entries)
    await _soft_disable_stale_plugins(prisma_client, marketplace_row.id, entries)
    await _record_sync_success(prisma_client, marketplace_row, source_type)
    return SyncResult(status="success", error=None, plugin_count=len(entries))
