"""
CLAUDE CODE MARKETPLACE

Provides a registry/discovery layer for Claude Code plugins.
Plugins are stored as metadata + source references in LiteLLM database.
Actual plugin files are hosted on GitHub/GitLab/Bitbucket or as a zip archive on
any HTTPS host (S3, Artifactory, a static file server).

Endpoints:
/claude-code/marketplace.json  - GET  - List plugins for Claude Code discovery (unauthenticated; `?key=` adds the key's granted skills)
/claude-code/plugins           - POST - Register a new plugin (create-only, proxy admin only)
/claude-code/plugins           - GET  - List plugins visible to the key (enabled, plus granted disabled ones)
/claude-code/plugins/{name}    - GET  - Get plugin details (403 on a disabled plugin the key is not granted)
/claude-code/plugins/{name}    - PUT  - Update an existing plugin (proxy admin only)
/claude-code/plugins/{name}/enable  - POST - Enable a plugin (proxy admin only)
/claude-code/plugins/{name}/disable - POST - Disable a plugin (proxy admin only)
/claude-code/plugins/{name}    - DELETE - Delete a plugin (proxy admin only)
"""

import json
import re
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Annotated, Final, Protocol, TypedDict, get_args
from urllib.parse import urlencode, urlsplit

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from pydantic import TypeAdapter
from typing_extensions import ReadOnly

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import CommonProxyErrors, ProxyException, UserAPIKeyAuth
from litellm.proxy.anthropic_endpoints.claude_code_endpoints import claude_code_plugin_archive as plugin_archive
from litellm.proxy.anthropic_endpoints.claude_code_endpoints.claude_code_skill_access import (
    SkillVisibility,
    resolve_skill_visibility,
)
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.common_utils.resource_ownership import is_proxy_admin
from litellm.proxy.utils import get_custom_url
from litellm.repositories.table_repositories import ClaudeCodePluginRepository
from litellm.types.proxy.claude_code_endpoints import (
    InstallationPreference,
    ListPluginsResponse,
    PluginListItem,
    PluginResponse,
    PluginSpec,
    RegisterPluginRequest,
    RegisterPluginResponse,
    UpdatePluginRequest,
)

router: Final = APIRouter()


class _PluginRecord(Protocol):
    id: str
    name: str
    version: str | None
    description: str | None
    manifest_json: str | None
    enabled: bool
    created_at: datetime | None
    updated_at: datetime | None
    created_by: str | None


class _MarketplaceEntry(TypedDict, total=False):
    name: str
    source: object
    version: str
    description: str
    author: object
    homepage: object
    keywords: object
    category: object
    installationPreference: ReadOnly[InstallationPreference]


_INSTALLATION_PREFERENCES: Final[tuple[InstallationPreference, ...]] = get_args(InstallationPreference)
MANIFEST_FIELDS: Final = TypeAdapter(dict[str, object])
SOURCE_FIELDS: Final = TypeAdapter(dict[str, str])


_SELF_INSTALLING_PREFERENCES: Final[frozenset[InstallationPreference]] = frozenset({"auto_install", "required"})


def get_installation_preference(manifest: Mapping[str, object]) -> InstallationPreference | None:
    value: Final = manifest.get("installation_preference")
    if value is None or value in _INSTALLATION_PREFERENCES:
        return value
    verbose_proxy_logger.warning(
        "Plugin %s has unknown installation_preference %r, serving it as unset", manifest.get("name"), value
    )
    return None


async def get_prisma_client() -> object:
    """Get the prisma client from proxy_server."""
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(
            status_code=500,
            detail={"error": CommonProxyErrors.db_not_connected_error.value},
        )
    return prisma_client


_get_prisma_client: Final = get_prisma_client


def _archive_url(request: Request, plugin_name: str, key: str | None) -> str:
    """Same-origin URL of the plugin's ZIP; carries `?key=` so a granted disabled plugin stays fetchable."""
    base: Final = get_custom_url(str(request.base_url), route=None).rstrip("/")
    url: Final = f"{base}/claude-code/plugins/{plugin_name}/archive.zip"
    return f"{url}?{urlencode({'key': key})}" if key else url


def _hosted_archive_sha256(manifest: Mapping[str, object]) -> str | None:
    """The sha256 of the ZIP LiteLLM hosts for this plugin, or None when it isn't hosted."""
    digest: Final = manifest.get("archive_sha256")
    return digest if manifest.get("serve_archive") is True and isinstance(digest, str) else None


def _wants_hosted_archive(spec: PluginSpec) -> bool:
    return spec.source.get("source") == "archive" if spec.serve_archive is None else spec.serve_archive


@router.get(
    "/claude-code/marketplace.json",
    tags=["Claude Code Marketplace"],
)
async def get_marketplace(request: Request, key: str | None = None):
    """
    Serve marketplace.json for Claude Code plugin discovery.

    This endpoint is accessed by Claude Code CLI when users run:
    - claude plugin marketplace add <url>
    - claude plugin install <name>@<marketplace>

    Without `key` the catalog holds the enabled (public) plugins. With `?key=sk-...`
    the key is authenticated and the catalog also holds the disabled plugins granted
    to it through `object_permission.skills` or an access group on the key or its team.

    Returns:
        Marketplace catalog with list of available plugins and their git sources.

    Example:
        ```bash
        claude plugin marketplace add http://localhost:4000/claude-code/marketplace.json
        claude plugin marketplace add "http://localhost:4000/claude-code/marketplace.json?key=sk-..."
        claude plugin install my-plugin@litellm
        ```
    """
    try:
        prisma_client: Final = await get_prisma_client()

        caller: Final[UserAPIKeyAuth | None] = (
            await user_api_key_auth(request=request, api_key=f"Bearer {key}") if key else None
        )
        visibility: Final[SkillVisibility] = await resolve_skill_visibility(caller)
        plugins: Final[Sequence[_PluginRecord]] = await ClaudeCodePluginRepository(prisma_client).table.find_many(
            where=visibility.where()
        )

        plugin_list: Final = []
        for plugin in plugins:
            if not visibility.allows(plugin):
                continue
            try:
                manifest: Mapping[str, object] = json.loads(plugin.manifest_json or "{}")
            except json.JSONDecodeError:
                verbose_proxy_logger.warning("Plugin %s has invalid manifest JSON, skipping", plugin.name)
                continue

            # Source must be specified for URL-based marketplaces
            if "source" not in manifest:
                verbose_proxy_logger.warning("Plugin %s has no source field, skipping", plugin.name)
                continue

            hosted_sha256 = _hosted_archive_sha256(manifest)
            source: object = (
                {
                    "source": "archive",
                    "url": _archive_url(request, plugin.name, key),
                    "sha256": hosted_sha256,
                }
                if hosted_sha256
                else manifest["source"]
            )

            installation_preference = get_installation_preference(manifest)
            entry: _MarketplaceEntry = (
                {"name": plugin.name, "source": source}
                if installation_preference is None
                else {
                    "name": plugin.name,
                    "source": source,
                    "installationPreference": installation_preference,
                }
            )

            if plugin.version:
                entry["version"] = plugin.version
            if plugin.description:
                entry["description"] = plugin.description
            if "author" in manifest:
                entry["author"] = manifest["author"]
            if "homepage" in manifest:
                entry["homepage"] = manifest["homepage"]
            if "keywords" in manifest:
                entry["keywords"] = manifest["keywords"]
            if "category" in manifest:
                entry["category"] = manifest["category"]

            plugin_list.append(entry)

        marketplace: Final = {
            "name": "litellm",
            "owner": {"name": "LiteLLM", "email": "support@litellm.ai"},
            "plugins": plugin_list,
        }

        return JSONResponse(content=marketplace)

    except (HTTPException, ProxyException):
        raise
    except Exception as e:
        verbose_proxy_logger.exception("Error generating marketplace: %s", e)
        raise HTTPException(
            status_code=500,
            detail={"error": f"Failed to generate marketplace: {e}"},
        )


# Allowlist for git-subdir paths: one or more segments separated by '/'.
# Each segment must start with an alphanumeric character and contain only
# alphanumeric characters, dots, hyphens, and underscores.
# This implicitly blocks '..', leading '/', backslashes, and percent-encoded sequences.
_VALID_GIT_SUBDIR_PATH_RE: Final = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]*(/[a-zA-Z0-9][a-zA-Z0-9._-]*)*$")
_VALID_SHA256_RE: Final = re.compile(r"^[0-9a-fA-F]{64}$")


def _is_https_url_with_host(url: str) -> bool:
    try:
        parts: Final = urlsplit(url)
    except ValueError:
        return False
    return parts.scheme == "https" and bool(parts.hostname)


def _validate_plugin_source(source: Mapping[str, str]) -> None:
    """Validate plugin source format, raising HTTPException on invalid input."""
    source_type: Final = source.get("source")
    if source_type == "github":
        if "repo" not in source:
            raise HTTPException(
                status_code=400,
                detail={"error": "GitHub source must include 'repo' field (e.g., 'org/repo')"},
            )
    elif source_type == "url":
        if "url" not in source:
            raise HTTPException(
                status_code=400,
                detail={"error": "URL source must include 'url' field (e.g., 'https://github.com/org/repo.git')"},
            )
    elif source_type == "git-subdir":
        if not source.get("url"):
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "git-subdir source must include 'url' field (e.g., 'https://github.com/org/repo.git')"
                },
            )
        if not source.get("path"):
            raise HTTPException(
                status_code=400,
                detail={"error": "git-subdir source must include 'path' field (e.g., 'plugins/plugin-name')"},
            )
        if not _VALID_GIT_SUBDIR_PATH_RE.match(source["path"]):
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "git-subdir 'path' must be a relative path of the form 'segment/segment' (alphanumeric, dots, hyphens, underscores only)"
                },
            )
    elif source_type == "archive":
        if not _is_https_url_with_host(source.get("url", "")):
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "archive source must include an https 'url' field "
                    "(e.g., 'https://bucket.s3.amazonaws.com/plugins/plugin-name.zip')"
                },
            )
        if "sha256" in source and not _VALID_SHA256_RE.match(source["sha256"]):
            raise HTTPException(
                status_code=400,
                detail={"error": "archive 'sha256' must be a 64-character hex digest"},
            )
    else:
        raise HTTPException(
            status_code=400,
            detail={"error": "source.source must be 'github', 'url', 'git-subdir', or 'archive'"},
        )


def _validate_installation_preference(spec: PluginSpec) -> None:
    if spec.installation_preference is None:
        return
    hosted: Final = _wants_hosted_archive(spec)
    if spec.source.get("source") != "archive" and not hosted:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "installation_preference needs an archive source or serve_archive, the only plugins "
                "Claude Desktop lists from a marketplace.json URL"
            },
        )
    if spec.installation_preference in _SELF_INSTALLING_PREFERENCES and not (hosted or spec.source.get("sha256")):
        raise HTTPException(
            status_code=400,
            detail={
                "error": f"installation_preference '{spec.installation_preference}' needs the archive's sha256, "
                "since Claude Desktop installs only pinned archives on its own"
            },
        )


def _build_plugin_manifest(name: str, spec: PluginSpec) -> Mapping[str, object]:
    """Build the stored manifest dict shared by plugin create and update."""
    dumped: Final[Mapping[str, object]] = MappingProxyType(spec.model_dump(exclude_none=True))
    return {"name": name, **{key: value for key, value in dumped.items() if value and key != "name"}}


def _error_response(status_code: int, message: str) -> HTTPException:
    return HTTPException(status_code=status_code, detail={"error": message})


async def _build_manifest_with_archive(
    name: str, spec: PluginSpec
) -> tuple[Mapping[str, object], plugin_archive.BuiltArchive | None]:
    """The stored manifest, plus the freshly built ZIP when the plugin is hosted by LiteLLM.

    Fetching happens before the DB write, so a source that can't be mirrored is rejected at
    registration instead of leaving a catalog entry that 404s when Claude Desktop installs it.
    """
    manifest: Final = dict(_build_plugin_manifest(name, spec))
    if not _wants_hosted_archive(spec):
        return manifest, None
    try:
        built: Final = await plugin_archive.build_archive(spec.source)
    except plugin_archive.ArchiveBuildError as e:
        raise _error_response(400, f"Cannot host plugin '{name}' as an archive: {e}")
    manifest["serve_archive"] = True
    manifest["archive_sha256"] = built.sha256
    if built.ref:
        manifest["archive_ref"] = built.ref
    return manifest, built


async def _load_archive(plugin_name: str, manifest: Mapping[str, object], source: Mapping[str, str]) -> bytes:
    """The ZIP bytes for a hosted plugin, rebuilt from its source when the local file is gone."""
    expected: Final = _hosted_archive_sha256(manifest)
    if expected is None:
        raise _error_response(404, f"Plugin '{plugin_name}' has no hosted archive")

    stored: Final = plugin_archive.read_stored_archive(plugin_name)
    if stored is not None and plugin_archive.sha256_hex(stored) == expected:
        return stored

    async with plugin_archive.archive_lock(plugin_name):
        stored_again: Final = plugin_archive.read_stored_archive(plugin_name)
        if stored_again is not None and plugin_archive.sha256_hex(stored_again) == expected:
            return stored_again

        verbose_proxy_logger.info("Archive for plugin %s missing or stale, refetching from its source", plugin_name)
        pinned_ref: Final = manifest.get("archive_ref")
        try:
            rebuilt: Final = await plugin_archive.build_archive(
                source, pinned_ref=pinned_ref if isinstance(pinned_ref, str) else None
            )
        except plugin_archive.ArchiveBuildError as e:
            raise _error_response(502, f"Could not rebuild the archive for plugin '{plugin_name}': {e}")
        if rebuilt.sha256 != expected:
            raise _error_response(
                502,
                f"The source of plugin '{plugin_name}' no longer matches the archive advertised in the catalog; "
                "update the plugin to republish it",
            )
        plugin_archive.store_archive(plugin_name, rebuilt.data)
        return rebuilt.data


@router.get(
    "/claude-code/plugins/{plugin_name}/archive.zip",
    tags=["Claude Code Marketplace"],
    response_class=Response,
)
async def get_plugin_archive(plugin_name: str, request: Request, key: str | None = None) -> Response:
    """
    Download a plugin as a ZIP with the plugin at the archive root.

    This is the same-origin `archive` source Claude Desktop installs from; marketplace.json
    advertises its URL and sha256 for plugins registered with `serve_archive` (the default for
    `archive` sources). Like marketplace.json it is unauthenticated for enabled plugins, and
    `?key=sk-...` unlocks the disabled plugins granted to that key.

    The ZIP is built when the plugin is registered or updated and kept on this server. If the
    file is missing (new container, another replica) it is refetched from the plugin's source.
    """
    prisma_client: Final = await get_prisma_client()
    caller: Final[UserAPIKeyAuth | None] = (
        await user_api_key_auth(request=request, api_key=f"Bearer {key}") if key else None
    )
    plugin: Final[_PluginRecord | None] = await ClaudeCodePluginRepository(prisma_client).table.find_unique(
        where={"name": plugin_name}
    )
    if plugin is None or not (await resolve_skill_visibility(caller)).allows(plugin):
        raise _error_response(404, f"Plugin '{plugin_name}' not found")

    manifest: Final[Mapping[str, object]] = json.loads(plugin.manifest_json or "{}")
    raw_source: Final = manifest.get("source")
    if not isinstance(raw_source, dict):
        raise _error_response(404, f"Plugin '{plugin_name}' has no hosted archive")

    data: Final = await _load_archive(plugin_name, manifest, SOURCE_FIELDS.validate_python(raw_source))
    return Response(
        content=data,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{plugin_name}.zip"'},
    )


def _name_conflict_error(name: str) -> HTTPException:
    return _error_response(
        409, f"A skill named '{name}' already exists. Update the existing skill instead of adding it again."
    )


def _require_proxy_admin(user_api_key_dict: UserAPIKeyAuth) -> None:
    """Catalog mutations are restricted to proxy admins: marketplace.json is served
    unauthenticated and any registered/updated entry is immediately installable by
    every user, so a non-admin key must never be able to add or overwrite one.
    """
    if not is_proxy_admin(user_api_key_dict):
        raise HTTPException(
            status_code=403,
            detail={"error": "Only proxy admins may modify the Claude Code plugin marketplace."},
        )


@router.post(
    "/claude-code/plugins",
    tags=["Claude Code Marketplace"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=RegisterPluginResponse,
)
async def register_plugin(
    request: RegisterPluginRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Register a new plugin in the LiteLLM marketplace.

    LiteLLM acts as a registry/discovery layer. Plugins are hosted on
    GitHub/GitLab/Bitbucket or as a zip archive on any https host (e.g. S3).
    Claude Code clones the git source or downloads the archive when users install.

    This endpoint is create-only and never overwrites. If a plugin with
    the same name already exists it returns 409 Conflict; use
    PUT /claude-code/plugins/{plugin_name} to update an existing plugin.

    Requires a proxy admin API key.

    Parameters:
        - name: Plugin name (kebab-case)
        - source: Plugin source reference (github, url, git-subdir, or archive format)
        - version: Semantic version (optional)
        - description: Plugin description (optional)
        - author: Author information (optional)
        - homepage: Plugin homepage URL (optional)
        - keywords: Search keywords (optional)
        - category: Plugin category (optional)
        - installation_preference: 'available', 'auto_install', or 'required' (optional)

    Returns:
        Registration status (action is always "created") and plugin information.

    Example:
        ```bash
        curl -X POST http://localhost:4000/claude-code/plugins \\
          -H "Authorization: Bearer sk-..." \\
          -H "Content-Type: application/json" \\
          -d '{
            "name": "my-plugin",
            "source": {"source": "github", "repo": "org/my-plugin"},
            "version": "1.0.0",
            "description": "My awesome plugin"
          }'
        ```
    """
    from prisma.errors import UniqueViolationError

    try:
        _require_proxy_admin(user_api_key_dict)

        prisma_client: Final = await get_prisma_client()

        if not re.match(r"^[a-z0-9-]+$", request.name):
            raise HTTPException(
                status_code=400,
                detail={"error": "Plugin name must be kebab-case (lowercase letters, numbers, hyphens)"},
            )

        _validate_plugin_source(request.source)
        _validate_installation_preference(request)

        existing: Final[_PluginRecord | None] = await ClaudeCodePluginRepository(prisma_client).table.find_unique(
            where={"name": request.name}
        )
        if existing:
            raise _name_conflict_error(request.name)

        manifest, built = await _build_manifest_with_archive(request.name, request)

        try:
            plugin: Final[_PluginRecord] = await ClaudeCodePluginRepository(prisma_client).table.create(
                data={
                    "name": request.name,
                    "version": request.version,
                    "description": request.description,
                    "manifest_json": json.dumps(manifest),
                    "files_json": "{}",
                    "enabled": True,
                    "created_at": datetime.now(timezone.utc),
                    "updated_at": datetime.now(timezone.utc),
                    "created_by": user_api_key_dict.user_id,
                }
            )
        except UniqueViolationError:
            raise _name_conflict_error(request.name)

        if built is not None:
            plugin_archive.store_archive(request.name, built.data)

        verbose_proxy_logger.info("Plugin %s created successfully", request.name)

        return RegisterPluginResponse(
            status="success",
            action="created",
            plugin=PluginResponse(
                id=plugin.id,
                name=plugin.name,
                version=plugin.version,
                description=plugin.description,
                source=request.source,
                installation_preference=request.installation_preference,
                serve_archive=built is not None,
                enabled=plugin.enabled,
            ),
        )

    except HTTPException:
        raise
    except Exception as e:
        verbose_proxy_logger.exception("Error registering plugin: %s", e)
        raise HTTPException(
            status_code=500,
            detail={"error": f"Registration failed: {e}"},
        )


@router.get(
    "/claude-code/plugins",
    tags=["Claude Code Marketplace"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=ListPluginsResponse,
)
async def list_plugins(
    enabled_only: bool = False,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    List all plugins in the marketplace.

    Parameters:
        - enabled_only: If true, only return enabled plugins

    Returns:
        List of plugins with their metadata.
    """
    try:
        prisma_client: Final = await get_prisma_client()

        visibility: Final[SkillVisibility] = await resolve_skill_visibility(user_api_key_dict)
        plugins: Final[Sequence[_PluginRecord]] = await ClaudeCodePluginRepository(prisma_client).table.find_many(
            where={"enabled": True} if enabled_only else visibility.where()
        )

        plugin_list: Final = []
        for p in plugins:
            if not visibility.allows(p):
                continue
            manifest = json.loads(p.manifest_json) if p.manifest_json else {}

            plugin_list.append(
                PluginListItem(
                    id=p.id,
                    name=p.name,
                    version=p.version,
                    description=p.description,
                    source=manifest.get("source", {}),
                    author=manifest.get("author"),
                    homepage=manifest.get("homepage"),
                    keywords=manifest.get("keywords"),
                    category=manifest.get("category"),
                    domain=manifest.get("domain"),
                    namespace=manifest.get("namespace"),
                    installation_preference=get_installation_preference(MANIFEST_FIELDS.validate_python(manifest)),
                    serve_archive=_hosted_archive_sha256(MANIFEST_FIELDS.validate_python(manifest)) is not None,
                    enabled=p.enabled,
                    created_at=p.created_at.isoformat() if p.created_at else None,
                    updated_at=p.updated_at.isoformat() if p.updated_at else None,
                )
            )

        # Sort by created_at descending (newest first)
        plugin_list.sort(key=lambda x: x.created_at or "", reverse=True)

        return ListPluginsResponse(
            plugins=plugin_list,
            count=len(plugin_list),
        )

    except HTTPException:
        raise
    except Exception as e:
        verbose_proxy_logger.exception("Error listing plugins: %s", e)
        raise HTTPException(
            status_code=500,
            detail={"error": str(e)},
        )


@router.get(
    "/claude-code/plugins/{plugin_name}",
    tags=["Claude Code Marketplace"],
    dependencies=[Depends(user_api_key_auth)],
)
async def get_plugin(
    plugin_name: str,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Get details of a specific plugin.

    Parameters:
        - plugin_name: The name of the plugin

    Returns:
        Plugin details including source and metadata.
    """
    try:
        prisma_client: Final = await get_prisma_client()

        plugin: Final[_PluginRecord | None] = await ClaudeCodePluginRepository(prisma_client).table.find_unique(
            where={"name": plugin_name}
        )

        if not plugin:
            raise HTTPException(
                status_code=404,
                detail={"error": f"Plugin '{plugin_name}' not found"},
            )

        if not (await resolve_skill_visibility(user_api_key_dict)).allows(plugin):
            raise HTTPException(
                status_code=403,
                detail={"error": f"Plugin '{plugin_name}' is not granted to this key"},
            )

        manifest: Final[Mapping[str, object]] = json.loads(plugin.manifest_json or "{}") if plugin.manifest_json else {}

        return {
            "id": plugin.id,
            "name": plugin.name,
            "version": plugin.version,
            "description": plugin.description,
            "source": manifest.get("source"),
            "author": manifest.get("author"),
            "homepage": manifest.get("homepage"),
            "keywords": manifest.get("keywords"),
            "category": manifest.get("category"),
            "installation_preference": get_installation_preference(manifest),
            "serve_archive": _hosted_archive_sha256(manifest) is not None,
            "enabled": plugin.enabled,
            "created_at": plugin.created_at.isoformat() if plugin.created_at else None,
            "updated_at": plugin.updated_at.isoformat() if plugin.updated_at else None,
            "created_by": plugin.created_by,
        }

    except HTTPException:
        raise
    except Exception as e:
        verbose_proxy_logger.exception("Error getting plugin: %s", e)
        raise HTTPException(
            status_code=500,
            detail={"error": str(e)},
        )


@router.put(
    "/claude-code/plugins/{plugin_name}",
    tags=["Claude Code Marketplace"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=RegisterPluginResponse,
)
async def update_plugin(
    plugin_name: str,
    request: UpdatePluginRequest,
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
):
    """
    Update an existing plugin in the LiteLLM marketplace.

    The plugin is identified by its name in the path, which is the resource
    identity and cannot be changed here. This is a full replace, not a merge:
    the manifest is rebuilt from the request body, so any optional field left
    out is reset to its default (e.g. an omitted version is cleared, not kept).
    Send the full desired state.

    Returns 404 if no plugin with the given name exists; use
    POST /claude-code/plugins to create a new plugin.

    Requires a proxy admin API key.

    Parameters:
        - plugin_name: Name of the plugin to update (path parameter)
        - source: Plugin source reference (github, url, git-subdir, or archive format)
        - version: Semantic version (optional)
        - description: Plugin description (optional)
        - author: Author information (optional)
        - homepage: Plugin homepage URL (optional)
        - keywords: Search keywords (optional)
        - category: Plugin category (optional)
        - installation_preference: 'available', 'auto_install', or 'required' (optional)

    Returns:
        Update status (action is always "updated") and plugin information.

    Example:
        ```bash
        curl -X PUT http://localhost:4000/claude-code/plugins/my-plugin \\
          -H "Authorization: Bearer sk-..." \\
          -H "Content-Type: application/json" \\
          -d '{
            "source": {"source": "github", "repo": "org/my-plugin"},
            "version": "2.0.0",
            "description": "My awesome plugin"
          }'
        ```
    """
    from prisma.errors import PrismaError

    try:
        _require_proxy_admin(user_api_key_dict)

        prisma_client: Final = await get_prisma_client()

        _validate_plugin_source(request.source)
        _validate_installation_preference(request)

        existing: Final[_PluginRecord | None] = await ClaudeCodePluginRepository(prisma_client).table.find_unique(
            where={"name": plugin_name}
        )
        if not existing:
            raise _error_response(404, f"Plugin '{plugin_name}' not found")

        manifest, built = await _build_manifest_with_archive(plugin_name, request)

        plugin: Final[_PluginRecord | None] = await ClaudeCodePluginRepository(prisma_client).table.update(
            where={"name": plugin_name},
            data={
                "version": request.version,
                "description": request.description,
                "manifest_json": json.dumps(manifest),
                "files_json": "{}",
                "updated_at": datetime.now(timezone.utc),
            },
        )
        if plugin is None:
            raise _error_response(404, f"Plugin '{plugin_name}' not found")

        if built is not None:
            plugin_archive.store_archive(plugin_name, built.data)
        else:
            plugin_archive.delete_stored_archive(plugin_name)

        verbose_proxy_logger.info("Plugin %s updated successfully", plugin_name)

        return RegisterPluginResponse(
            status="success",
            action="updated",
            plugin=PluginResponse(
                id=plugin.id,
                name=plugin.name,
                version=plugin.version,
                description=plugin.description,
                source=request.source,
                installation_preference=request.installation_preference,
                serve_archive=built is not None,
                enabled=plugin.enabled,
            ),
        )

    except HTTPException:
        raise
    except PrismaError as e:
        verbose_proxy_logger.exception("Error updating plugin: %s", e)
        raise _error_response(500, f"Update failed: {e}")


@router.post(
    "/claude-code/plugins/{plugin_name}/enable",
    tags=["Claude Code Marketplace"],
    dependencies=[Depends(user_api_key_auth)],
)
async def enable_plugin(
    plugin_name: str,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Enable a disabled plugin.

    Requires a proxy admin API key.

    Parameters:
        - plugin_name: The name of the plugin to enable
    """
    try:
        _require_proxy_admin(user_api_key_dict)

        prisma_client: Final = await get_prisma_client()

        plugin: Final[_PluginRecord | None] = await ClaudeCodePluginRepository(prisma_client).table.find_unique(
            where={"name": plugin_name}
        )
        if not plugin:
            raise HTTPException(
                status_code=404,
                detail={"error": f"Plugin '{plugin_name}' not found"},
            )

        await ClaudeCodePluginRepository(prisma_client).table.update(
            where={"name": plugin_name},
            data={"enabled": True, "updated_at": datetime.now(timezone.utc)},
        )

        verbose_proxy_logger.info("Plugin %s enabled", plugin_name)
        return {"status": "success", "message": f"Plugin '{plugin_name}' enabled"}

    except HTTPException:
        raise
    except Exception as e:
        verbose_proxy_logger.exception("Error enabling plugin: %s", e)
        raise HTTPException(
            status_code=500,
            detail={"error": str(e)},
        )


@router.post(
    "/claude-code/plugins/{plugin_name}/disable",
    tags=["Claude Code Marketplace"],
    dependencies=[Depends(user_api_key_auth)],
)
async def disable_plugin(
    plugin_name: str,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Disable a plugin without deleting it.

    Requires a proxy admin API key.

    Parameters:
        - plugin_name: The name of the plugin to disable
    """
    try:
        _require_proxy_admin(user_api_key_dict)

        prisma_client: Final = await get_prisma_client()

        plugin: Final[_PluginRecord | None] = await ClaudeCodePluginRepository(prisma_client).table.find_unique(
            where={"name": plugin_name}
        )
        if not plugin:
            raise HTTPException(
                status_code=404,
                detail={"error": f"Plugin '{plugin_name}' not found"},
            )

        await ClaudeCodePluginRepository(prisma_client).table.update(
            where={"name": plugin_name},
            data={"enabled": False, "updated_at": datetime.now(timezone.utc)},
        )

        verbose_proxy_logger.info("Plugin %s disabled", plugin_name)
        return {"status": "success", "message": f"Plugin '{plugin_name}' disabled"}

    except HTTPException:
        raise
    except Exception as e:
        verbose_proxy_logger.exception("Error disabling plugin: %s", e)
        raise HTTPException(
            status_code=500,
            detail={"error": str(e)},
        )


@router.delete(
    "/claude-code/plugins/{plugin_name}",
    tags=["Claude Code Marketplace"],
    dependencies=[Depends(user_api_key_auth)],
)
async def delete_plugin(
    plugin_name: str,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Delete a plugin from the marketplace.

    Requires a proxy admin API key.

    Parameters:
        - plugin_name: The name of the plugin to delete
    """
    try:
        _require_proxy_admin(user_api_key_dict)

        prisma_client: Final = await get_prisma_client()

        plugin: Final[_PluginRecord | None] = await ClaudeCodePluginRepository(prisma_client).table.find_unique(
            where={"name": plugin_name}
        )
        if not plugin:
            raise HTTPException(
                status_code=404,
                detail={"error": f"Plugin '{plugin_name}' not found"},
            )

        await ClaudeCodePluginRepository(prisma_client).table.delete(where={"name": plugin_name})
        plugin_archive.delete_stored_archive(plugin_name)

        verbose_proxy_logger.info("Plugin %s deleted", plugin_name)
        return {"status": "success", "message": f"Plugin '{plugin_name}' deleted"}

    except HTTPException:
        raise
    except Exception as e:
        verbose_proxy_logger.exception("Error deleting plugin: %s", e)
        raise HTTPException(
            status_code=500,
            detail={"error": str(e)},
        )
