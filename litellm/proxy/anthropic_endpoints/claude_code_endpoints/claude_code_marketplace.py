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

from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import json
import re
import zipfile
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import PurePosixPath, PureWindowsPath
from typing import TYPE_CHECKING, Annotated, Final, Protocol, TypedDict, TypeVar, get_args
from urllib.parse import quote, urlsplit

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from pydantic import TypeAdapter, ValidationError
from typing_extensions import ReadOnly

from litellm._logging import verbose_proxy_logger
from litellm.constants import CLAUDE_CODE_PLUGIN_ARCHIVE_MAX_BYTES
from litellm.proxy._types import CommonProxyErrors, ProxyException, UserAPIKeyAuth
from litellm.proxy.anthropic_endpoints.claude_code_endpoints.claude_code_skill_access import (
    SkillVisibility,
    resolve_skill_visibility,
)
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.common_utils.resource_ownership import is_proxy_admin
from litellm.repositories.table_repositories import ClaudeCodePluginArchiveRepository, ClaudeCodePluginRepository
from litellm.types.proxy.claude_code_endpoints import (
    InstallationPreference,
    ListPluginsResponse,
    PluginListItem,
    PluginResponse,
    PluginSpec,
    RegisterPluginRequest,
    RegisterPluginResponse,
    UpdatePluginRequest,
    UploadPluginArchiveResponse,
)

if TYPE_CHECKING:
    from litellm.proxy.utils import PrismaClient


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
    archive_sha256: str | None


class _PluginArchiveRecord(Protocol):
    def model_dump(self) -> Mapping[str, object]: ...


class _MarketplaceEntry(TypedDict, total=False):
    name: ReadOnly[str]
    source: ReadOnly[object]
    installationPreference: ReadOnly[InstallationPreference]
    version: ReadOnly[str]
    description: ReadOnly[str]
    author: ReadOnly[object]
    homepage: ReadOnly[object]
    keywords: ReadOnly[object]
    category: ReadOnly[object]


_INSTALLATION_PREFERENCES: Final[tuple[InstallationPreference, ...]] = get_args(InstallationPreference)
MARKETPLACE_ENTRY_FIELDS: Final = TypeAdapter(_MarketplaceEntry)
MANIFEST_FIELDS: Final = TypeAdapter(dict[str, object])
PLUGIN_LIST_ITEM_FIELDS: Final = TypeAdapter(PluginListItem)


_SELF_INSTALLING_PREFERENCES: Final[frozenset[InstallationPreference]] = frozenset({"auto_install", "required"})
_EncodedArchive = TypeVar("_EncodedArchive")  # rebind-ok: generic encoder type must be module scoped


def get_installation_preference(manifest: Mapping[str, object]) -> InstallationPreference | None:
    value: Final = manifest.get("installation_preference")
    if value is None or value in _INSTALLATION_PREFERENCES:
        return value
    verbose_proxy_logger.warning(
        "Plugin %s has unknown installation_preference %r, serving it as unset", manifest.get("name"), value
    )
    return None


async def get_prisma_client() -> PrismaClient:
    """Get the prisma client from proxy_server."""
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(
            status_code=500,
            detail={"error": CommonProxyErrors.db_not_connected_error.value},
        )
    return prisma_client


def _hosted_marketplace_source(
    plugin: _PluginRecord,
    marketplace_base_url: str,
) -> object | None:
    if plugin.archive_sha256 is None:
        verbose_proxy_logger.warning("Hosted plugin %s has no archive, skipping", plugin.name)
        return None

    return {
        "source": "archive",
        "url": f"{marketplace_base_url}/claude-code/plugins/{quote(plugin.name, safe='')}/archive.zip",
        "sha256": plugin.archive_sha256,
    }


def _marketplace_entry(
    plugin: _PluginRecord,
    marketplace_base_url: str,
    visibility: SkillVisibility,
) -> _MarketplaceEntry | None:
    if not visibility.allows(plugin):
        return None
    try:
        manifest: Final[Mapping[str, object]] = MANIFEST_FIELDS.validate_json(plugin.manifest_json or "{}")
    except ValidationError:
        verbose_proxy_logger.warning("Plugin %s has invalid manifest JSON, skipping", plugin.name)
        return None

    if "source" not in manifest:
        verbose_proxy_logger.warning("Plugin %s has no source field, skipping", plugin.name)
        return None

    source: Final = manifest["source"]
    marketplace_source: Final[object] = (
        _hosted_marketplace_source(plugin, marketplace_base_url) if source == {"source": "hosted"} else source
    )
    if marketplace_source is None:
        return None

    installation_preference: Final = get_installation_preference(manifest)
    return MARKETPLACE_ENTRY_FIELDS.validate_python(
        {
            "name": plugin.name,
            "source": marketplace_source,
            **({"installationPreference": installation_preference} if installation_preference is not None else {}),
            **({"version": plugin.version} if plugin.version else {}),
            **({"description": plugin.description} if plugin.description else {}),
            **({"author": manifest["author"]} if "author" in manifest else {}),
            **({"homepage": manifest["homepage"]} if "homepage" in manifest else {}),
            **({"keywords": manifest["keywords"]} if "keywords" in manifest else {}),
            **({"category": manifest["category"]} if "category" in manifest else {}),
        }
    )


async def _update_plugin_and_delete_archive(
    prisma_client: PrismaClient,
    plugin_id: str,
    plugin_name: str,
    update_data: Mapping[str, object],
) -> _PluginRecord | None:
    async with prisma_client.db.tx() as transaction:
        await transaction.litellm_claudecodepluginarchivetable.delete_many(where={"plugin_id": plugin_id})
        return await transaction.litellm_claudecodeplugintable.update(
            where={"name": plugin_name},
            data=update_data,
        )


_get_prisma_client: Final = get_prisma_client


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
        from litellm.proxy._experimental.mcp_server.oauth_utils import get_request_base_url

        marketplace_base_url: Final = get_request_base_url(request)

        caller: Final[UserAPIKeyAuth | None] = await _resolve_marketplace_caller(request, key)
        visibility: Final[SkillVisibility] = await resolve_skill_visibility(caller)
        plugins: Final[Sequence[_PluginRecord]] = await ClaudeCodePluginRepository(prisma_client).table.find_many(
            where=visibility.where()
        )
        plugin_entries: Final = tuple(
            _marketplace_entry(plugin, marketplace_base_url, visibility) for plugin in plugins
        )
        plugin_list: Final = tuple(entry for entry in plugin_entries if entry is not None)

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


async def _resolve_marketplace_caller(request: Request, key: str | None) -> UserAPIKeyAuth | None:
    if key:
        return await user_api_key_auth(request=request, api_key=f"Bearer {key}")
    authorization: Final = request.headers.get("Authorization")
    if isinstance(authorization, str) and authorization:
        return await user_api_key_auth(request=request, api_key=authorization)
    return None


def _archive_bytes(archive: _PluginArchiveRecord) -> bytes:
    content: Final = archive.model_dump().get("content")
    if isinstance(content, str):
        return base64.b64decode(content, validate=True)
    if isinstance(content, bytes):
        return content
    raise ValueError("Stored plugin archive content is not bytes")


def _archive_not_found() -> HTTPException:
    return _error_response(404, "Plugin not found")


def _validate_and_hash_plugin_archive(content: bytes) -> str:
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            paths: Final = tuple(PurePosixPath(info.filename.replace("\\", "/")) for info in archive.infolist())
    except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
        raise ValueError("Uploaded file is not a valid zip archive") from exc

    if any(
        path.is_absolute()
        or ".." in path.parts
        or PureWindowsPath(path.as_posix()).is_absolute()
        or bool(PureWindowsPath(path.as_posix()).drive)
        for path in paths
    ):
        raise ValueError("Archive paths must not be absolute or contain '..' segments")

    if not any(
        path.parts == (".claude-plugin", "plugin.json")
        or (len(path.parts) == 3 and path.parts[1:] == (".claude-plugin", "plugin.json"))
        for path in paths
    ):
        raise ValueError(
            "Archive must contain .claude-plugin/plugin.json at the root or under exactly one top-level folder"
        )

    return hashlib.sha256(content).hexdigest()


def _validate_and_encode_plugin_archive(
    content: bytes,
    encode: Callable[[bytes], _EncodedArchive],
) -> tuple[str, _EncodedArchive]:
    return _validate_and_hash_plugin_archive(content), encode(content)


async def _read_plugin_archive_body(request: Request) -> bytes:
    chunks: tuple[bytes, ...] = ()  # rebind-ok: request bytes must be accumulated within the size limit
    total_bytes: int = 0  # rebind-ok: request bytes must be counted while streaming
    async for chunk in request.stream():
        if total_bytes + len(chunk) > CLAUDE_CODE_PLUGIN_ARCHIVE_MAX_BYTES:
            raise _error_response(413, "Plugin archive exceeds the maximum allowed size")
        total_bytes += len(chunk)
        chunks += (chunk,)
    return b"".join(chunks)


def _plugin_list_item(plugin: _PluginRecord) -> PluginListItem:
    manifest: Final[Mapping[str, object]] = json.loads(plugin.manifest_json) if plugin.manifest_json else {}
    return PLUGIN_LIST_ITEM_FIELDS.validate_python(
        {
            "id": plugin.id,
            "name": plugin.name,
            "version": plugin.version,
            "description": plugin.description,
            "source": manifest.get("source", {}),
            "author": manifest.get("author"),
            "homepage": manifest.get("homepage"),
            "keywords": manifest.get("keywords"),
            "category": manifest.get("category"),
            "domain": manifest.get("domain"),
            "namespace": manifest.get("namespace"),
            "installation_preference": get_installation_preference(manifest),
            "archive_sha256": (plugin.archive_sha256 if manifest.get("source") == {"source": "hosted"} else None),
            "enabled": plugin.enabled,
            "created_at": plugin.created_at.isoformat() if plugin.created_at else None,
            "updated_at": plugin.updated_at.isoformat() if plugin.updated_at else None,
        }
    )


@router.get(
    "/claude-code/plugins/{plugin_name}/archive.zip",
    tags=["Claude Code Marketplace"],
)
async def download_plugin_archive(
    plugin_name: str,
    request: Request,
    key: str | None = None,
) -> Response:
    """Download a hosted plugin archive when it is visible to the caller."""
    try:
        prisma_client: Final = await get_prisma_client()
        caller: Final = await _resolve_marketplace_caller(request, key)
        visibility: Final = await resolve_skill_visibility(caller)
        plugin: Final[_PluginRecord | None] = await ClaudeCodePluginRepository(prisma_client).table.find_unique(
            where={"name": plugin_name}
        )
        if plugin is None or not visibility.allows(plugin):
            raise _archive_not_found()

        try:
            manifest_value: Final[Mapping[str, object]] = MANIFEST_FIELDS.validate_json(plugin.manifest_json or "{}")
        except ValidationError:
            raise _archive_not_found()
        if manifest_value.get("source") != {"source": "hosted"} or plugin.archive_sha256 is None:
            raise _archive_not_found()

        archive: Final[_PluginArchiveRecord | None] = await ClaudeCodePluginArchiveRepository(
            prisma_client
        ).table.find_unique(where={"plugin_id": plugin.id})
        if archive is None:
            raise _archive_not_found()

        return Response(
            content=_archive_bytes(archive),
            media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="{plugin.name}.zip"'},
        )
    except HTTPException:
        raise
    except ProxyException:
        raise
    except Exception as e:  # noqa: BLE001  # unexpected download errors map to HTTP 500
        verbose_proxy_logger.exception("Error downloading plugin archive: %s", e)
        raise HTTPException(status_code=500, detail={"error": str(e)})


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
    elif source_type == "hosted":
        if set(source) != {"source"}:
            raise HTTPException(status_code=400, detail={"error": "hosted source must contain only 'source'"})
    else:
        raise HTTPException(
            status_code=400,
            detail={"error": "source.source must be 'github', 'url', 'git-subdir', 'archive', or 'hosted'"},
        )


def _validate_installation_preference(spec: PluginSpec) -> None:
    if spec.installation_preference is None:
        return
    source_type: Final = spec.source.get("source")
    if source_type not in {"archive", "hosted"}:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "installation_preference needs an archive source, the only plugins Claude Desktop lists "
                "from a marketplace.json URL"
            },
        )
    if source_type == "hosted":
        return
    if spec.installation_preference in _SELF_INSTALLING_PREFERENCES and not spec.source.get("sha256"):
        raise HTTPException(
            status_code=400,
            detail={
                "error": f"installation_preference '{spec.installation_preference}' needs the archive's sha256, "
                "since Claude Desktop installs only pinned archives on its own"
            },
        )


def _build_plugin_manifest(name: str, spec: PluginSpec) -> Mapping[str, object]:
    """Build the stored manifest dict shared by plugin create and update."""
    dumped: Final[Mapping[str, object]] = spec.model_dump(exclude_none=True)
    return {"name": name, **{key: value for key, value in dumped.items() if value and key != "name"}}


def _error_response(status_code: int, message: str) -> HTTPException:
    return HTTPException(status_code=status_code, detail={"error": message})


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

        manifest: Final[Mapping[str, object]] = _build_plugin_manifest(request.name, request)

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

        plugin_list: Final = [_plugin_list_item(p) for p in plugins if visibility.allows(p)]

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
            "archive_sha256": plugin.archive_sha256 if manifest.get("source") == {"source": "hosted"} else None,
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

        existing_manifest: Final[Mapping[str, object]] = MANIFEST_FIELDS.validate_json(existing.manifest_json or "{}")
        clear_archive: Final = existing_manifest.get("source") == {"source": "hosted"} and request.source != {
            "source": "hosted"
        }
        manifest: Final[Mapping[str, object]] = _build_plugin_manifest(plugin_name, request)
        update_data: Final = {
            "version": request.version,
            "description": request.description,
            "manifest_json": json.dumps(manifest),
            "files_json": "{}",
            "updated_at": datetime.now(timezone.utc),
            **({"archive_sha256": None} if clear_archive else {}),
        }
        plugin: Final[_PluginRecord | None] = (
            await _update_plugin_and_delete_archive(prisma_client, existing.id, plugin_name, update_data)
            if clear_archive
            else await ClaudeCodePluginRepository(prisma_client).table.update(
                where={"name": plugin_name},
                data=update_data,
            )
        )
        if plugin is None:
            raise _error_response(404, f"Plugin '{plugin_name}' not found")

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
                enabled=plugin.enabled,
            ),
        )

    except HTTPException:
        raise
    except PrismaError as e:
        verbose_proxy_logger.exception("Error updating plugin: %s", e)
        raise _error_response(500, f"Update failed: {e}")


@router.put(
    "/claude-code/plugins/{plugin_name}/archive",
    tags=["Claude Code Marketplace"],
    response_model=UploadPluginArchiveResponse,
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {
                "application/zip": {"schema": {"type": "string", "format": "binary"}},
                "application/octet-stream": {"schema": {"type": "string", "format": "binary"}},
            },
        }
    },
)
async def upload_plugin_archive(
    plugin_name: str,
    request: Request,
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
) -> UploadPluginArchiveResponse:
    """Upload and validate a zip archive for an existing hosted plugin."""
    try:
        _require_proxy_admin(user_api_key_dict)
        prisma_client: Final = await get_prisma_client()
        plugin: Final[_PluginRecord | None] = await ClaudeCodePluginRepository(prisma_client).table.find_unique(
            where={"name": plugin_name}
        )
        if plugin is None:
            raise _error_response(404, f"Plugin '{plugin_name}' not found")

        if (content_length := request.headers.get("content-length")) is not None:
            try:
                declared_length: Final = int(content_length)
            except ValueError:
                raise _error_response(400, "Invalid Content-Length") from None
            if declared_length < 0:
                raise _error_response(400, "Invalid Content-Length")
            if declared_length > CLAUDE_CODE_PLUGIN_ARCHIVE_MAX_BYTES:
                raise _error_response(413, "Plugin archive exceeds the maximum allowed size")

        content_type: Final = request.headers.get("content-type", "").split(";", maxsplit=1)[0].strip().lower()
        if content_type not in ("application/zip", "application/octet-stream"):
            raise _error_response(415, "Content-Type must be application/zip or application/octet-stream")

        content: Final[bytes] = await _read_plugin_archive_body(request)
        from prisma.fields import Base64

        try:
            sha256, encoded_content = await asyncio.to_thread(
                _validate_and_encode_plugin_archive, content, Base64.encode
            )
        except ValueError as exc:
            raise _error_response(400, str(exc)) from exc

        manifest: Final[Mapping[str, object]] = MANIFEST_FIELDS.validate_json(plugin.manifest_json or "{}")
        hosted_manifest: Final = {**manifest, "source": {"source": "hosted"}}
        updated_at: Final = datetime.now(timezone.utc)
        async with prisma_client.db.tx() as transaction:
            await transaction.litellm_claudecodepluginarchivetable.upsert(
                where={"plugin_id": plugin.id},
                data={
                    "create": {
                        "plugin_id": plugin.id,
                        "content": encoded_content,
                        "size_bytes": len(content),
                        "updated_at": updated_at,
                    },
                    "update": {
                        "content": encoded_content,
                        "size_bytes": len(content),
                        "updated_at": updated_at,
                    },
                },
            )
            updated_plugin: Final[_PluginRecord | None] = await transaction.litellm_claudecodeplugintable.update(
                where={"name": plugin_name},
                data={
                    "manifest_json": json.dumps(hosted_manifest),
                    "archive_sha256": sha256,
                    "updated_at": updated_at,
                },
            )
        if updated_plugin is None:
            raise _error_response(404, f"Plugin '{plugin_name}' not found")

        return UploadPluginArchiveResponse(
            plugin=PluginResponse(
                id=updated_plugin.id,
                name=updated_plugin.name,
                version=updated_plugin.version,
                description=updated_plugin.description,
                source={"source": "hosted"},
                installation_preference=get_installation_preference(hosted_manifest),
                enabled=updated_plugin.enabled,
            ),
            sha256=sha256,
            size_bytes=len(content),
        )
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001  # unexpected upload errors map to HTTP 500
        verbose_proxy_logger.exception("Error uploading plugin archive: %s", e)
        raise HTTPException(status_code=500, detail={"error": str(e)})


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
