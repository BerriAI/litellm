"""
CLAUDE CODE MARKETPLACE

Provides a registry/discovery layer for Claude Code plugins.
Plugins are stored as metadata + source references in LiteLLM database.
Actual plugin files are hosted on GitHub/GitLab/Bitbucket or as a zip archive on
any HTTPS host (S3, Artifactory, a static file server).

Endpoints:
/claude-code/marketplace.json  - GET  - List plugins for Claude Code discovery (anonymous unless
                                 general_settings.claude_code_marketplace_auth_required; a key adds its granted skills)
/claude-code/plugins           - POST - Register a new plugin (create-only, proxy admin only)
/claude-code/plugins           - GET  - List plugins visible to the key (enabled, plus granted disabled ones)
/claude-code/plugins/{name}    - GET  - Get plugin details (403 on a disabled plugin the key is not granted)
/claude-code/plugins/{name}    - PUT  - Update an existing plugin (proxy admin only)
/claude-code/plugins/{name}/enable  - POST - Enable a plugin (proxy admin only)
/claude-code/plugins/{name}/disable - POST - Disable a plugin (proxy admin only)
/claude-code/plugins/{name}    - DELETE - Delete a plugin (proxy admin only)
"""

import asyncio
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Annotated, Final, Protocol, TypedDict, get_args

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import JSONResponse, Response
from pydantic import Json, TypeAdapter
from typing_extensions import ReadOnly, assert_never

from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.url_utils import is_https_url_with_host
from litellm.models.plugin_upload import (
    PluginDownloader,
    PluginDownloadResult,
    PluginDownloadsResponse,
    PluginPublication,
    PluginPublicationBase,
    PluginPublicationError,
    has_agent_skill_metadata,
)
from litellm.proxy._types import CommonProxyErrors, ProxyErrorTypes, ProxyException, UserAPIKeyAuth
from litellm.proxy.anthropic_endpoints.claude_code_endpoints.claude_code_skill_access import (
    SkillVisibility,
    resolve_skill_visibility,
)
from litellm.proxy.anthropic_endpoints.claude_code_endpoints.skill_upload_archive import (
    MAX_SKILL_UPLOAD_BYTES,
    build_uploaded_skill_archive,
)
from litellm.proxy.auth.auth_exception_handler import DB_UNAVAILABLE_FALLBACK_USER_ID
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.common_utils.claude_code_marketplace_auth import (
    marketplace_auth_required,
    marketplace_key_header_present,
)
from litellm.proxy.common_utils.resource_ownership import is_proxy_admin
from litellm.proxy.deps.repositories import PluginUploadRepositoryDep
from litellm.repositories.plugin_publication import read_plugin_publisher, read_plugin_review
from litellm.repositories.table_repositories import ClaudeCodePluginRepository
from litellm.types.proxy.auth.special_headers import SpecialHeaders
from litellm.types.proxy.claude_code_endpoints import (
    InstallationPreference,
    ListPluginsResponse,
    PluginListItem,
    PluginResponse,
    PluginSpec,
    RegisterPluginRequest,
    RegisterPluginResponse,
    UpdatePluginRequest,
    UploadPluginRequest,
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


def plugin_archive_url(request: Request, name: str, digest: str) -> str:
    from litellm.proxy.utils import get_custom_url

    return get_custom_url(
        request_base_url=str(request.base_url), route=f"claude-code/plugins/{name}/archive/{digest}.zip"
    )


def plugin_archive_source_matches(
    request: Request, name: str, source: Mapping[str, str], digest: str, *, has_agent_skill_metadata: bool
) -> bool:
    if has_agent_skill_metadata:
        return source.get("source") == "archive" and source.get("sha256") == digest
    return source == {"source": "archive", "url": plugin_archive_url(request, name, digest), "sha256": digest}


async def header_key_caller(request: Request) -> UserAPIKeyAuth:
    try:
        caller: Final = await user_api_key_auth(
            request=request,
            api_key=request.headers.get(SpecialHeaders.openai_authorization.value) or "",
            custom_litellm_key_header=request.headers.get(SpecialHeaders.custom_litellm_api_key.value),
        )
    except ProxyException as e:
        if e.code != "401":
            raise
        raise ProxyException(
            message=e.message,
            type=e.type,
            param=e.param,
            code=401,
            headers={**e.headers, "WWW-Authenticate": "Bearer"},
        ) from e
    if caller.user_id == DB_UNAVAILABLE_FALLBACK_USER_ID:
        raise ProxyException(
            message="The LiteLLM key could not be verified because the database is unavailable",
            type=ProxyErrorTypes.no_db_connection,
            param=None,
            code=503,
        )
    return caller


async def _marketplace_caller(request: Request, key: str | None) -> UserAPIKeyAuth | None:
    if marketplace_auth_required():
        return await header_key_caller(request)
    if key:
        return await user_api_key_auth(request=request, api_key=f"Bearer {key}")
    return None


@router.post(
    "/claude-code/plugins/upload",
    tags=["Claude Code Marketplace"],
    response_model=RegisterPluginResponse,
)
async def upload_plugin(
    request: Request,
    uploads: PluginUploadRepositoryDep,
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
    file: Annotated[UploadFile, File(description="Skill ZIP containing SKILL.md and supporting files, up to 10 MiB")],
    metadata: Annotated[Json[UploadPluginRequest], Form(description="JSON skill metadata without a source URL")],
) -> RegisterPluginResponse:
    """Create a skill and its uploaded ZIP atomically. Duplicate names return 409."""
    _require_proxy_admin(user_api_key_dict)
    if uploads is None:
        raise _error_response(500, CommonProxyErrors.db_not_connected_error.value)
    content: Final = await file.read(MAX_SKILL_UPLOAD_BYTES + 1)
    if len(content) > MAX_SKILL_UPLOAD_BYTES:
        raise _error_response(413, "Skill ZIP must be 10 MiB or smaller")
    archive: Final = await asyncio.to_thread(build_uploaded_skill_archive, content, metadata.name)
    if archive is None:
        raise _error_response(
            400, "Upload a valid ZIP with SKILL.md at the root or inside one skill folder, and safe supporting files"
        )
    digest: Final = hashlib.sha256(archive).hexdigest()
    source: Final = {
        "source": "archive",
        "url": plugin_archive_url(request, metadata.name, digest),
        "sha256": digest,
    }
    spec: Final = RegisterPluginRequest.model_validate({**metadata.model_dump(), "source": source})
    _validate_installation_preference(spec)
    published: Final = await uploads.publish(
        PluginPublication(
            name=spec.name,
            version=spec.version,
            description=spec.description,
            manifest=_build_plugin_manifest(spec.name, spec),
            content=archive,
            created_by=user_api_key_dict.user_id,
        )
    )
    if isinstance(published, PluginPublicationError):
        raise _publication_error(published)
    return RegisterPluginResponse(
        status="success",
        action="created",
        plugin=PluginResponse(
            id=published.id,
            name=spec.name,
            version=published.version,
            description=spec.description,
            source=source,
            installation_preference=spec.installation_preference,
            enabled=True,
        ),
    )


def require_download_recorded(result: PluginDownloadResult) -> None:
    match result:
        case PluginDownloadResult.RECORDED:
            return
        case PluginDownloadResult.SKILL_NOT_FOUND:
            raise _error_response(404, "Skill archive not found")
        case _:
            assert_never(result)


def plugin_downloader(caller: UserAPIKeyAuth | None) -> PluginDownloader:
    return PluginDownloader(
        authenticated=caller is not None,
        user_id=caller.user_id if caller is not None else None,
        user_email=caller.user_email if caller is not None else None,
        key_alias=caller.key_alias if caller is not None else None,
        team_id=caller.team_id if caller is not None else None,
        team_alias=caller.team_alias if caller is not None else None,
    )


@router.get(
    "/claude-code/plugins/{plugin_name}/archive/{digest}.zip",
    tags=["Claude Code Marketplace"],
    response_class=Response,
    responses={200: {"content": {"application/zip": {"schema": {"type": "string", "format": "binary"}}}}},
)
async def download_plugin_archive(
    request: Request,
    plugin_name: str,
    digest: str,
    uploads: PluginUploadRepositoryDep,
) -> Response:
    """Download an uploaded ZIP under the same visibility policy as its skill."""
    caller: Final = (
        await header_key_caller(request)
        if marketplace_key_header_present(request)
        else await _marketplace_caller(request, None)
    )
    visibility: Final = await resolve_skill_visibility(caller)
    if uploads is None:
        raise _error_response(500, CommonProxyErrors.db_not_connected_error.value)
    archive: Final = await uploads.get_metadata(plugin_name)
    if (
        archive is None
        or not visibility.allows(archive)
        or not plugin_archive_source_matches(
            request, plugin_name, archive.source, digest, has_agent_skill_metadata=archive.has_agent_skill_metadata
        )
    ):
        raise _error_response(404, "Skill archive not found")
    content: Final = await uploads.get_content(archive)
    if content is None:
        raise _error_response(404, "Skill archive not found")
    require_download_recorded(await uploads.record_download(archive, digest, plugin_downloader(caller)))
    return Response(
        content=content,
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="{archive.name}.zip"',
            "Cache-Control": "private, no-store",
            "X-Content-Type-Options": "nosniff",
        },
    )


@router.get(
    "/claude-code/plugins/{plugin_name}/downloads",
    tags=["Claude Code Marketplace"],
    response_model=PluginDownloadsResponse,
)
async def get_plugin_downloads(
    plugin_name: str,
    uploads: PluginUploadRepositoryDep,
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
    limit: Annotated[int, Query(ge=1, le=100, description="Maximum number of download events")] = 25,
    cursor: Annotated[str | None, Query(pattern=r"^[1-9][0-9]{0,18}$", description="Previous next_cursor")] = None,
) -> PluginDownloadsResponse:
    """Admin-only ZIP download totals and authenticated downloader history.

    Anonymous downloads contribute only to totals. Counts include repeat requests and client prefetches,
    starting when tracking is deployed. External sources and installation completion are not observed.
    """
    _require_proxy_admin(user_api_key_dict)
    if uploads is None:
        raise _error_response(500, CommonProxyErrors.db_not_connected_error.value)
    before: Final = int(cursor) if cursor is not None else None
    if before is not None and before > 9223372036854775807:
        raise _error_response(422, "Invalid download cursor")
    result: Final = await uploads.downloads(plugin_name, limit, before)
    if result is None:
        raise _error_response(404, "Plugin not found")
    return result


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

    Without a key the catalog holds the enabled (public) plugins. With `?key=sk-...`
    the key is authenticated and the catalog also holds the disabled plugins granted
    to it through `object_permission.skills` or an access group on the key or its team.

    With `general_settings.claude_code_marketplace_auth_required: true` the key must come
    in the `Authorization` or `x-litellm-api-key` header (Claude Code sends them from the
    marketplace source's `headers`), `?key=` is ignored, and a caller with no valid key
    gets a 401 with `WWW-Authenticate: Bearer`.

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
        caller: Final = await _marketplace_caller(request, key)
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

            installation_preference = get_installation_preference(manifest)
            entry: _MarketplaceEntry = (
                {"name": plugin.name, "source": manifest["source"]}
                if installation_preference is None
                else {
                    "name": plugin.name,
                    "source": manifest["source"],
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
            detail={"error": "Failed to generate marketplace"},
        )


# Allowlist for git-subdir paths: one or more segments separated by '/'.
# Each segment must start with an alphanumeric character and contain only
# alphanumeric characters, dots, hyphens, and underscores.
# This implicitly blocks '..', leading '/', backslashes, and percent-encoded sequences.
_VALID_GIT_SUBDIR_PATH_RE: Final = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]*(/[a-zA-Z0-9][a-zA-Z0-9._-]*)*$")
_VALID_SHA256_RE: Final = re.compile(r"^[0-9a-fA-F]{64}$")


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
        if not is_https_url_with_host(source.get("url", "")):
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
    if spec.source.get("source") != "archive":
        raise HTTPException(
            status_code=400,
            detail={
                "error": "installation_preference needs an archive source, the only plugins Claude Desktop lists "
                "from a marketplace.json URL"
            },
        )
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
    dumped: Final[Mapping[str, object]] = MappingProxyType(spec.model_dump(exclude_none=True))
    return {"name": name, **{key: value for key, value in dumped.items() if value and key != "name"}}


def _error_response(status_code: int, message: str) -> HTTPException:
    return HTTPException(status_code=status_code, detail={"error": message})


def _publication_error(error: PluginPublicationError) -> HTTPException:
    match error.kind:
        case "invalid_version":
            return _error_response(400, error.message)
        case "not_found":
            return _error_response(404, error.message)
        case "conflict":
            return _error_response(409, error.message)


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
    uploads: PluginUploadRepositoryDep,
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
    try:
        _require_proxy_admin(user_api_key_dict)

        if uploads is None:
            raise _error_response(500, CommonProxyErrors.db_not_connected_error.value)

        if not re.match(r"^[a-z0-9-]+$", request.name):
            raise HTTPException(
                status_code=400,
                detail={"error": "Plugin name must be kebab-case (lowercase letters, numbers, hyphens)"},
            )

        _validate_plugin_source(request.source)
        _validate_installation_preference(request)

        manifest: Final[Mapping[str, object]] = _build_plugin_manifest(request.name, request)

        plugin: Final = await uploads.publish(
            PluginPublication(
                name=request.name,
                version=request.version,
                description=request.description,
                manifest=manifest,
                created_by=user_api_key_dict.user_id,
            )
        )
        if isinstance(plugin, PluginPublicationError):
            raise _publication_error(plugin)

        verbose_proxy_logger.info("Plugin %s created successfully", request.name)

        return RegisterPluginResponse(
            status="success",
            action="created",
            plugin=PluginResponse(
                id=plugin.id,
                name=plugin.name,
                version=plugin.version,
                description=request.description,
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
) -> ListPluginsResponse:
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
            # Parse manifest to get additional fields
            manifest = json.loads(p.manifest_json) if p.manifest_json else {}

            review = read_plugin_review(MANIFEST_FIELDS.validate_python(manifest))
            plugin_list.append(
                PluginListItem(
                    id=p.id,
                    created_by=p.created_by,
                    approved_by=review.approved_by if review else None,
                    published_by=read_plugin_publisher(MANIFEST_FIELDS.validate_python(manifest)),
                    security_scan_passed=review.security_scan_passed if review else False,
                    name=p.name,
                    version=p.version,
                    description=p.description,
                    source=manifest.get("source", {}),
                    has_agent_skill_metadata=has_agent_skill_metadata(MANIFEST_FIELDS.validate_python(manifest)),
                    author=manifest.get("author"),
                    homepage=manifest.get("homepage"),
                    keywords=manifest.get("keywords"),
                    category=manifest.get("category"),
                    domain=manifest.get("domain"),
                    namespace=manifest.get("namespace"),
                    installation_preference=get_installation_preference(MANIFEST_FIELDS.validate_python(manifest)),
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
    uploads: PluginUploadRepositoryDep,
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
):
    """
    Update an existing plugin in the LiteLLM marketplace.

    The plugin is identified by its name in the path, which is the resource
    identity and cannot be changed here. This is a full replace, not a merge:
    the manifest is rebuilt from the request body, so any optional field left
    out is reset to its default, except version which automatically increments
    the published patch version when omitted or null.
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
        if uploads is None:
            raise _error_response(500, CommonProxyErrors.db_not_connected_error.value)

        prisma_client: Final = await get_prisma_client()

        repository: Final = ClaudeCodePluginRepository(prisma_client, use_writer=True)
        existing: Final[_PluginRecord | None] = await repository.table.find_unique(where={"name": plugin_name})
        if not existing:
            raise _error_response(404, f"Plugin '{plugin_name}' not found")
        if request.expected_updated_at is not None and request.expected_updated_at != existing.updated_at:
            raise _error_response(409, "The published skill changed. Reload it before publishing an update")

        current_manifest: Final = MANIFEST_FIELDS.validate_json(existing.manifest_json or "{}")
        if request.source != current_manifest.get("source"):
            _validate_plugin_source(request.source)
        _validate_installation_preference(request)

        manifest: Final[Mapping[str, object]] = {
            **_build_plugin_manifest(plugin_name, request),
            **(
                {"agent_skill": current_manifest["agent_skill"]}
                if "agent_skill" in current_manifest and request.source == current_manifest.get("source")
                else {}
            ),
        }

        plugin: Final = await uploads.publish(
            PluginPublication(
                name=plugin_name,
                version=request.version,
                description=request.description,
                manifest=manifest,
                created_by=user_api_key_dict.user_id,
            ),
            PluginPublicationBase(existing.id, existing.updated_at),
        )
        if isinstance(plugin, PluginPublicationError):
            raise _publication_error(plugin)

        verbose_proxy_logger.info("Plugin %s updated successfully", plugin_name)

        return RegisterPluginResponse(
            status="success",
            action="updated",
            plugin=PluginResponse(
                id=plugin.id,
                name=plugin.name,
                version=plugin.version,
                description=request.description,
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
