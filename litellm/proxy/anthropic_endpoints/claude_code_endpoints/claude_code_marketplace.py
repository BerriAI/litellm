"""
CLAUDE CODE MARKETPLACE

Provides a registry/discovery layer for Claude Code plugins.
Plugins are stored as metadata + git source references in LiteLLM database.
Actual plugin files are hosted on GitHub/GitLab/Bitbucket.

Endpoints:
/claude-code/marketplace.json  - GET  - List plugins for Claude Code discovery
/claude-code/plugins           - POST - Register a new plugin (create-only)
/claude-code/plugins           - GET  - List plugins (admin)
/claude-code/plugins/{name}    - GET  - Get plugin details
/claude-code/plugins/{name}    - PUT  - Update an existing plugin
/claude-code/plugins/{name}/enable  - POST - Enable a plugin
/claude-code/plugins/{name}/disable - POST - Disable a plugin
/claude-code/plugins/{name}    - DELETE - Delete a plugin
"""

import json
import re
from datetime import datetime, timezone
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse

try:
    from prisma.errors import PrismaError, UniqueViolationError
except ImportError:

    class PrismaError(Exception):  # type: ignore[no-redef]
        """Sentinel used when prisma is not installed; never raised in that case."""

    class UniqueViolationError(PrismaError):  # type: ignore[no-redef]
        """Sentinel used when prisma is not installed; never raised in that case."""


from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import CommonProxyErrors, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.repositories.table_repositories import ClaudeCodePluginRepository
from litellm.types.proxy.claude_code_endpoints import (
    ListPluginsResponse,
    PluginListItem,
    PluginSpec,
    RegisterPluginRequest,
    UpdatePluginRequest,
)

router = APIRouter()


async def _get_prisma_client():
    """Get the prisma client from proxy_server."""
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(
            status_code=500,
            detail={"error": CommonProxyErrors.db_not_connected_error.value},
        )
    return prisma_client


@router.get(
    "/claude-code/marketplace.json",
    tags=["Claude Code Marketplace"],
)
async def get_marketplace():
    """
    Serve marketplace.json for Claude Code plugin discovery.

    This endpoint is accessed by Claude Code CLI when users run:
    - claude plugin marketplace add <url>
    - claude plugin install <name>@<marketplace>

    Returns:
        Marketplace catalog with list of available plugins and their git sources.

    Example:
        ```bash
        claude plugin marketplace add http://localhost:4000/claude-code/marketplace.json
        claude plugin install my-plugin@litellm
        ```
    """
    try:
        prisma_client = await _get_prisma_client()

        plugins = await ClaudeCodePluginRepository(prisma_client).table.find_many(where={"enabled": True})

        plugin_list = []
        for plugin in plugins:
            try:
                manifest = json.loads(plugin.manifest_json)
            except json.JSONDecodeError:
                verbose_proxy_logger.warning(f"Plugin {plugin.name} has invalid manifest JSON, skipping")
                continue

            # Source must be specified for URL-based marketplaces
            if "source" not in manifest:
                verbose_proxy_logger.warning(f"Plugin {plugin.name} has no source field, skipping")
                continue

            entry: Dict[str, Any] = {
                "name": plugin.name,
                "source": manifest["source"],
            }

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

        marketplace = {
            "name": "litellm",
            "owner": {"name": "LiteLLM", "email": "support@litellm.ai"},
            "plugins": plugin_list,
        }

        return JSONResponse(content=marketplace)

    except HTTPException:
        raise
    except Exception as e:
        verbose_proxy_logger.exception(f"Error generating marketplace: {e}")
        raise HTTPException(
            status_code=500,
            detail={"error": f"Failed to generate marketplace: {str(e)}"},
        )


# Allowlist for git-subdir paths: one or more segments separated by '/'.
# Each segment must start with an alphanumeric character and contain only
# alphanumeric characters, dots, hyphens, and underscores.
# This implicitly blocks '..', leading '/', backslashes, and percent-encoded sequences.
_VALID_GIT_SUBDIR_PATH_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]*(/[a-zA-Z0-9][a-zA-Z0-9._-]*)*$")


def _validate_plugin_source(source: Dict[str, Any]) -> None:
    """Validate plugin source format, raising HTTPException on invalid input."""
    source_type = source.get("source")
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
    else:
        raise HTTPException(
            status_code=400,
            detail={"error": "source.source must be 'github', 'url', or 'git-subdir'"},
        )


def _build_plugin_manifest(name: str, spec: PluginSpec) -> Dict[str, Any]:
    """Build the stored manifest dict shared by plugin create and update."""
    optional_fields = {
        "version": spec.version,
        "description": spec.description,
        "author": spec.author.model_dump(exclude_none=True) if spec.author else None,
        "homepage": spec.homepage,
        "keywords": spec.keywords,
        "category": spec.category,
        "domain": spec.domain,
        "namespace": spec.namespace,
    }
    return {
        "name": name,
        "source": spec.source,
        **{key: value for key, value in optional_fields.items() if value},
    }


def _name_conflict_error(name: str) -> HTTPException:
    return HTTPException(
        status_code=409,
        detail={
            "error": f"A skill named '{name}' already exists. Update the existing skill instead of adding it again."
        },
    )


@router.post(
    "/claude-code/plugins",
    tags=["Claude Code Marketplace"],
    dependencies=[Depends(user_api_key_auth)],
)
async def register_plugin(
    request: RegisterPluginRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Register a new plugin in the LiteLLM marketplace.

    LiteLLM acts as a registry/discovery layer. Plugins are hosted on
    GitHub/GitLab/Bitbucket. Claude Code will clone from the git source
    when users install.

    This endpoint is create-only and never overwrites. If a plugin with
    the same name already exists it returns 409 Conflict; use
    PUT /claude-code/plugins/{plugin_name} to update an existing plugin.

    Parameters:
        - name: Plugin name (kebab-case)
        - source: Git source reference (github, url, or git-subdir format)
        - version: Semantic version (optional)
        - description: Plugin description (optional)
        - author: Author information (optional)
        - homepage: Plugin homepage URL (optional)
        - keywords: Search keywords (optional)
        - category: Plugin category (optional)

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
        prisma_client = await _get_prisma_client()

        if not re.match(r"^[a-z0-9-]+$", request.name):
            raise HTTPException(
                status_code=400,
                detail={"error": "Plugin name must be kebab-case (lowercase letters, numbers, hyphens)"},
            )

        _validate_plugin_source(request.source)

        existing = await ClaudeCodePluginRepository(prisma_client).table.find_unique(where={"name": request.name})
        if existing:
            raise _name_conflict_error(request.name)

        manifest = _build_plugin_manifest(request.name, request)

        try:
            plugin = await ClaudeCodePluginRepository(prisma_client).table.create(
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

        verbose_proxy_logger.info(f"Plugin {request.name} created successfully")

        return {
            "status": "success",
            "action": "created",
            "plugin": {
                "id": plugin.id,
                "name": plugin.name,
                "version": plugin.version,
                "description": plugin.description,
                "source": request.source,
                "enabled": plugin.enabled,
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        verbose_proxy_logger.exception(f"Error registering plugin: {e}")
        raise HTTPException(
            status_code=500,
            detail={"error": f"Registration failed: {str(e)}"},
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
        prisma_client = await _get_prisma_client()

        where = {"enabled": True} if enabled_only else {}
        plugins = await ClaudeCodePluginRepository(prisma_client).table.find_many(where=where)

        plugin_list = []
        for p in plugins:
            # Parse manifest to get additional fields
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
        verbose_proxy_logger.exception(f"Error listing plugins: {e}")
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
        prisma_client = await _get_prisma_client()

        plugin = await ClaudeCodePluginRepository(prisma_client).table.find_unique(where={"name": plugin_name})

        if not plugin:
            raise HTTPException(
                status_code=404,
                detail={"error": f"Plugin '{plugin_name}' not found"},
            )

        manifest = json.loads(plugin.manifest_json) if plugin.manifest_json else {}

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
            "enabled": plugin.enabled,
            "created_at": plugin.created_at.isoformat() if plugin.created_at else None,
            "updated_at": plugin.updated_at.isoformat() if plugin.updated_at else None,
            "created_by": plugin.created_by,
        }

    except HTTPException:
        raise
    except Exception as e:
        verbose_proxy_logger.exception(f"Error getting plugin: {e}")
        raise HTTPException(
            status_code=500,
            detail={"error": str(e)},
        )


@router.put(
    "/claude-code/plugins/{plugin_name}",
    tags=["Claude Code Marketplace"],
    dependencies=[Depends(user_api_key_auth)],
)
async def update_plugin(
    plugin_name: str,
    request: UpdatePluginRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
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

    Parameters:
        - plugin_name: Name of the plugin to update (path parameter)
        - source: Git source reference (github, url, or git-subdir format)
        - version: Semantic version (optional)
        - description: Plugin description (optional)
        - author: Author information (optional)
        - homepage: Plugin homepage URL (optional)
        - keywords: Search keywords (optional)
        - category: Plugin category (optional)

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
    try:
        prisma_client = await _get_prisma_client()

        _validate_plugin_source(request.source)

        existing = await ClaudeCodePluginRepository(prisma_client).table.find_unique(where={"name": plugin_name})
        if not existing:
            raise HTTPException(
                status_code=404,
                detail={"error": f"Plugin '{plugin_name}' not found"},
            )

        manifest = _build_plugin_manifest(plugin_name, request)

        plugin = await ClaudeCodePluginRepository(prisma_client).table.update(
            where={"name": plugin_name},
            data={
                "version": request.version,
                "description": request.description,
                "manifest_json": json.dumps(manifest),
                "files_json": "{}",
                "updated_at": datetime.now(timezone.utc),
            },
        )

        verbose_proxy_logger.info(f"Plugin {plugin_name} updated successfully")

        return {
            "status": "success",
            "action": "updated",
            "plugin": {
                "id": plugin.id,
                "name": plugin.name,
                "version": plugin.version,
                "description": plugin.description,
                "source": request.source,
                "enabled": plugin.enabled,
            },
        }

    except HTTPException:
        raise
    except PrismaError as e:
        verbose_proxy_logger.exception(f"Error updating plugin: {e}")
        raise HTTPException(
            status_code=500,
            detail={"error": f"Update failed: {str(e)}"},
        )


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

    Parameters:
        - plugin_name: The name of the plugin to enable
    """
    try:
        prisma_client = await _get_prisma_client()

        plugin = await ClaudeCodePluginRepository(prisma_client).table.find_unique(where={"name": plugin_name})
        if not plugin:
            raise HTTPException(
                status_code=404,
                detail={"error": f"Plugin '{plugin_name}' not found"},
            )

        await ClaudeCodePluginRepository(prisma_client).table.update(
            where={"name": plugin_name},
            data={"enabled": True, "updated_at": datetime.now(timezone.utc)},
        )

        verbose_proxy_logger.info(f"Plugin {plugin_name} enabled")
        return {"status": "success", "message": f"Plugin '{plugin_name}' enabled"}

    except HTTPException:
        raise
    except Exception as e:
        verbose_proxy_logger.exception(f"Error enabling plugin: {e}")
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

    Parameters:
        - plugin_name: The name of the plugin to disable
    """
    try:
        prisma_client = await _get_prisma_client()

        plugin = await ClaudeCodePluginRepository(prisma_client).table.find_unique(where={"name": plugin_name})
        if not plugin:
            raise HTTPException(
                status_code=404,
                detail={"error": f"Plugin '{plugin_name}' not found"},
            )

        await ClaudeCodePluginRepository(prisma_client).table.update(
            where={"name": plugin_name},
            data={"enabled": False, "updated_at": datetime.now(timezone.utc)},
        )

        verbose_proxy_logger.info(f"Plugin {plugin_name} disabled")
        return {"status": "success", "message": f"Plugin '{plugin_name}' disabled"}

    except HTTPException:
        raise
    except Exception as e:
        verbose_proxy_logger.exception(f"Error disabling plugin: {e}")
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

    Parameters:
        - plugin_name: The name of the plugin to delete
    """
    try:
        prisma_client = await _get_prisma_client()

        plugin = await ClaudeCodePluginRepository(prisma_client).table.find_unique(where={"name": plugin_name})
        if not plugin:
            raise HTTPException(
                status_code=404,
                detail={"error": f"Plugin '{plugin_name}' not found"},
            )

        await ClaudeCodePluginRepository(prisma_client).table.delete(where={"name": plugin_name})

        verbose_proxy_logger.info(f"Plugin {plugin_name} deleted")
        return {"status": "success", "message": f"Plugin '{plugin_name}' deleted"}

    except HTTPException:
        raise
    except Exception as e:
        verbose_proxy_logger.exception(f"Error deleting plugin: {e}")
        raise HTTPException(
            status_code=500,
            detail={"error": str(e)},
        )
