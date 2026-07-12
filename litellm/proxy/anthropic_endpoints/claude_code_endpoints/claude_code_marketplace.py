"""
CLAUDE CODE MARKETPLACE

Provides a registry/discovery layer for Claude Code plugins.
Plugins are stored as metadata + git source references in LiteLLM database.
Actual plugin files are hosted on GitHub/GitLab/Bitbucket.

Endpoints:
/claude-code/marketplace.json  - GET  - List plugins for Claude Code discovery
/claude-code/plugins           - POST - Register a plugin
/claude-code/plugins           - GET  - List plugins (admin)
/claude-code/plugins/{name}    - GET  - Get plugin details
/claude-code/plugins/{name}/enable  - POST - Enable a plugin
/claude-code/plugins/{name}/disable - POST - Disable a plugin
/claude-code/plugins/{name}    - DELETE - Delete a plugin
"""

import json
import re
from datetime import datetime, timezone
from typing import Any, Dict, FrozenSet, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, Security
from fastapi.responses import JSONResponse

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import CommonProxyErrors, ProxyException, UserAPIKeyAuth
from litellm.proxy.anthropic_endpoints.claude_code_endpoints.claude_code_skill_authz import (
    get_allowed_skills,
)
from litellm.proxy.auth.user_api_key_auth import (
    anthropic_api_key_header,
    api_key_header,
    azure_api_key_header,
    azure_apim_header,
    custom_litellm_key_header,
    google_ai_studio_api_key_header,
    user_api_key_auth,
)
from litellm.repositories.table_repositories import (
    ClaudeCodePluginRepository,
    SkillMarketplaceRepository,
)
from litellm.types.proxy.claude_code_endpoints import (
    ListPluginsResponse,
    PluginListItem,
    RegisterPluginRequest,
)

router = APIRouter()


async def _optional_user_api_key_auth(
    request: Request,
    api_key: str = Security(api_key_header),
    azure_api_key: str = Security(azure_api_key_header),
    anthropic_api_key: Optional[str] = Security(anthropic_api_key_header),
    google_ai_studio_api_key: Optional[str] = Security(google_ai_studio_api_key_header),
    azure_apim_key: Optional[str] = Security(azure_apim_header),
    custom_litellm_key: Optional[str] = Security(custom_litellm_key_header),
) -> Optional[UserAPIKeyAuth]:
    """
    Resolve UserAPIKeyAuth if a key is present (header or, for this route,
    the `key` query param wired up via RouteChecks.is_claude_code_marketplace_route),
    but never raise - this route must stay accessible with no key at all.
    """
    try:
        return await user_api_key_auth(
            request=request,
            api_key=api_key,
            azure_api_key_header=azure_api_key,
            anthropic_api_key_header=anthropic_api_key,
            google_ai_studio_api_key_header=google_ai_studio_api_key,
            azure_apim_header=azure_apim_key,
            custom_litellm_key_header=custom_litellm_key,
        )
    except (HTTPException, ProxyException):
        return None


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
async def get_marketplace(
    user_api_key_dict: Optional[UserAPIKeyAuth] = Depends(_optional_user_api_key_auth),  # noqa: B008  # DI idiom
):
    """
    Serve marketplace.json for Claude Code plugin discovery.

    This endpoint is accessed by Claude Code CLI when users run:
    - claude plugin marketplace add <url>
    - claude plugin install <name>@<marketplace>

    No key is required - unauthenticated requests see every enabled plugin.
    A valid key (header, or `?key=` query param for the CLI) additionally
    unlocks any plugin whose name is in that key/team/org's allowed_skills.

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

        allowed_skills: FrozenSet[str] = (
            await get_allowed_skills(user_api_key_dict, prisma_client) if user_api_key_dict is not None else frozenset()
        )
        where = (
            {"OR": [{"enabled": True}, {"name": {"in": list(allowed_skills)}}]} if allowed_skills else {"enabled": True}
        )

        plugins = await ClaudeCodePluginRepository(prisma_client).table.find_many(where=where)

        # A per-skill grant (allowed_skills) is a standing entry on a
        # key/team/org's object_permission - it isn't cleared just because an
        # admin later disables the marketplace that owned the skill. Without
        # this, a disabled marketplace's skills stay reachable forever by
        # anyone previously granted one by name. Plugins with no
        # marketplace_id (hand-registered) are unaffected.
        marketplace_ids = {p.marketplace_id for p in plugins if p.marketplace_id}
        if marketplace_ids:
            disabled_marketplace_ids = {
                m.id
                for m in await SkillMarketplaceRepository(prisma_client).table.find_many(
                    where={"id": {"in": list(marketplace_ids)}, "enabled": False}
                )
            }
            if disabled_marketplace_ids:
                plugins = [p for p in plugins if p.marketplace_id not in disabled_marketplace_ids]

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
    Register a plugin in the LiteLLM marketplace.

    LiteLLM acts as a registry/discovery layer. Plugins are hosted on
    GitHub/GitLab/Bitbucket. Claude Code will clone from the git source
    when users install.

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
        Registration status and plugin information.

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

        # Validate name format
        if not re.match(r"^[a-z0-9-]+$", request.name):
            raise HTTPException(
                status_code=400,
                detail={"error": "Plugin name must be kebab-case (lowercase letters, numbers, hyphens)"},
            )

        # Validate source format
        source = request.source
        _validate_plugin_source(source)

        # Build manifest for storage
        manifest: Dict[str, Any] = {
            "name": request.name,
            "source": request.source,
        }
        if request.version:
            manifest["version"] = request.version
        if request.description:
            manifest["description"] = request.description
        if request.author:
            manifest["author"] = request.author.model_dump(exclude_none=True)
        if request.homepage:
            manifest["homepage"] = request.homepage
        if request.keywords:
            manifest["keywords"] = request.keywords
        if request.category:
            manifest["category"] = request.category
        if request.domain:
            manifest["domain"] = request.domain
        if request.namespace:
            manifest["namespace"] = request.namespace

        # Check if plugin exists
        existing = await ClaudeCodePluginRepository(prisma_client).table.find_unique(where={"name": request.name})

        if existing:
            plugin = await ClaudeCodePluginRepository(prisma_client).table.update(
                where={"name": request.name},
                data={
                    "version": request.version,
                    "description": request.description,
                    "manifest_json": json.dumps(manifest),
                    "files_json": "{}",
                    "updated_at": datetime.now(timezone.utc),
                },
            )
            action = "updated"
        else:
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
            action = "created"

        verbose_proxy_logger.info(f"Plugin {request.name} {action} successfully")

        return {
            "status": "success",
            "action": action,
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
