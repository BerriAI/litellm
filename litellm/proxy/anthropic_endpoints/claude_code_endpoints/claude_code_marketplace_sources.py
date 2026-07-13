"""
CLAUDE CODE MARKETPLACE SOURCES

Manages external Claude Code marketplace sources (git repos / marketplace.json URLs)
that LiteLLM syncs plugin listings from into LiteLLM_ClaudeCodePluginTable.

Endpoints:
/claude-code/marketplaces               - POST   - Register an external marketplace source
/claude-code/marketplaces               - GET    - List registered marketplace sources
/claude-code/marketplaces/{name}        - GET    - Get a marketplace source's details
/claude-code/marketplaces/{name}/sync   - POST   - Re-sync a marketplace source
/claude-code/marketplaces/{name}        - DELETE - Disable a marketplace source and its plugins
"""

import re

from fastapi import APIRouter, Depends, HTTPException

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import CommonProxyErrors, LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.anthropic_endpoints.claude_code_endpoints.claude_code_marketplace_sync import (
    MarketplaceRow,
    resolve_and_sync,
)
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.repositories.table_repositories import (
    ClaudeCodePluginRepository,
    SkillMarketplaceRepository,
)
from litellm.types.proxy.claude_code_endpoints import (
    ListMarketplacesResponse,
    MarketplaceSourceResponse,
    RegisterMarketplaceRequest,
    SyncMarketplaceResponse,
)

router = APIRouter()

_MARKETPLACE_NAME_RE = re.compile(r"^[a-z0-9-]+$")


async def _get_prisma_client():
    """Get the prisma client from proxy_server."""
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(
            status_code=500,
            detail={"error": CommonProxyErrors.db_not_connected_error.value},
        )
    return prisma_client


def _require_proxy_admin(user_api_key_dict: UserAPIKeyAuth) -> None:
    if user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN:
        raise HTTPException(
            status_code=403,
            detail={"error": "Only proxy admins can manage Claude Code marketplace sources"},
        )


def _derive_marketplace_slug(source: str) -> str:
    trimmed = source.rstrip("/")
    trimmed = trimmed[: -len(".git")] if trimmed.endswith(".git") else trimmed
    last_segment = trimmed.rsplit("/", 1)[-1]
    return re.sub(r"[^a-z0-9]+", "-", last_segment.lower()).strip("-")


async def _resolve_marketplace_name(prisma_client, request: RegisterMarketplaceRequest) -> str:
    slug = request.name if request.name else _derive_marketplace_slug(request.source)

    if not slug or not _MARKETPLACE_NAME_RE.match(slug):
        raise HTTPException(
            status_code=400,
            detail={"error": "Marketplace name must be kebab-case (lowercase letters, numbers, hyphens)"},
        )

    existing = await SkillMarketplaceRepository(prisma_client).table.find_unique(where={"name": slug})
    if existing:
        raise HTTPException(
            status_code=400,
            detail={"error": f"Marketplace '{slug}' already exists"},
        )

    return slug


async def _get_marketplace_or_404(prisma_client, name: str):
    marketplace = await SkillMarketplaceRepository(prisma_client).table.find_unique(where={"name": name})
    if not marketplace:
        raise HTTPException(
            status_code=404,
            detail={"error": f"Marketplace '{name}' not found"},
        )
    return marketplace


async def _count_plugins(prisma_client, marketplace_id: str) -> int:
    return await ClaudeCodePluginRepository(prisma_client).table.count(where={"marketplace_id": marketplace_id})


def _to_marketplace_source_response(marketplace, plugin_count: int | None) -> MarketplaceSourceResponse:
    return MarketplaceSourceResponse(
        id=marketplace.id,
        name=marketplace.name,
        display_name=marketplace.display_name,
        source_type=marketplace.source_type,
        source_ref=marketplace.source_ref,
        branch=marketplace.branch,
        enabled=marketplace.enabled,
        sync_status=marketplace.sync_status,
        sync_error=marketplace.sync_error,
        last_synced_at=marketplace.last_synced_at.isoformat() if marketplace.last_synced_at else None,
        plugin_count=plugin_count,
        skipped_count=marketplace.skipped_count,
        created_at=marketplace.created_at.isoformat() if marketplace.created_at else None,
        updated_at=marketplace.updated_at.isoformat() if marketplace.updated_at else None,
    )


async def _sync_and_build_response(prisma_client, marketplace) -> SyncMarketplaceResponse:
    marketplace_row = MarketplaceRow(
        id=marketplace.id,
        name=marketplace.name,
        source_ref=marketplace.source_ref,
        branch=marketplace.branch,
    )
    result = await resolve_and_sync(prisma_client, marketplace_row)
    refreshed = await SkillMarketplaceRepository(prisma_client).table.find_unique(where={"id": marketplace.id})
    return SyncMarketplaceResponse(
        status=result.status,
        marketplace=_to_marketplace_source_response(refreshed, result.plugin_count),
    )


@router.post(
    "/claude-code/marketplaces",
    tags=["Claude Code Marketplace"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=SyncMarketplaceResponse,
)
async def register_marketplace(
    request: RegisterMarketplaceRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),  # noqa: B008  # FastAPI DI idiom
):
    """
    Register and sync an external Claude Code marketplace source.

    Parameters:
        - source: 'org/repo' shorthand, a git URL, or a direct URL to a marketplace.json file
        - name: Marketplace slug to register under (optional, derived from source if omitted)

    Returns:
        Sync status and the registered marketplace's details.
    """
    try:
        prisma_client = await _get_prisma_client()
        _require_proxy_admin(user_api_key_dict)

        name = await _resolve_marketplace_name(prisma_client, request)

        marketplace = await SkillMarketplaceRepository(prisma_client).table.create(
            data={
                "name": name,
                "display_name": name,
                "source_type": "claude_marketplace_json",
                "source_ref": request.source,
                "sync_status": "pending",
                "created_by": user_api_key_dict.user_id,
            }
        )

        verbose_proxy_logger.info(f"Marketplace {name} registered, syncing now")
        return await _sync_and_build_response(prisma_client, marketplace)

    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001  # top-level endpoint boundary
        verbose_proxy_logger.exception(f"Error registering marketplace: {e}")
        raise HTTPException(
            status_code=500,
            detail={"error": f"Registration failed: {str(e)}"},
        )


@router.get(
    "/claude-code/marketplaces",
    tags=["Claude Code Marketplace"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=ListMarketplacesResponse,
)
async def list_marketplaces(
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),  # noqa: B008  # FastAPI DI idiom
):
    """List all registered marketplace sources."""
    try:
        prisma_client = await _get_prisma_client()
        _require_proxy_admin(user_api_key_dict)

        marketplaces = await SkillMarketplaceRepository(prisma_client).table.find_many()

        marketplace_list = [
            _to_marketplace_source_response(marketplace, await _count_plugins(prisma_client, marketplace.id))
            for marketplace in marketplaces
        ]

        return ListMarketplacesResponse(marketplaces=marketplace_list, count=len(marketplace_list))

    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001  # top-level endpoint boundary
        verbose_proxy_logger.exception(f"Error listing marketplaces: {e}")
        raise HTTPException(
            status_code=500,
            detail={"error": str(e)},
        )


@router.get(
    "/claude-code/marketplaces/{marketplace_name}",
    tags=["Claude Code Marketplace"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=MarketplaceSourceResponse,
)
async def get_marketplace_source(
    marketplace_name: str,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),  # noqa: B008  # FastAPI DI idiom
):
    """Get details of a registered marketplace source."""
    try:
        prisma_client = await _get_prisma_client()
        _require_proxy_admin(user_api_key_dict)

        marketplace = await _get_marketplace_or_404(prisma_client, marketplace_name)
        plugin_count = await _count_plugins(prisma_client, marketplace.id)

        return _to_marketplace_source_response(marketplace, plugin_count)

    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001  # top-level endpoint boundary
        verbose_proxy_logger.exception(f"Error getting marketplace: {e}")
        raise HTTPException(
            status_code=500,
            detail={"error": str(e)},
        )


@router.post(
    "/claude-code/marketplaces/{marketplace_name}/sync",
    tags=["Claude Code Marketplace"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=SyncMarketplaceResponse,
)
async def sync_marketplace(
    marketplace_name: str,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),  # noqa: B008  # FastAPI DI idiom
):
    """Re-sync a registered marketplace source."""
    try:
        prisma_client = await _get_prisma_client()
        _require_proxy_admin(user_api_key_dict)

        marketplace = await _get_marketplace_or_404(prisma_client, marketplace_name)

        verbose_proxy_logger.info(f"Re-syncing marketplace {marketplace_name}")
        return await _sync_and_build_response(prisma_client, marketplace)

    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001  # top-level endpoint boundary
        verbose_proxy_logger.exception(f"Error syncing marketplace: {e}")
        raise HTTPException(
            status_code=500,
            detail={"error": str(e)},
        )


@router.delete(
    "/claude-code/marketplaces/{marketplace_name}",
    tags=["Claude Code Marketplace"],
    dependencies=[Depends(user_api_key_auth)],
)
async def delete_marketplace(
    marketplace_name: str,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),  # noqa: B008  # FastAPI DI idiom
):
    """Disable a marketplace source and all plugins it registered."""
    try:
        prisma_client = await _get_prisma_client()
        _require_proxy_admin(user_api_key_dict)

        marketplace = await _get_marketplace_or_404(prisma_client, marketplace_name)

        await SkillMarketplaceRepository(prisma_client).table.update(
            where={"name": marketplace_name},
            data={"enabled": False},
        )
        await ClaudeCodePluginRepository(prisma_client).table.update_many(
            where={"marketplace_id": marketplace.id},
            data={"enabled": False},
        )

        verbose_proxy_logger.info(f"Marketplace {marketplace_name} disabled")
        return {"status": "success", "message": f"Marketplace '{marketplace_name}' disabled"}

    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001  # top-level endpoint boundary
        verbose_proxy_logger.exception(f"Error deleting marketplace: {e}")
        raise HTTPException(
            status_code=500,
            detail={"error": str(e)},
        )
