"""
XCT Agent Marketplace config endpoint (S3-07).

Server-side source of truth for the marketplace gateway URL so the
dashboard no longer hardcodes ``DEFAULT_GATEWAY_URL``. Admins can flip
the URL at runtime from the UI; a sane default is shipped here so a
fresh proxy install just works.

GET  /v1/xct-marketplace/config  — any authenticated caller
PUT  /v1/xct-marketplace/config  — PROXY_ADMIN only
"""

import os
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

import litellm
from litellm.caching.dual_cache import DualCache
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

router = APIRouter()

# Settled prod gateway as of 2026-05-19 — same URL used by the dashboard
# marketplace component pre-S3-07.
DEFAULT_GATEWAY_URL = "https://xct-agents-production.up.railway.app"

# DualCache shared with capability_endpoints would risk cross-pollution; keep
# this module's own tiny in-memory cache.
_cache = DualCache(default_in_memory_ttl=60)
_CACHE_KEY = "xct-marketplace:config"


class MarketplaceConfig(BaseModel):
    gateway_url: str = Field(..., description="Base URL of the XCT agent gateway.")
    description: Optional[str] = None


def _is_admin(uak: UserAPIKeyAuth) -> bool:
    return uak.user_role in (
        LitellmUserRoles.PROXY_ADMIN,
        LitellmUserRoles.PROXY_ADMIN.value,
    )


async def _load_config() -> MarketplaceConfig:
    """Resolve config from (in priority order): cache → env override → DB row → default.

    Storage layer in this iteration is just the cache. A persistent backing
    is added once the marketplace settles on its own table; until then the
    env var lets ops pin a URL across restarts.
    """
    hit = await _cache.async_get_cache(_CACHE_KEY)
    if hit is not None:
        if isinstance(hit, MarketplaceConfig):
            return hit
        if isinstance(hit, dict):
            return MarketplaceConfig(**hit)
    env = os.environ.get("XCT_AGENT_GATEWAY_URL")
    url = env or getattr(litellm, "xct_agent_gateway_url", None) or DEFAULT_GATEWAY_URL
    return MarketplaceConfig(gateway_url=url)


@router.get(
    "/v1/xct-marketplace/config",
    tags=["[beta] XCT Marketplace"],
    response_model=MarketplaceConfig,
)
async def get_marketplace_config(
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> MarketplaceConfig:
    return await _load_config()


@router.put(
    "/v1/xct-marketplace/config",
    tags=["[beta] XCT Marketplace"],
    response_model=MarketplaceConfig,
)
async def update_marketplace_config(
    body: MarketplaceConfig,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> MarketplaceConfig:
    if not _is_admin(user_api_key_dict):
        raise HTTPException(
            status_code=403,
            detail="Only proxy admins may change marketplace config.",
        )
    if not body.gateway_url.startswith(("https://", "http://")):
        raise HTTPException(
            status_code=400, detail="gateway_url must be an absolute http(s) URL."
        )
    # Stash on litellm so other modules can read it without a DB lookup.
    litellm.xct_agent_gateway_url = body.gateway_url  # type: ignore[attr-defined]
    await _cache.async_set_cache(_CACHE_KEY, body, ttl=60)
    return body
