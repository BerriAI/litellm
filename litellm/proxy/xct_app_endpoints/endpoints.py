"""
XCT App management — ``/v1/xct-apps`` (S4-03).

Public surface that lets a proxy admin provision a consumer app
(``xct-chat``, ``xct-home``, ``xct-agent-desktop``…) with its own
OAuth client_id + client_secret + capability scope.

Per ADR-0002 the OAuth flow itself is a separate module
(``litellm/proxy/xct_oauth_endpoints``); this file is purely the
admin / IT-ops CRUD layer.
"""

import hashlib
import secrets
import uuid
from typing import List, Optional
from urllib.parse import urlparse

from fastapi import APIRouter, Depends, HTTPException, Query

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.types.xct_apps import (
    XCTApp,
    XCTAppCreate,
    XCTAppCreateResponse,
    XCTAppPatch,
)

router = APIRouter()


def _is_admin(uak: UserAPIKeyAuth) -> bool:
    return uak.user_role in (
        LitellmUserRoles.PROXY_ADMIN,
        LitellmUserRoles.PROXY_ADMIN.value,
    )


def _require_admin(uak: UserAPIKeyAuth) -> None:
    if not _is_admin(uak):
        raise HTTPException(
            status_code=403, detail="Only proxy admins may manage XCT apps."
        )


def _generate_client_id() -> str:
    """Stable, URL-safe public identifier (no entropy concerns)."""
    return f"xct_{uuid.uuid4().hex[:24]}"


def _generate_client_secret() -> str:
    """32-byte URL-safe secret. Returned exactly once at create / rotate."""
    return secrets.token_urlsafe(32)


def _hash_secret(secret: str) -> str:
    """SHA-256 of the secret (high-entropy → rainbow tables don't apply).

    Same trade-off as `webhook_endpoints`: cleartext is 256 bits of random,
    so bcrypt's slow-hash advantage doesn't buy anything that matters.
    """
    return hashlib.sha256(secret.encode()).hexdigest()


def _validate_redirect_uri(uri: str) -> None:
    """Tighter than the webhook URL check: scheme + must not be wildcard."""
    parsed = urlparse(uri)
    if parsed.scheme not in ("https", "http"):
        raise HTTPException(
            status_code=400,
            detail=f"redirect_uri scheme must be http(s): {uri!r}",
        )
    if "*" in uri:
        raise HTTPException(
            status_code=400,
            detail=f"redirect_uri must not contain wildcards: {uri!r}",
        )
    if parsed.hostname is None:
        raise HTTPException(
            status_code=400, detail=f"redirect_uri missing hostname: {uri!r}"
        )


def _validate_redirect_uris(uris: List[str]) -> None:
    if not uris:
        return
    for uri in uris:
        _validate_redirect_uri(uri)


def _row_to_app(row) -> XCTApp:
    data = row.model_dump() if hasattr(row, "model_dump") else dict(row)
    return XCTApp(
        app_id=data["app_id"],
        app_name=data["app_name"],
        display_name=data["display_name"],
        description=data.get("description"),
        icon_url=data.get("icon_url"),
        oauth_client_id=data["oauth_client_id"],
        redirect_uris=data.get("redirect_uris") or [],
        default_team_id=data.get("default_team_id"),
        default_scopes=data.get("default_scopes") or [],
        capability_scope_id=data.get("capability_scope_id"),
        rpm_limit=data.get("rpm_limit"),
        daily_budget=(
            float(data["daily_budget"])
            if data.get("daily_budget") is not None
            else None
        ),
        is_active=bool(data.get("is_active", True)),
        created_at=data.get("created_at"),
        created_by=data.get("created_by"),
        updated_at=data.get("updated_at"),
    )


async def _load_or_404(prisma_client, app_id: str):
    row = await prisma_client.db.litellm_xctapptable.find_unique(
        where={"app_id": app_id}
    )
    if row is None:
        raise HTTPException(status_code=404, detail=f"XCT app '{app_id}' not found.")
    return row


@router.post(
    "/v1/xct-apps",
    tags=["[beta] XCT Apps"],
    response_model=XCTAppCreateResponse,
)
async def create_app(
    payload: XCTAppCreate,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> XCTAppCreateResponse:
    """Create a new XCT app + provision its OAuth client_id/secret.

    Cleartext secret is returned ONCE in the response. The proxy keeps
    only the SHA-256 hash.
    """
    _require_admin(user_api_key_dict)
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=503, detail="DB not initialized")

    _validate_redirect_uris(payload.redirect_uris)

    client_id = _generate_client_id()
    client_secret = _generate_client_secret()
    secret_hash = _hash_secret(client_secret)

    try:
        row = await prisma_client.db.litellm_xctapptable.create(
            data={
                "app_name": payload.app_name,
                "display_name": payload.display_name,
                "description": payload.description,
                "icon_url": payload.icon_url,
                "oauth_client_id": client_id,
                "oauth_client_secret_hash": secret_hash,
                "redirect_uris": payload.redirect_uris,
                "default_team_id": payload.default_team_id,
                "default_scopes": payload.default_scopes,
                "capability_scope_id": payload.capability_scope_id,
                "rpm_limit": payload.rpm_limit,
                "daily_budget": payload.daily_budget,
                "is_active": payload.is_active,
                "created_by": user_api_key_dict.user_id,
            }
        )
    except Exception as e:
        verbose_proxy_logger.exception("XCT app create failed: %s", e)
        # Most likely a unique-violation on app_name.
        raise HTTPException(status_code=409, detail=f"XCT app conflict: {e}") from e

    base = _row_to_app(row)
    return XCTAppCreateResponse(**base.model_dump(), client_secret=client_secret)


@router.get(
    "/v1/xct-apps",
    tags=["[beta] XCT Apps"],
    response_model=List[XCTApp],
)
async def list_apps(
    is_active: Optional[bool] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> List[XCTApp]:
    _require_admin(user_api_key_dict)
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        return []
    where: dict = {}
    if is_active is not None:
        where["is_active"] = is_active
    rows = await prisma_client.db.litellm_xctapptable.find_many(
        where=where, take=limit, order={"created_at": "desc"}
    )
    return [_row_to_app(r) for r in rows]


@router.get(
    "/v1/xct-apps/{app_id}",
    tags=["[beta] XCT Apps"],
    response_model=XCTApp,
)
async def get_app(
    app_id: str,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> XCTApp:
    _require_admin(user_api_key_dict)
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=503, detail="DB not initialized")
    return _row_to_app(await _load_or_404(prisma_client, app_id))


@router.patch(
    "/v1/xct-apps/{app_id}",
    tags=["[beta] XCT Apps"],
    response_model=XCTApp,
)
async def patch_app(
    app_id: str,
    patch: XCTAppPatch,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> XCTApp:
    _require_admin(user_api_key_dict)
    from litellm.proxy.proxy_server import prisma_client

    await _load_or_404(prisma_client, app_id)
    update_data = {k: v for k, v in patch.model_dump().items() if v is not None}
    if "redirect_uris" in update_data:
        _validate_redirect_uris(update_data["redirect_uris"])
    if not update_data:
        return await get_app(app_id, user_api_key_dict)
    row = await prisma_client.db.litellm_xctapptable.update(
        where={"app_id": app_id}, data=update_data
    )
    return _row_to_app(row)


@router.delete(
    "/v1/xct-apps/{app_id}",
    tags=["[beta] XCT Apps"],
)
async def delete_app(
    app_id: str,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> dict:
    """Hard-delete an XCT app.

    Outstanding OAuth tokens (LiteLLM_VerificationToken rows with this
    app_id) are NOT cascade-deleted — operators may want to keep the
    audit trail. Mark them revoked via /key/revoke if needed.
    """
    _require_admin(user_api_key_dict)
    from litellm.proxy.proxy_server import prisma_client

    await _load_or_404(prisma_client, app_id)
    await prisma_client.db.litellm_xctapptable.delete(where={"app_id": app_id})
    return {"app_id": app_id, "deleted": True}


@router.post(
    "/v1/xct-apps/{app_id}/rotate-secret",
    tags=["[beta] XCT Apps"],
    response_model=XCTAppCreateResponse,
)
async def rotate_app_secret(
    app_id: str,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> XCTAppCreateResponse:
    """Generate a fresh client_secret. Old secret is immediately invalid.

    Returns the new cleartext secret ONCE. Operators must update the
    app's deployment with this new value; the previous secret is dead.
    """
    _require_admin(user_api_key_dict)
    from litellm.proxy.proxy_server import prisma_client

    await _load_or_404(prisma_client, app_id)
    new_secret = _generate_client_secret()
    row = await prisma_client.db.litellm_xctapptable.update(
        where={"app_id": app_id},
        data={"oauth_client_secret_hash": _hash_secret(new_secret)},
    )
    base = _row_to_app(row)
    return XCTAppCreateResponse(**base.model_dump(), client_secret=new_secret)
