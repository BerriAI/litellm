"""
Webhook subscription CRUD — ``/v1/webhooks`` (S6-04).

Delivery (S6-05) and event emission (S6-06) live in sibling modules
``dispatcher.py`` and ``events.py``. This file is pure HTTP surface.
"""

import hashlib
import secrets
from typing import Optional
from urllib.parse import urlparse

from fastapi import APIRouter, Depends, HTTPException, Query

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.types.webhooks import (
    KNOWN_WEBHOOK_EVENTS,
    WebhookSubscription,
    WebhookSubscriptionCreate,
    WebhookSubscriptionCreateResponse,
    WebhookSubscriptionPatch,
)

router = APIRouter()


def _is_admin(uak: UserAPIKeyAuth) -> bool:
    return uak.user_role in (
        LitellmUserRoles.PROXY_ADMIN,
        LitellmUserRoles.PROXY_ADMIN.value,
    )


def _generate_secret() -> str:
    """High-entropy secret (32 bytes URL-safe). Returned ONCE on create."""
    return secrets.token_urlsafe(32)


def _hash_secret(secret: str) -> str:
    """SHA-256 of the secret. Cheap-to-compare for HMAC validation paths;
    sufficient because the secret is high-entropy (no rainbow table risk).

    We deliberately don't bcrypt: per-request webhook delivery would pay the
    bcrypt cost on every fan-out. SHA-256 of a 32-byte URL-safe random is
    indistinguishable from random to anyone without the original.
    """
    return hashlib.sha256(secret.encode()).hexdigest()


def _validate_target_url(url: str) -> None:
    """Reject non-http(s) and ban localhost/loopback/private ranges by name.

    SSRF guard: webhook delivery is a server-initiated HTTP call into
    operator-supplied URLs. We refuse any host that obviously points at
    internal infrastructure. Real netmask checks live in the dispatcher
    (S6-05) — this is a structural smoke test on create.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise HTTPException(
            status_code=400, detail="target_url must be http:// or https://"
        )
    if not parsed.hostname:
        raise HTTPException(status_code=400, detail="target_url missing hostname")
    host = parsed.hostname.lower()
    banned_hosts = {
        "localhost",
        "127.0.0.1",
        "::1",
        "0.0.0.0",
        "metadata.google.internal",
    }
    if host in banned_hosts:
        raise HTTPException(
            status_code=400,
            detail=f"target_url host '{host}' is not allowed (SSRF guard).",
        )
    # AWS / GCP / Azure metadata services
    if host in {"169.254.169.254"}:
        raise HTTPException(
            status_code=400,
            detail="target_url points at a cloud metadata service (refused).",
        )


def _row_to_subscription(row) -> WebhookSubscription:
    data = row.model_dump() if hasattr(row, "model_dump") else dict(row)
    return WebhookSubscription(
        subscription_id=data["subscription_id"],
        app_id=data.get("app_id"),
        team_id=data.get("team_id"),
        user_id=data.get("user_id"),
        events=data.get("events") or [],
        target_url=data["target_url"],
        filters=data.get("filters"),
        is_active=bool(data.get("is_active", True)),
        created_at=data.get("created_at"),
        created_by=data.get("created_by"),
        updated_at=data.get("updated_at"),
        last_success_at=data.get("last_success_at"),
        last_failure_at=data.get("last_failure_at"),
        consecutive_failures=int(data.get("consecutive_failures") or 0),
    )


async def _load_or_404(prisma_client, subscription_id: str):
    row = await prisma_client.db.litellm_webhooksubscriptiontable.find_unique(
        where={"subscription_id": subscription_id}
    )
    if row is None:
        raise HTTPException(
            status_code=404, detail=f"Webhook '{subscription_id}' not found"
        )
    return row


async def _require_writable(prisma_client, subscription_id: str, uak: UserAPIKeyAuth):
    row = await _load_or_404(prisma_client, subscription_id)
    if _is_admin(uak):
        return row
    if uak.user_id and getattr(row, "user_id", None) == uak.user_id:
        return row
    raise HTTPException(
        status_code=403,
        detail="Only the webhook owner or a proxy admin may modify this subscription.",
    )


@router.post(
    "/v1/webhooks",
    tags=["[beta] Webhooks"],
    response_model=WebhookSubscriptionCreateResponse,
)
async def create_webhook(
    payload: WebhookSubscriptionCreate,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> WebhookSubscriptionCreateResponse:
    """Register a webhook subscription. Returns the unhashed secret ONE time.

    The secret is what the dispatcher HMAC-signs each payload with; the
    caller stores it on their side, the proxy keeps only the SHA-256 hash.
    """
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=503, detail="DB not initialized")
    _validate_target_url(payload.target_url)

    unknown = set(payload.events) - set(KNOWN_WEBHOOK_EVENTS)
    if unknown:
        verbose_proxy_logger.warning(
            "Webhook subscribed to unknown event(s) %s — will never fire", unknown
        )

    secret = _generate_secret()
    row = await prisma_client.db.litellm_webhooksubscriptiontable.create(
        data={
            "app_id": payload.app_id,
            "team_id": payload.team_id or user_api_key_dict.team_id,
            "user_id": user_api_key_dict.user_id,
            "events": payload.events,
            "target_url": payload.target_url,
            "secret_hash": _hash_secret(secret),
            "filters": payload.filters,
            "is_active": payload.is_active,
            "created_by": user_api_key_dict.user_id,
        }
    )
    sub = _row_to_subscription(row)
    return WebhookSubscriptionCreateResponse(**sub.model_dump(), secret=secret)


@router.get(
    "/v1/webhooks",
    tags=["[beta] Webhooks"],
    response_model=list[WebhookSubscription],
)
async def list_webhooks(
    app_id: Optional[str] = Query(None),
    event: Optional[str] = Query(
        None, description="Filter to subscriptions listening for this event name."
    ),
    is_active: Optional[bool] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> list[WebhookSubscription]:
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        return []
    where: dict = {}
    if app_id is not None:
        where["app_id"] = app_id
    if is_active is not None:
        where["is_active"] = is_active
    if event is not None:
        where["events"] = {"has": event}
    # Non-admin caller can only see their own subscriptions.
    if not _is_admin(user_api_key_dict) and user_api_key_dict.user_id:
        where["user_id"] = user_api_key_dict.user_id
    rows = await prisma_client.db.litellm_webhooksubscriptiontable.find_many(
        where=where, take=limit, order={"created_at": "desc"}
    )
    return [_row_to_subscription(r) for r in rows]


@router.get(
    "/v1/webhooks/{subscription_id}",
    tags=["[beta] Webhooks"],
    response_model=WebhookSubscription,
)
async def get_webhook(
    subscription_id: str,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> WebhookSubscription:
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=503, detail="DB not initialized")
    row = await _load_or_404(prisma_client, subscription_id)
    if not _is_admin(user_api_key_dict):
        if (
            user_api_key_dict.user_id
            and getattr(row, "user_id", None) != user_api_key_dict.user_id
        ):
            raise HTTPException(status_code=403, detail="Not your webhook.")
    return _row_to_subscription(row)


@router.patch(
    "/v1/webhooks/{subscription_id}",
    tags=["[beta] Webhooks"],
    response_model=WebhookSubscription,
)
async def patch_webhook(
    subscription_id: str,
    patch: WebhookSubscriptionPatch,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> WebhookSubscription:
    from litellm.proxy.proxy_server import prisma_client

    await _require_writable(prisma_client, subscription_id, user_api_key_dict)
    update_data = {k: v for k, v in patch.model_dump().items() if v is not None}
    if "target_url" in update_data:
        _validate_target_url(update_data["target_url"])
    if not update_data:
        return await get_webhook(subscription_id, user_api_key_dict)
    row = await prisma_client.db.litellm_webhooksubscriptiontable.update(
        where={"subscription_id": subscription_id},
        data=update_data,
    )
    return _row_to_subscription(row)


@router.delete("/v1/webhooks/{subscription_id}", tags=["[beta] Webhooks"])
async def delete_webhook(
    subscription_id: str,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> dict:
    from litellm.proxy.proxy_server import prisma_client

    await _require_writable(prisma_client, subscription_id, user_api_key_dict)
    await prisma_client.db.litellm_webhooksubscriptiontable.delete(
        where={"subscription_id": subscription_id}
    )
    return {"subscription_id": subscription_id, "deleted": True}


@router.post("/v1/webhooks/{subscription_id}/test", tags=["[beta] Webhooks"])
async def test_webhook(
    subscription_id: str,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> dict:
    """Send a synthetic ``webhook.test`` event so the subscriber can verify wiring.

    Returns the dispatcher's result so the caller sees whether their endpoint
    accepted the payload.
    """
    from litellm.proxy.proxy_server import prisma_client
    from litellm.proxy.webhook_endpoints.dispatcher import dispatch_to_subscription

    row = await _require_writable(prisma_client, subscription_id, user_api_key_dict)
    return await dispatch_to_subscription(
        subscription=row,
        event_type="webhook.test",
        payload={
            "subscription_id": subscription_id,
            "message": "If you can read this, your endpoint is wired correctly.",
        },
        max_attempts=1,
    )
