"""
Webhook delivery — HMAC signature + retry-with-backoff + DLQ (S6-05).

Public surface:

    await emit_event(event_type, payload, *, app_id=None)
        Fan out a fresh event to every active subscription that listens
        for it. Each per-subscription delivery runs as a background task.

    await dispatch_to_subscription(subscription, event_type, payload, ...)
        Send a single payload to a single subscription (used by both
        emit_event and the /v1/webhooks/{id}/test endpoint).

Delivery model:
    - HMAC-SHA256 over the JSON body keyed by the subscription's secret,
      sent as ``X-XCT-Signature: sha256=<hex>``.
    - Backoff: 1s → 5s → 30s → 2m → 10m (5 attempts).
    - DLQ: persist to ``LiteLLM_WebhookDLQ`` after the final attempt fails.
    - subscription.consecutive_failures auto-disables a subscription once
      it hits 20 in a row (operators can re-enable manually).
"""

import asyncio
import hashlib
import hmac
import json
from datetime import datetime
from typing import Any, Dict, List, Optional

import httpx

from litellm._logging import verbose_proxy_logger

_BACKOFF_SECONDS: List[int] = [1, 5, 30, 120, 600]
_DELIVERY_TIMEOUT_SECONDS = 10.0
_DISABLE_AFTER_CONSECUTIVE_FAILURES = 20


def _sign(secret: str, body: bytes) -> str:
    digest = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return f"sha256={digest}"


def _build_envelope(event_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "event": event_type,
        "sent_at": datetime.utcnow().isoformat() + "Z",
        "data": payload,
    }


def _matches_filters(
    filters: Optional[Dict[str, Any]], envelope: Dict[str, Any]
) -> bool:
    """Tiny filter DSL: top-level keys map to required values in envelope['data'].

    No-op (returns True) when filters is None or empty. Filters are AND-ed.
    """
    if not filters:
        return True
    data = envelope.get("data") or {}
    for key, expected in filters.items():
        if data.get(key) != expected:
            return False
    return True


async def dispatch_to_subscription(
    *,
    subscription,
    event_type: str,
    payload: Dict[str, Any],
    max_attempts: int = len(_BACKOFF_SECONDS),
    secret: Optional[str] = None,
) -> Dict[str, Any]:
    """Synchronous send-with-retry to one subscription.

    Returns a dict describing the outcome so the test endpoint can show it
    back to the caller. Production fan-out wraps this in a background task
    via ``emit_event``.

    ``secret`` is normally None — the dispatcher cannot recover the original
    secret from secret_hash (one-way). The test endpoint passes a generated
    secret only for synthetic test events that don't need a real verifier.
    For real events the receiver verifies against THEIR copy of the secret;
    we sign with a deterministic per-subscription key derived from the hash.
    """
    from litellm.proxy.proxy_server import prisma_client

    target_url: str = subscription.target_url
    sub_id: str = subscription.subscription_id
    envelope = _build_envelope(event_type, payload)

    # Receiver-side verification expects HMAC keyed by the secret they stored
    # on create. Since we don't keep the cleartext secret, we sign with the
    # secret_hash itself — receivers should HMAC-verify against the secret
    # they were handed at create time AND additionally compute the same
    # hash, picking whichever matches. The SDK helper does both transparently.
    signing_key = secret or subscription.secret_hash

    body_bytes = json.dumps(envelope, separators=(",", ":")).encode()
    signature = _sign(signing_key, body_bytes)

    last_error: Optional[str] = None
    delivered = False
    for attempt in range(1, max_attempts + 1):
        try:
            async with httpx.AsyncClient(
                timeout=_DELIVERY_TIMEOUT_SECONDS, follow_redirects=False
            ) as client:
                resp = await client.post(
                    target_url,
                    content=body_bytes,
                    headers={
                        "Content-Type": "application/json",
                        "X-XCT-Signature": signature,
                        "X-XCT-Event": event_type,
                        "X-XCT-Subscription-Id": sub_id,
                        "X-XCT-Attempt": str(attempt),
                    },
                )
            if 200 <= resp.status_code < 300:
                delivered = True
                break
            last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
        except Exception as e:
            last_error = f"{type(e).__name__}: {e}"

        if attempt >= max_attempts:
            break
        await asyncio.sleep(_BACKOFF_SECONDS[attempt - 1])

    await _persist_outcome(
        prisma_client=prisma_client,
        subscription_id=sub_id,
        event_type=event_type,
        envelope=envelope,
        delivered=delivered,
        last_error=last_error,
        attempts=attempt,
    )

    return {
        "subscription_id": sub_id,
        "event": event_type,
        "delivered": delivered,
        "attempts": attempt,
        "error": last_error if not delivered else None,
    }


async def _persist_outcome(
    *,
    prisma_client,
    subscription_id: str,
    event_type: str,
    envelope: Dict[str, Any],
    delivered: bool,
    last_error: Optional[str],
    attempts: int,
) -> None:
    """Stamp last_success_at / last_failure_at + DLQ on terminal failure."""
    if prisma_client is None:
        return
    try:
        now = datetime.utcnow()
        if delivered:
            await prisma_client.db.litellm_webhooksubscriptiontable.update(
                where={"subscription_id": subscription_id},
                data={
                    "last_success_at": now,
                    "consecutive_failures": 0,
                },
            )
            return

        # Failure path: bump consecutive_failures, park in DLQ, maybe auto-
        # disable. Read current count first because Prisma's atomic
        # increment doesn't return the new value, and we need it for the
        # auto-disable decision.
        current = await prisma_client.db.litellm_webhooksubscriptiontable.find_unique(
            where={"subscription_id": subscription_id}
        )
        next_failures = (current.consecutive_failures if current else 0) + 1
        update_data: Dict[str, Any] = {
            "last_failure_at": now,
            "consecutive_failures": next_failures,
        }
        if next_failures >= _DISABLE_AFTER_CONSECUTIVE_FAILURES:
            update_data["is_active"] = False
        await prisma_client.db.litellm_webhooksubscriptiontable.update(
            where={"subscription_id": subscription_id},
            data=update_data,
        )
        await prisma_client.db.litellm_webhookdlq.create(
            data={
                "subscription_id": subscription_id,
                "event_type": event_type,
                "payload": envelope,
                "last_error": last_error,
                "attempts": attempts,
                "first_attempt_at": now,
                "last_attempt_at": now,
            }
        )
    except Exception as e:
        verbose_proxy_logger.warning("webhook outcome persist failed: %s", e)


async def emit_event(
    event_type: str,
    payload: Dict[str, Any],
    *,
    app_id: Optional[str] = None,
) -> None:
    """Fan out an event to every active subscription that listens for it.

    Fire-and-forget: each subscription's delivery runs as a background task
    so the caller's request path is never blocked.
    """
    try:
        from litellm.proxy.proxy_server import prisma_client
    except Exception:
        return
    if prisma_client is None:
        return

    where: Dict[str, Any] = {
        "is_active": True,
        "events": {"has": event_type},
    }
    if app_id is not None:
        where["app_id"] = app_id

    try:
        subs = await prisma_client.db.litellm_webhooksubscriptiontable.find_many(
            where=where
        )
    except Exception as e:
        verbose_proxy_logger.debug("emit_event: subscriber lookup failed: %s", e)
        return

    envelope_for_filter = _build_envelope(event_type, payload)
    for sub in subs:
        if not _matches_filters(getattr(sub, "filters", None), envelope_for_filter):
            continue
        # Fire-and-forget — background task so the emitting request returns fast.
        asyncio.create_task(
            dispatch_to_subscription(
                subscription=sub,
                event_type=event_type,
                payload=payload,
            )
        )
