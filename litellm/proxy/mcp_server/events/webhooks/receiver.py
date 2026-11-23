import hashlib
import time
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Final

from pydantic import JsonValue, TypeAdapter, ValidationError
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from litellm._logging import verbose_logger
from litellm.proxy.mcp_server.events.config import EventsSettings, events_settings
from litellm.proxy.mcp_server.events.webhooks.signing import parse_secret, verify
from litellm.proxy.mcp_server.events.webhooks.store import EventStore, QueuedDelivery, default_store

MAX_BODY_BYTES: Final = 256 * 1024
_JSON_OBJECT: Final = TypeAdapter[dict[str, JsonValue]](dict[str, JsonValue])


async def _read_body(request: Request) -> bytes | None:
    chunks: Final[list[bytes]] = []  # mutable-ok: bounded request buffer
    size = 0  # rebind-ok: running byte count of the capped read
    async for chunk in request.stream():
        size += len(chunk)
        if size > MAX_BODY_BYTES:
            return None
        if chunk:
            chunks.append(chunk)
    return b"".join(chunks)


def gateway_delivery_id(subscription_id: str, upstream_webhook_id: str) -> str:
    return "msg_" + hashlib.sha256(f"{subscription_id}:{upstream_webhook_id}".encode()).hexdigest()[:32]


async def receive_webhook(
    subscription_id: str,
    request: Request,
    settings: Callable[[], EventsSettings] = events_settings,
    store: Callable[[], EventStore | None] = default_store,
) -> Response:
    events_store: Final = store()
    if not settings().enabled or events_store is None:
        return Response(status_code=404)
    body: Final = await _read_body(request)
    if body is None:
        return Response(status_code=413)
    row: Final = await events_store.get(subscription_id)
    if row is None or row.status == "terminated":
        return Response(status_code=410)
    secret: Final = parse_secret(row.upstream_secret)
    headers: Final = {key.lower(): value for key, value in request.headers.items()}
    if secret is None or not verify((secret,), headers, body, int(time.time())):
        return Response(status_code=401)
    try:
        envelope: Final = _JSON_OBJECT.validate_json(body)
    except ValidationError:
        return Response(status_code=400)
    kind: Final = envelope.get("type")
    if kind == "verification":
        challenge: Final = envelope.get("challenge")
        return JSONResponse({"challenge": challenge}) if isinstance(challenge, str) else Response(status_code=400)
    if row.status == "pending":
        return Response(status_code=425)
    if headers.get("x-mcp-subscription-id") != row.upstream_subscription_id:
        return Response(status_code=401)
    if kind not in (None, "gap", "terminated") or (row.status == "suspended" and kind is None):
        return Response(status_code=204)
    delivery_id: Final = gateway_delivery_id(subscription_id, headers.get("webhook-id", ""))
    try:
        accepted: Final = await events_store.enqueue_current(
            row,
            QueuedDelivery(
                delivery_id=delivery_id,
                subscription_id=subscription_id,
                body=body.decode("utf-8"),
                webhook_id=delivery_id,
                next_attempt_at=datetime.now(timezone.utc),
            ),
        )
        if not accepted:
            return Response(status_code=503)
    except Exception as error:  # noqa: BLE001  # the upstream retries a delivery the gateway could not store durably
        verbose_logger.warning("Failed to enqueue MCP event delivery for %s: %s", subscription_id, error)
        return Response(status_code=503)
    return Response(status_code=204)
