import base64
import binascii
import hashlib
import hmac
import secrets
from collections.abc import Mapping, Sequence
from typing import Final

_PREFIX: Final = "whsec_"
_MIN_SECRET_BYTES: Final = 24
_MAX_SECRET_BYTES: Final = 64
_TOLERANCE_SECONDS: Final = 300


def parse_secret(value: str) -> bytes | None:
    if not value.startswith(_PREFIX):
        return None
    try:
        decoded: Final = base64.b64decode(value[len(_PREFIX) :], validate=True)
    except (binascii.Error, ValueError):
        return None
    return decoded if _MIN_SECRET_BYTES <= len(decoded) <= _MAX_SECRET_BYTES else None


def new_secret() -> str:
    return _PREFIX + base64.b64encode(secrets.token_bytes(32)).decode("ascii")


def new_webhook_id(kind: str | None = None) -> str:
    random: Final = secrets.token_urlsafe(18).replace("-", "_")
    return f"msg_{kind}_{random}" if kind is not None else f"msg_{random}"


def sign(secret: bytes, webhook_id: str, timestamp: int, body: bytes) -> str:
    digest: Final = hmac.new(secret, f"{webhook_id}.{timestamp}.".encode() + body, hashlib.sha256).digest()
    return "v1," + base64.b64encode(digest).decode("ascii")


def signed_headers(
    secret: bytes, webhook_id: str, timestamp: int, body: bytes, subscription_id: str
) -> Mapping[str, str]:
    return {
        "content-type": "application/json",
        "webhook-id": webhook_id,
        "webhook-timestamp": str(timestamp),
        "webhook-signature": sign(secret, webhook_id, timestamp, body),
        "x-mcp-subscription-id": subscription_id,
    }


def verify(secret_candidates: Sequence[bytes], headers: Mapping[str, str], body: bytes, now: int) -> bool:
    webhook_id: Final = headers.get("webhook-id")
    timestamp: Final = headers.get("webhook-timestamp")
    signatures: Final = (headers.get("webhook-signature") or "").split()
    if not webhook_id or not timestamp or not timestamp.isdigit() or not signatures:
        return False
    if abs(now - int(timestamp)) > _TOLERANCE_SECONDS:
        return False
    expected: Final = tuple(sign(secret, webhook_id, int(timestamp), body) for secret in secret_candidates)
    return any(_matches_any(candidate, expected) for candidate in signatures)


def _matches_any(candidate: str, expected: tuple[str, ...]) -> bool:
    return any(hmac.compare_digest(candidate, wanted) for wanted in expected)
