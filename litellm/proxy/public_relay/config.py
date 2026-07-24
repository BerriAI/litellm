from __future__ import annotations

import base64
import binascii
import os
from dataclasses import dataclass

from litellm.secret_managers.main import str_to_bool


@dataclass(frozen=True, slots=True)
class PublicRelaySettings:
    enabled: bool
    session_ttl_seconds: int
    verification_ttl_seconds: int
    verification_resend_seconds: int
    verification_max_attempts: int
    max_api_keys: int
    min_checkout_cents: int
    max_checkout_cents: int
    reservation_ttl_seconds: int
    content_retention_days: int
    metadata_retention_days: int
    session_secret: bytes
    content_encryption_key: bytes | None
    content_encryption_key_version: int
    turnstile_verify_url: str | None
    stripe_secret_key: str | None
    stripe_webhook_secret: str | None
    checkout_success_url: str | None
    checkout_cancel_url: str | None

    @classmethod
    def from_env(cls) -> PublicRelaySettings:
        return cls(
            enabled=str_to_bool(os.getenv("PUBLIC_RELAY_ENABLED")) is True,
            session_ttl_seconds=int(os.getenv("PUBLIC_RELAY_SESSION_TTL_SECONDS", "604800")),
            verification_ttl_seconds=int(os.getenv("PUBLIC_RELAY_VERIFICATION_TTL_SECONDS", "600")),
            verification_resend_seconds=int(os.getenv("PUBLIC_RELAY_VERIFICATION_RESEND_SECONDS", "60")),
            verification_max_attempts=int(os.getenv("PUBLIC_RELAY_VERIFICATION_MAX_ATTEMPTS", "5")),
            max_api_keys=int(os.getenv("PUBLIC_RELAY_MAX_API_KEYS", "5")),
            min_checkout_cents=int(os.getenv("PUBLIC_RELAY_MIN_CHECKOUT_CENTS", "500")),
            max_checkout_cents=int(os.getenv("PUBLIC_RELAY_MAX_CHECKOUT_CENTS", "50000")),
            reservation_ttl_seconds=int(os.getenv("PUBLIC_RELAY_RESERVATION_TTL_SECONDS", "1800")),
            content_retention_days=int(os.getenv("PUBLIC_RELAY_CONTENT_RETENTION_DAYS", "7")),
            metadata_retention_days=int(os.getenv("PUBLIC_RELAY_METADATA_RETENTION_DAYS", "90")),
            session_secret=_decode_secret(os.getenv("PUBLIC_RELAY_SESSION_SECRET")),
            content_encryption_key=_decode_optional_key(os.getenv("PUBLIC_RELAY_CONTENT_ENCRYPTION_KEY")),
            content_encryption_key_version=int(os.getenv("PUBLIC_RELAY_CONTENT_ENCRYPTION_KEY_VERSION", "1")),
            turnstile_verify_url=os.getenv("PUBLIC_RELAY_TURNSTILE_VERIFY_URL"),
            stripe_secret_key=os.getenv("STRIPE_SECRET_KEY"),
            stripe_webhook_secret=os.getenv("STRIPE_WEBHOOK_SECRET"),
            checkout_success_url=os.getenv("PUBLIC_RELAY_CHECKOUT_SUCCESS_URL"),
            checkout_cancel_url=os.getenv("PUBLIC_RELAY_CHECKOUT_CANCEL_URL"),
        )

    def missing_runtime_configuration(self) -> tuple[str, ...]:
        required = (
            ("PUBLIC_RELAY_SESSION_SECRET", self.session_secret),
            ("PUBLIC_RELAY_CONTENT_ENCRYPTION_KEY", self.content_encryption_key),
            ("PUBLIC_RELAY_TURNSTILE_VERIFY_URL", self.turnstile_verify_url),
            ("STRIPE_SECRET_KEY", self.stripe_secret_key),
            ("STRIPE_WEBHOOK_SECRET", self.stripe_webhook_secret),
            ("PUBLIC_RELAY_CHECKOUT_SUCCESS_URL", self.checkout_success_url),
            ("PUBLIC_RELAY_CHECKOUT_CANCEL_URL", self.checkout_cancel_url),
        )
        missing = [name for name, value in required if not value]
        sender = os.getenv("RESEND_FROM_EMAIL") or os.getenv("SMTP_SENDER_EMAIL")
        provider = os.getenv("RESEND_API_KEY") or os.getenv("SMTP_HOST")
        if not sender or not provider:
            missing.append("RESEND_API_KEY/SMTP_HOST and sender email")
        return tuple(missing)


def _decode_secret(value: str | None) -> bytes:
    if value is None:
        return b""
    try:
        decoded = base64.urlsafe_b64decode(value)
    except (ValueError, binascii.Error):
        return b""
    return decoded if len(decoded) >= 32 else b""


def _decode_optional_key(value: str | None) -> bytes | None:
    decoded = _decode_secret(value)
    return decoded if len(decoded) in (16, 24, 32) else None
