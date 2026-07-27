from __future__ import annotations

import base64
import binascii
import os
from dataclasses import dataclass

from litellm.secret_managers.main import str_to_bool


@dataclass(frozen=True, slots=True)
class PublicRelaySettings:
    enabled: bool
    base_url: str | None
    session_ttl_seconds: int
    max_api_keys: int
    reservation_ttl_seconds: int
    content_retention_days: int
    metadata_retention_days: int
    session_secret: bytes
    content_encryption_key: bytes | None
    content_encryption_key_version: int

    @classmethod
    def from_env(cls) -> PublicRelaySettings:
        return cls(
            enabled=str_to_bool(os.getenv("PUBLIC_RELAY_ENABLED")) is True,
            base_url=os.getenv("PUBLIC_RELAY_BASE_URL"),
            session_ttl_seconds=int(os.getenv("PUBLIC_RELAY_SESSION_TTL_SECONDS", "604800")),
            max_api_keys=int(os.getenv("PUBLIC_RELAY_MAX_API_KEYS", "5")),
            reservation_ttl_seconds=int(os.getenv("PUBLIC_RELAY_RESERVATION_TTL_SECONDS", "1800")),
            content_retention_days=int(os.getenv("PUBLIC_RELAY_CONTENT_RETENTION_DAYS", "7")),
            metadata_retention_days=int(os.getenv("PUBLIC_RELAY_METADATA_RETENTION_DAYS", "90")),
            session_secret=_decode_secret(os.getenv("PUBLIC_RELAY_SESSION_SECRET")),
            content_encryption_key=_decode_optional_key(os.getenv("PUBLIC_RELAY_CONTENT_ENCRYPTION_KEY")),
            content_encryption_key_version=int(os.getenv("PUBLIC_RELAY_CONTENT_ENCRYPTION_KEY_VERSION", "1")),
        )

    def missing_runtime_configuration(self) -> tuple[str, ...]:
        required = (
            ("PUBLIC_RELAY_SESSION_SECRET", self.session_secret),
            ("PUBLIC_RELAY_BASE_URL", self.base_url),
            ("PUBLIC_RELAY_CONTENT_ENCRYPTION_KEY", self.content_encryption_key),
        )
        missing = [name for name, value in required if not value]
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
