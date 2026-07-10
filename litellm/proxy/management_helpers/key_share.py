"""
Ephemeral, self-hosted secure sharing of virtual keys.

A proxy admin turns a plaintext ``sk-...`` key into a single-use, auto-expiring
link served by the proxy itself. The key is encrypted at rest with the proxy's
salt/master key and held in the proxy cache (Redis when configured, in-memory
otherwise) only until it expires or is viewed. No external service, credential
or dependency is involved.
"""

import html
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional, Protocol

from pydantic import BaseModel

from litellm.proxy._types import KeyShareExpiry
from litellm.proxy.common_utils.encrypt_decrypt_utils import (
    decrypt_value_helper,
    encrypt_value_helper,
)


class ShareSecretCache(Protocol):
    """Typed view of the proxy cache methods this module needs.

    The proxy passes its ``DualCache`` here; typing against a Protocol keeps the
    untyped cache surface from leaking ``Any`` into this module.
    """

    async def async_set_cache(self, key: str, value: str, *, ttl: int) -> None: ...

    async def async_get_cache(self, key: str) -> object: ...

    async def async_delete_cache(self, key: str) -> None: ...


_CACHE_PREFIX = "litellm:key_share:"

_EXPIRY_SECONDS: dict[KeyShareExpiry, int] = {
    "OneHour": 60 * 60,
    "OneDay": 24 * 60 * 60,
    "SevenDays": 7 * 24 * 60 * 60,
    "FourteenDays": 14 * 24 * 60 * 60,
    "ThirtyDays": 30 * 24 * 60 * 60,
}


class SharedKeyRecord(BaseModel):
    encrypted_key: str
    one_time_only: bool


class CreatedShare(BaseModel):
    token: str
    expires_at: datetime
    one_time_only: bool


def _cache_key(token: str) -> str:
    return f"{_CACHE_PREFIX}{token}"


async def create_share(
    *,
    key: str,
    expire_after: KeyShareExpiry,
    one_time_only: bool,
    cache: ShareSecretCache,
) -> CreatedShare:
    token = secrets.token_urlsafe(32)
    ttl_seconds = _EXPIRY_SECONDS[expire_after]
    record = SharedKeyRecord(
        encrypted_key=encrypt_value_helper(key),
        one_time_only=one_time_only,
    )
    await cache.async_set_cache(_cache_key(token), record.model_dump_json(), ttl=ttl_seconds)
    return CreatedShare(
        token=token,
        expires_at=datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds),
        one_time_only=one_time_only,
    )


async def reveal_share(*, token: str, cache: ShareSecretCache) -> Optional[str]:
    raw = await cache.async_get_cache(_cache_key(token))
    if not isinstance(raw, (str, bytes, bytearray)):
        return None
    record = SharedKeyRecord.model_validate_json(raw)
    if record.one_time_only:
        await cache.async_delete_cache(key=_cache_key(token))
    return decrypt_value_helper(record.encrypted_key, key="key_share")


def build_share_html(key: Optional[str]) -> str:
    if key is None:
        return (
            "<!doctype html><html><head><meta charset='utf-8'>"
            "<title>LiteLLM secure share</title></head>"
            "<body style='font-family:system-ui;max-width:640px;margin:80px auto;text-align:center'>"
            "<h2>Link expired or already used</h2>"
            "<p>This secure share link is no longer valid. Ask the proxy admin to create a new one.</p>"
            "</body></html>"
        )
    safe_key = html.escape(key)
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>LiteLLM secure share</title></head>"
        "<body style='font-family:system-ui;max-width:640px;margin:80px auto'>"
        "<h2>Your API key</h2>"
        "<p>Copy it now. For your security this link may stop working after you leave this page.</p>"
        f"<pre id='k' style='background:#f4f4f5;padding:16px;border-radius:8px;overflow:auto'>{safe_key}</pre>"
        "<button onclick=\"navigator.clipboard.writeText(document.getElementById('k').innerText)\">Copy</button>"
        "</body></html>"
    )
