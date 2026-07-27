from __future__ import annotations

import base64
import secrets
from pathlib import Path

values = {
    "POSTGRES_PASSWORD": secrets.token_urlsafe(32),
    "LITELLM_MASTER_KEY": f"sk-{secrets.token_urlsafe(36)}",
    "PUBLIC_RELAY_SESSION_SECRET": base64.urlsafe_b64encode(secrets.token_bytes(32)).decode(),
    "PUBLIC_RELAY_CONTENT_ENCRYPTION_KEY": base64.urlsafe_b64encode(secrets.token_bytes(32)).decode(),
    "PUBLIC_RELAY_CONTENT_ENCRYPTION_KEY_VERSION": "1",
    "PUBLIC_RELAY_BASE_URL": "https://47.236.187.190/ui",
    "PUBLIC_RELAY_ENABLED": "false",
    "PUBLIC_RELAY_MAX_API_KEYS": "5",
    "PUBLIC_RELAY_RESERVATION_TTL_SECONDS": "1800",
    "PUBLIC_RELAY_CONTENT_RETENTION_DAYS": "7",
    "PUBLIC_RELAY_METADATA_RETENTION_DAYS": "90",
    "ENTERPRISE_UPSTREAM_BASE_URL": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    "ENTERPRISE_UPSTREAM_API_KEY": "replace-before-enabling",
}
target = Path("/opt/litellm-relay/.env")
target.write_text("".join(f"{key}={value}\n" for key, value in values.items()))
target.chmod(0o600)
