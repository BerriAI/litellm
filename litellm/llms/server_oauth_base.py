import base64
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import httpx

from litellm._logging import verbose_logger
from litellm.llms.custom_httpx.http_handler import _get_httpx_client

TOKEN_EXPIRY_SKEW_SECONDS = 300
DEFAULT_TOKEN_DIR = Path(os.getenv("SERVER_OAUTH_TOKEN_DIR", os.path.expanduser("~/.config/litellm/server_oauth")))


class ServerOAuthError(Exception):
    pass


class JsonOAuthTokenStore:
    def __init__(self, provider: str, env_prefix: str) -> None:
        self.provider = provider
        self.env_prefix = env_prefix
        self.path = Path(os.getenv(f"{env_prefix}_OAUTH_FILE", str(DEFAULT_TOKEN_DIR / f"{provider}.json")))

    def load(self) -> Dict[str, Any]:
        self._restore_from_env_if_needed()
        try:
            with open(self.path, "r") as f:
                data = json.load(f)
        except IOError as exc:
            raise ServerOAuthError(
                f"{self.provider} OAuth token file missing at {self.path}. Set {self.env_prefix}_OAUTH_JSON_B64 or mount a token file."
            ) from exc
        except json.JSONDecodeError as exc:
            raise ServerOAuthError(f"Invalid {self.provider} OAuth token file at {self.path}: {exc}") from exc
        return data

    def save(self, data: Dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(data, f)
        try:
            self.path.chmod(0o600)
        except OSError:
            pass

    def _restore_from_env_if_needed(self) -> None:
        encoded = os.getenv(f"{self.env_prefix}_OAUTH_JSON_B64")
        if not encoded or self.path.exists():
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_bytes(base64.b64decode(encoded))
        try:
            self.path.chmod(0o600)
        except OSError:
            pass


def normalize_expires_at(raw: Dict[str, Any]) -> int:
    expires_at = raw.get("expires_at") or raw.get("expiresAt") or raw.get("expiry_date")
    if expires_at is None and isinstance(raw.get("expires_in"), (int, float)):
        expires_at = int(time.time()) + int(raw["expires_in"])
    if expires_at is None:
        return 0
    expires_at = int(expires_at)
    if expires_at > 10_000_000_000:
        expires_at //= 1000
    return expires_at


class RefreshTokenOAuthAuthenticator:
    provider: str
    env_prefix: str
    token_url: str

    def __init__(self) -> None:
        self.store = JsonOAuthTokenStore(self.provider, self.env_prefix)

    def get_access_token(self) -> str:
        data = self.store.load()
        access_token = data.get("access_token") or data.get("accessToken")
        if access_token and not self._is_expired(data):
            return access_token
        refresh_token = data.get("refresh_token") or data.get("refreshToken")
        if not refresh_token:
            if access_token:
                return access_token
            raise ServerOAuthError(f"{self.provider} OAuth token file has no access_token or refresh_token")
        refreshed = self.refresh_tokens(data, refresh_token)
        self.store.save(refreshed)
        return refreshed["access_token"]

    def _is_expired(self, data: Dict[str, Any]) -> bool:
        return normalize_expires_at(data) <= int(time.time()) + TOKEN_EXPIRY_SKEW_SECONDS

    def refresh_tokens(self, data: Dict[str, Any], refresh_token: str) -> Dict[str, Any]:
        raise NotImplementedError

    def _post_refresh_json(self, payload: Dict[str, Any], headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        client = _get_httpx_client()
        resp = client.post(self.token_url, json=payload, headers=headers or {})
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            verbose_logger.warning("%s OAuth refresh failed: %s", self.provider, exc)
            raise ServerOAuthError(f"{self.provider} OAuth refresh failed: {exc}") from exc
        return resp.json()

    def _post_refresh_form(self, payload: Dict[str, Any], headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        client = _get_httpx_client()
        resp = client.post(self.token_url, data=payload, headers=headers or {})
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            verbose_logger.warning("%s OAuth refresh failed: %s", self.provider, exc)
            raise ServerOAuthError(f"{self.provider} OAuth refresh failed: {exc}") from exc
        return resp.json()
