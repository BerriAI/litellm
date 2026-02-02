import base64
import json
import os
import time
from typing import Any, Dict, Optional

import httpx

from litellm._logging import verbose_logger
from litellm.llms.custom_httpx.http_handler import _get_httpx_client

from .common_utils import (
    CHATGPT_API_BASE,
    CHATGPT_AUTH_BASE,
    CHATGPT_CLIENT_ID,
    CHATGPT_DEVICE_CODE_URL,
    CHATGPT_DEVICE_TOKEN_URL,
    CHATGPT_DEVICE_VERIFY_URL,
    CHATGPT_OAUTH_TOKEN_URL,
    GetAccessTokenError,
    GetDeviceCodeError,
    RefreshAccessTokenError,
)

TOKEN_EXPIRY_SKEW_SECONDS = 60
DEVICE_CODE_TIMEOUT_SECONDS = 15 * 60
DEVICE_CODE_COOLDOWN_SECONDS = 5 * 60
DEVICE_CODE_POLL_SLEEP_SECONDS = 5


class Authenticator:
    def __init__(self) -> None:
        self.token_dir = os.getenv(
            "CHATGPT_TOKEN_DIR",
            os.path.expanduser("~/.config/litellm/chatgpt"),
        )
        self.auth_file = os.path.join(
            self.token_dir, os.getenv("CHATGPT_AUTH_FILE", "auth.json")
        )
        self._ensure_token_dir()

    def get_api_base(self) -> str:
        return (
            os.getenv("CHATGPT_API_BASE")
            or os.getenv("OPENAI_CHATGPT_API_BASE")
            or CHATGPT_API_BASE
        )

    def get_access_token(self) -> str:
        auth_data = self._read_auth_file()
        if auth_data:
            access_token = auth_data.get("access_token")
            if access_token and not self._is_token_expired(auth_data, access_token):
                return access_token
            refresh_token = auth_data.get("refresh_token")
            if refresh_token:
                try:
                    refreshed = self._refresh_tokens(refresh_token)
                    return refreshed["access_token"]
                except RefreshAccessTokenError as exc:
                    verbose_logger.warning(
                        "ChatGPT refresh token failed, re-login required: %s", exc
                    )

        cooldown_remaining = self._get_device_code_cooldown_remaining(auth_data)
        if cooldown_remaining > 0:
            token = self._wait_for_access_token(cooldown_remaining)
            if token:
                return token

        tokens = self._login_device_code()
        return tokens["access_token"]

    def get_account_id(self) -> Optional[str]:
        auth_data = self._read_auth_file()
        if not auth_data:
            return None
        account_id = auth_data.get("account_id")
        if account_id:
            return account_id
        id_token = auth_data.get("id_token")
        access_token = auth_data.get("access_token")
        derived = self._extract_account_id(id_token or access_token)
        if derived:
            auth_data["account_id"] = derived
            self._write_auth_file(auth_data)
        return derived

    def _ensure_token_dir(self) -> None:
        if not os.path.exists(self.token_dir):
            os.makedirs(self.token_dir, exist_ok=True)

    def _read_auth_file(self) -> Optional[Dict[str, Any]]:
        try:
            with open(self.auth_file, "r") as f:
                return json.load(f)
        except IOError:
            return None
        except json.JSONDecodeError as exc:
            verbose_logger.warning("Invalid ChatGPT auth file: %s", exc)
            return None

    def _write_auth_file(self, data: Dict[str, Any]) -> None:
        try:
            with open(self.auth_file, "w") as f:
                json.dump(data, f)
        except IOError as exc:
            verbose_logger.error("Failed to write ChatGPT auth file: %s", exc)

    def _is_token_expired(self, auth_data: Dict[str, Any], access_token: str) -> bool:
        expires_at = auth_data.get("expires_at")
        if expires_at is None:
            expires_at = self._get_expires_at(access_token)
            if expires_at:
                auth_data["expires_at"] = expires_at
                self._write_auth_file(auth_data)
        if expires_at is None:
            return True
        return time.time() >= float(expires_at) - TOKEN_EXPIRY_SKEW_SECONDS

    def _get_expires_at(self, token: str) -> Optional[int]:
        claims = self._decode_jwt_claims(token)
        exp = claims.get("exp")
        if isinstance(exp, (int, float)):
            return int(exp)
        return None

    def _decode_jwt_claims(self, token: str) -> Dict[str, Any]:
        try:
            parts = token.split(".")
            if len(parts) < 2:
                return {}
            payload_b64 = parts[1]
            payload_b64 += "=" * (-len(payload_b64) % 4)
            payload_bytes = base64.urlsafe_b64decode(payload_b64)
            return json.loads(payload_bytes.decode("utf-8"))
        except Exception:
            return {}

    def _extract_account_id(self, token: Optional[str]) -> Optional[str]:
        if not token:
            return None
        claims = self._decode_jwt_claims(token)
        auth_claims = claims.get("https://api.openai.com/auth")
        if isinstance(auth_claims, dict):
            account_id = auth_claims.get("chatgpt_account_id")
            if isinstance(account_id, str) and account_id:
                return account_id
        return None

    def _login_device_code(self) -> Dict[str, str]:
        cooldown_remaining = self._get_device_code_cooldown_remaining(
            self._read_auth_file()
        )
        if cooldown_remaining > 0:
            token = self._wait_for_access_token(cooldown_remaining)
            if token:
                return {"access_token": token}

        device_code = self._request_device_code()
        self._record_device_code_request()
        print(  # noqa: T201
            "Sign in with ChatGPT using device code:\n"
            f"1) Visit {CHATGPT_DEVICE_VERIFY_URL}\n"
            f"2) Enter code: {device_code['user_code']}\n"
            "Device codes are a common phishing target. Never share this code.",
            flush=True,
        )
        auth_code = self._poll_for_authorization_code(device_code)
        tokens = self._exchange_code_for_tokens(auth_code)
        auth_data = self._build_auth_record(tokens)
        self._write_auth_file(auth_data)
        return tokens

    def _request_device_code(self) -> Dict[str, str]:
        try:
            client = _get_httpx_client()
            resp = client.post(
                CHATGPT_DEVICE_CODE_URL,
                json={"client_id": CHATGPT_CLIENT_ID},
            )
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPStatusError as exc:
            raise GetDeviceCodeError(
                message=f"Failed to request device code: {exc}",
                status_code=exc.response.status_code,
            )
        except Exception as exc:
            raise GetDeviceCodeError(
                message=f"Failed to request device code: {exc}",
                status_code=400,
            )

        device_auth_id = data.get("device_auth_id")
        user_code = data.get("user_code") or data.get("usercode")
        interval = data.get("interval")
        if not device_auth_id or not user_code:
            raise GetDeviceCodeError(
                message=f"Device code response missing fields: {data}",
                status_code=400,
            )
        return {
            "device_auth_id": device_auth_id,
            "user_code": user_code,
            "interval": str(interval or "5"),
        }

    def _poll_for_authorization_code(self, device_code: Dict[str, str]) -> Dict[str, str]:
        client = _get_httpx_client()
        interval = int(device_code.get("interval", "5"))
        start_time = time.time()
        while time.time() - start_time < DEVICE_CODE_TIMEOUT_SECONDS:
            try:
                resp = client.post(
                    CHATGPT_DEVICE_TOKEN_URL,
                    json={
                        "device_auth_id": device_code["device_auth_id"],
                        "user_code": device_code["user_code"],
                    },
                )
                if resp.status_code == 200:
                    data = resp.json()
                    if all(
                        key in data
                        for key in (
                            "authorization_code",
                            "code_challenge",
                            "code_verifier",
                        )
                    ):
                        return data
                if resp.status_code in (403, 404):
                    time.sleep(max(interval, DEVICE_CODE_POLL_SLEEP_SECONDS))
                    continue
                resp.raise_for_status()
            except httpx.HTTPStatusError as exc:
                status_code = exc.response.status_code if exc.response else None
                if status_code in (403, 404):
                    time.sleep(max(interval, DEVICE_CODE_POLL_SLEEP_SECONDS))
                    continue
                raise GetAccessTokenError(
                    message=f"Polling failed: {exc}",
                    status_code=exc.response.status_code,
                )
            except Exception as exc:
                raise GetAccessTokenError(
                    message=f"Polling failed: {exc}",
                    status_code=400,
                )
            time.sleep(max(interval, DEVICE_CODE_POLL_SLEEP_SECONDS))

        raise GetAccessTokenError(
            message="Timed out waiting for device authorization",
            status_code=408,
        )

    def _exchange_code_for_tokens(self, code_data: Dict[str, str]) -> Dict[str, str]:
        try:
            client = _get_httpx_client()
            redirect_uri = f"{CHATGPT_AUTH_BASE}/deviceauth/callback"
            body = (
                "grant_type=authorization_code"
                f"&code={code_data['authorization_code']}"
                f"&redirect_uri={redirect_uri}"
                f"&client_id={CHATGPT_CLIENT_ID}"
                f"&code_verifier={code_data['code_verifier']}"
            )
            resp = client.post(
                CHATGPT_OAUTH_TOKEN_URL,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                content=body,
            )
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPStatusError as exc:
            raise GetAccessTokenError(
                message=f"Token exchange failed: {exc}",
                status_code=exc.response.status_code,
            )
        except Exception as exc:
            raise GetAccessTokenError(
                message=f"Token exchange failed: {exc}",
                status_code=400,
            )

        if not all(key in data for key in ("access_token", "refresh_token", "id_token")):
            raise GetAccessTokenError(
                message=f"Token exchange response missing fields: {data}",
                status_code=400,
            )
        return {
            "access_token": data["access_token"],
            "refresh_token": data["refresh_token"],
            "id_token": data["id_token"],
        }

    def _refresh_tokens(self, refresh_token: str) -> Dict[str, str]:
        try:
            client = _get_httpx_client()
            resp = client.post(
                CHATGPT_OAUTH_TOKEN_URL,
                json={
                    "client_id": CHATGPT_CLIENT_ID,
                    "grant_type": "refresh_token",
                    "refresh_token": refresh_token,
                    "scope": "openid profile email",
                },
            )
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPStatusError as exc:
            raise RefreshAccessTokenError(
                message=f"Refresh token failed: {exc}",
                status_code=exc.response.status_code,
            )
        except Exception as exc:
            raise RefreshAccessTokenError(
                message=f"Refresh token failed: {exc}",
                status_code=400,
            )

        access_token = data.get("access_token")
        id_token = data.get("id_token")
        if not access_token or not id_token:
            raise RefreshAccessTokenError(
                message=f"Refresh response missing fields: {data}",
                status_code=400,
            )

        refreshed = {
            "access_token": access_token,
            "refresh_token": data.get("refresh_token", refresh_token),
            "id_token": id_token,
        }
        auth_data = self._build_auth_record(refreshed)
        self._write_auth_file(auth_data)
        return refreshed

    def _build_auth_record(self, tokens: Dict[str, str]) -> Dict[str, Any]:
        access_token = tokens.get("access_token")
        id_token = tokens.get("id_token")
        expires_at = self._get_expires_at(access_token) if access_token else None
        account_id = self._extract_account_id(id_token or access_token)
        return {
            "access_token": access_token,
            "refresh_token": tokens.get("refresh_token"),
            "id_token": id_token,
            "expires_at": expires_at,
            "account_id": account_id,
        }

    def _get_device_code_cooldown_remaining(
        self, auth_data: Optional[Dict[str, Any]]
    ) -> float:
        if not auth_data:
            return 0.0
        requested_at = auth_data.get("device_code_requested_at")
        if not isinstance(requested_at, (int, float, str)):
            return 0.0
        try:
            requested_at = float(requested_at)
        except (TypeError, ValueError):
            return 0.0
        elapsed = time.time() - requested_at
        remaining = DEVICE_CODE_COOLDOWN_SECONDS - elapsed
        return max(0.0, remaining)

    def _record_device_code_request(self) -> None:
        auth_data = self._read_auth_file() or {}
        auth_data["device_code_requested_at"] = time.time()
        self._write_auth_file(auth_data)

    def _wait_for_access_token(self, timeout_seconds: float) -> Optional[str]:
        deadline = time.time() + timeout_seconds
        while time.time() < deadline:
            auth_data = self._read_auth_file()
            if auth_data:
                access_token = auth_data.get("access_token")
                if access_token and not self._is_token_expired(
                    auth_data, access_token
                ):
                    return access_token
            sleep_for = min(DEVICE_CODE_POLL_SLEEP_SECONDS, max(0.0, deadline - time.time()))
            if sleep_for <= 0:
                break
            time.sleep(sleep_for)
        return None
