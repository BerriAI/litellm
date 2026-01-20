import base64
import hashlib
import json
import os
import time
from typing import Any, Dict, Optional
from urllib.parse import urlencode
from uuid import uuid4

import httpx

from litellm._logging import verbose_logger
from litellm.llms.custom_httpx.http_handler import _get_httpx_client

from .common_utils import (
    CredentialsClearRequiredError,
    GetAccessTokenError,
    GetDeviceCodeError,
    QWEN_DEFAULT_API_BASE,
    QWEN_OAUTH_CLIENT_ID,
    QWEN_OAUTH_CREDENTIAL_FILE,
    QWEN_OAUTH_DEVICE_CODE_ENDPOINT,
    QWEN_OAUTH_GRANT_TYPE,
    QWEN_OAUTH_SCOPE,
    QWEN_OAUTH_TOKEN_ENDPOINT,
    QWEN_TOKEN_DIRNAME,
    RefreshAccessTokenError,
    normalize_qwen_api_base,
)

TOKEN_EXPIRY_SKEW_SECONDS = 60
DEVICE_CODE_TIMEOUT_SECONDS = 15 * 60
DEVICE_CODE_COOLDOWN_SECONDS = 5 * 60
DEVICE_CODE_POLL_INTERVAL_SECONDS = 2
DEVICE_CODE_POLL_MAX_INTERVAL_SECONDS = 10


def _base64url_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("utf-8").rstrip("=")


def _generate_code_verifier() -> str:
    return _base64url_encode(os.urandom(32))


def _generate_code_challenge(code_verifier: str) -> str:
    digest = hashlib.sha256(code_verifier.encode("utf-8")).digest()
    return _base64url_encode(digest)


class Authenticator:
    def __init__(self) -> None:
        self.token_dir = os.getenv(
            "QWEN_TOKEN_DIR",
            os.path.expanduser(f"~/{QWEN_TOKEN_DIRNAME}"),
        )
        auth_file = os.getenv("QWEN_AUTH_FILE", QWEN_OAUTH_CREDENTIAL_FILE)
        self.auth_file = os.path.join(self.token_dir, auth_file)
        self._ensure_token_dir()

    def get_access_token(self) -> str:
        credentials = self._read_credentials()
        if credentials:
            access_token = credentials.get("access_token")
            if access_token and not self._is_token_expired(credentials):
                return access_token
            refresh_token = credentials.get("refresh_token")
            if refresh_token:
                try:
                    refreshed = self._refresh_tokens(refresh_token, credentials)
                    return refreshed["access_token"]
                except (RefreshAccessTokenError, CredentialsClearRequiredError) as exc:
                    verbose_logger.warning(
                        "Qwen refresh token failed, re-login required: %s",
                        exc,
                    )

        cooldown_remaining = self._get_device_code_cooldown_remaining(credentials)
        if cooldown_remaining > 0:
            token = self._wait_for_access_token(cooldown_remaining)
            if token:
                return token

        tokens = self._login_device_code()
        return tokens["access_token"]

    def get_api_base(self) -> str:
        env_base = normalize_qwen_api_base(os.getenv("QWEN_API_BASE"))
        if env_base:
            return env_base
        credentials = self._read_credentials() or {}
        resource_url = credentials.get("resource_url")
        normalized = normalize_qwen_api_base(resource_url)
        return normalized or QWEN_DEFAULT_API_BASE

    def _ensure_token_dir(self) -> None:
        if not os.path.exists(self.token_dir):
            os.makedirs(self.token_dir, exist_ok=True)

    def _read_credentials(self) -> Optional[Dict[str, Any]]:
        try:
            with open(self.auth_file, "r") as f:
                return json.load(f)
        except IOError:
            return None
        except json.JSONDecodeError as exc:
            verbose_logger.warning("Invalid Qwen auth file: %s", exc)
            return None

    def _write_credentials(self, data: Dict[str, Any]) -> None:
        try:
            with open(self.auth_file, "w") as f:
                json.dump(data, f)
        except IOError as exc:
            verbose_logger.error("Failed to write Qwen auth file: %s", exc)

    def _clear_credentials(self) -> None:
        try:
            if os.path.exists(self.auth_file):
                os.remove(self.auth_file)
        except OSError as exc:
            verbose_logger.warning("Failed to clear Qwen auth file: %s", exc)

    def _is_token_expired(self, credentials: Dict[str, Any]) -> bool:
        expiry_date = credentials.get("expiry_date")
        if expiry_date is None:
            return True
        try:
            expiry_ms = int(expiry_date)
        except (TypeError, ValueError):
            return True
        now_ms = int(time.time() * 1000)
        return now_ms >= expiry_ms - (TOKEN_EXPIRY_SKEW_SECONDS * 1000)

    def _get_device_code_cooldown_remaining(
        self, credentials: Optional[Dict[str, Any]]
    ) -> float:
        if not credentials:
            return 0.0
        requested_at = credentials.get("device_code_requested_at")
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
        credentials = self._read_credentials() or {}
        credentials["device_code_requested_at"] = time.time()
        self._write_credentials(credentials)

    def _wait_for_access_token(self, timeout_seconds: float) -> Optional[str]:
        deadline = time.time() + timeout_seconds
        while time.time() < deadline:
            credentials = self._read_credentials()
            if credentials:
                access_token = credentials.get("access_token")
                if access_token and not self._is_token_expired(credentials):
                    return access_token
            sleep_for = min(
                DEVICE_CODE_POLL_INTERVAL_SECONDS, max(0.0, deadline - time.time())
            )
            if sleep_for <= 0:
                break
            time.sleep(sleep_for)
        return None

    def _refresh_tokens(
        self, refresh_token: str, credentials: Dict[str, Any]
    ) -> Dict[str, Any]:
        body = {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": QWEN_OAUTH_CLIENT_ID,
        }
        try:
            client = _get_httpx_client()
            resp = client.post(
                QWEN_OAUTH_TOKEN_ENDPOINT,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                content=urlencode(body),
            )
            if resp.status_code == 400:
                self._clear_credentials()
                raise CredentialsClearRequiredError(
                    message="Refresh token expired or invalid. Please re-authenticate.",
                    status_code=resp.status_code,
                )
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPStatusError as exc:
            raise RefreshAccessTokenError(
                message=f"Refresh token failed: {exc}",
                status_code=exc.response.status_code,
            )
        except CredentialsClearRequiredError:
            raise
        except Exception as exc:
            raise RefreshAccessTokenError(
                message=f"Refresh token failed: {exc}",
                status_code=400,
            )

        access_token = data.get("access_token")
        if not access_token:
            raise RefreshAccessTokenError(
                message=f"Refresh response missing access_token: {data}",
                status_code=400,
            )

        refreshed = {
            "access_token": access_token,
            "refresh_token": data.get("refresh_token") or refresh_token,
            "token_type": data.get("token_type"),
            "resource_url": data.get("resource_url") or credentials.get("resource_url"),
            "expiry_date": (
                int(time.time() * 1000) + int(data.get("expires_in", 0)) * 1000
                if data.get("expires_in") is not None
                else None
            ),
        }
        self._write_credentials(refreshed)
        return refreshed

    def _login_device_code(self) -> Dict[str, Any]:
        cooldown_remaining = self._get_device_code_cooldown_remaining(
            self._read_credentials()
        )
        if cooldown_remaining > 0:
            token = self._wait_for_access_token(cooldown_remaining)
            if token:
                return {"access_token": token}

        code_verifier = _generate_code_verifier()
        code_challenge = _generate_code_challenge(code_verifier)
        device_code = self._request_device_code(code_challenge)
        self._record_device_code_request()
        verification_uri = device_code.get("verification_uri")
        verification_uri_complete = device_code.get(
            "verification_uri_complete", verification_uri
        )
        user_code = device_code.get("user_code")
        if verification_uri_complete:
            print(  # noqa: T201
                "Sign in with Qwen using device code:\n"
                f"{verification_uri_complete}\n"
                "Device codes are a common phishing target. Never share this code.",
                flush=True,
            )
        elif verification_uri and user_code:
            print(  # noqa: T201
                "Sign in with Qwen using device code:\n"
                f"1) Visit {verification_uri}\n"
                f"2) Enter code: {user_code}\n"
                "Device codes are a common phishing target. Never share this code.",
                flush=True,
            )
        tokens = self._poll_for_tokens(
            device_code=device_code["device_code"],
            code_verifier=code_verifier,
            expires_in=int(device_code.get("expires_in", DEVICE_CODE_TIMEOUT_SECONDS)),
        )
        self._write_credentials(tokens)
        return tokens

    def _request_device_code(self, code_challenge: str) -> Dict[str, Any]:
        body = {
            "client_id": QWEN_OAUTH_CLIENT_ID,
            "scope": QWEN_OAUTH_SCOPE,
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
        }
        try:
            client = _get_httpx_client()
            resp = client.post(
                QWEN_OAUTH_DEVICE_CODE_ENDPOINT,
                headers={
                    "Content-Type": "application/x-www-form-urlencoded",
                    "Accept": "application/json",
                    "x-request-id": str(uuid4()),
                },
                content=urlencode(body),
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

        required_fields = {"device_code", "user_code", "verification_uri", "expires_in"}
        if not required_fields.issubset(set(data.keys())):
            raise GetDeviceCodeError(
                message=f"Device code response missing fields: {data}",
                status_code=400,
            )
        return data

    def _poll_for_tokens(
        self, device_code: str, code_verifier: str, expires_in: int
    ) -> Dict[str, Any]:
        poll_interval: float = float(DEVICE_CODE_POLL_INTERVAL_SECONDS)
        timeout_seconds = max(expires_in, DEVICE_CODE_TIMEOUT_SECONDS)
        deadline = time.time() + timeout_seconds
        client = _get_httpx_client()
        while time.time() < deadline:
            body = {
                "grant_type": QWEN_OAUTH_GRANT_TYPE,
                "client_id": QWEN_OAUTH_CLIENT_ID,
                "device_code": device_code,
                "code_verifier": code_verifier,
            }
            try:
                resp = client.post(
                    QWEN_OAUTH_TOKEN_ENDPOINT,
                    headers={
                        "Content-Type": "application/x-www-form-urlencoded",
                        "Accept": "application/json",
                    },
                    content=urlencode(body),
                )
                data = resp.json()
            except httpx.HTTPStatusError as exc:
                response = exc.response
                error_data = None
                if response is not None:
                    try:
                        error_data = response.json()
                    except Exception:
                        error_data = None
                if (
                    response is not None
                    and response.status_code == 400
                    and error_data
                    and error_data.get("error") == "authorization_pending"
                ):
                    poll_interval = DEVICE_CODE_POLL_INTERVAL_SECONDS
                    time.sleep(poll_interval)
                    continue
                if (
                    response is not None
                    and response.status_code == 429
                    and error_data
                    and error_data.get("error") == "slow_down"
                ):
                    poll_interval = min(
                        poll_interval * 1.5, DEVICE_CODE_POLL_MAX_INTERVAL_SECONDS
                    )
                    time.sleep(poll_interval)
                    continue
                raise GetAccessTokenError(
                    message=f"Device token poll failed: {exc}",
                    status_code=response.status_code if response is not None else 400,
                )
            except Exception as exc:
                raise GetAccessTokenError(
                    message=f"Device token poll failed: {exc}",
                    status_code=400,
                )

            access_token = data.get("access_token")
            if not access_token:
                error = data.get("error") if isinstance(data, dict) else None
                if error == "authorization_pending":
                    poll_interval = DEVICE_CODE_POLL_INTERVAL_SECONDS
                    time.sleep(poll_interval)
                    continue
                if error == "slow_down":
                    poll_interval = min(
                        poll_interval * 1.5, DEVICE_CODE_POLL_MAX_INTERVAL_SECONDS
                    )
                    time.sleep(poll_interval)
                    continue
                raise GetAccessTokenError(
                    message=f"Device token response missing access_token: {data}",
                    status_code=400,
                )

            return {
                "access_token": access_token,
                "refresh_token": data.get("refresh_token"),
                "token_type": data.get("token_type"),
                "resource_url": data.get("resource_url"),
                "expiry_date": (
                    int(time.time() * 1000) + int(data.get("expires_in", 0)) * 1000
                    if data.get("expires_in") is not None
                    else None
                ),
            }

        raise GetAccessTokenError(
            message="Timed out waiting for device authorization",
            status_code=408,
        )
