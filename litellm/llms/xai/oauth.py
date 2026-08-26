import base64
import hashlib
import json
import os
import secrets
import sys
import threading
import time
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Final
from urllib.parse import parse_qs, urlencode, urlparse

import httpx

from litellm._logging import verbose_logger
from litellm.constants import XAI_API_BASE
from litellm.llms.custom_httpx.http_handler import HTTPHandler, _get_httpx_client
from litellm.secret_managers.main import get_secret_str

XAI_OAUTH_ISSUER: Final = "https://auth.x.ai"
XAI_OAUTH_DISCOVERY_URL: Final = f"{XAI_OAUTH_ISSUER}/.well-known/openid-configuration"
XAI_OAUTH_CLIENT_ID: Final = "b1a00492-073a-47ea-816f-4c329264a828"
XAI_OAUTH_SCOPE: Final = "openid profile email offline_access grok-cli:access api:access"
XAI_OAUTH_REDIRECT_HOST: Final = "127.0.0.1"
XAI_OAUTH_REDIRECT_PORT: Final = 56121
XAI_OAUTH_REDIRECT_PATH: Final = "/callback"
XAI_OAUTH_EXPIRY_SKEW_SECONDS: Final = 120
XAI_OAUTH_CALLBACK_TIMEOUT_SECONDS: Final = 180
_XAI_OAUTH_REFRESH_LOCK: Final = threading.Lock()


class XAIOAuthError(Exception):
    pass


class XAIOAuthLoginRequiredError(XAIOAuthError):
    pass


class _CallbackHandler(BaseHTTPRequestHandler):
    server: "_CallbackServer"  # pyright: ignore[reportIncompatibleVariableOverride]  # stdlib stubs type server as BaseServer; _CallbackServer is the only server this handler is registered on

    def do_GET(self) -> None:
        parsed: Final = urlparse(self.path)
        if parsed.path != XAI_OAUTH_REDIRECT_PATH:
            self.send_response(404)
            self.end_headers()
            return

        params: Final = parse_qs(parsed.query)
        result: Final = {
            "code": params.get("code", [None])[0],
            "state": params.get("state", [None])[0],
            "error": params.get("error", [None])[0],
            "error_description": params.get("error_description", [None])[0],
        }
        self.server.callback_result = result

        if result["state"] != self.server.expected_state:
            self.send_response(400)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(b"<html><body><h1>xAI authorization state mismatch.</h1></body></html>")
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        body: Final = (
            b"<html><body><h1>xAI authorization failed.</h1>You can close this tab.</body></html>"
            if result["error"]
            else b"<html><body><h1>xAI authorization received.</h1>You can close this tab.</body></html>"
        )
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        return


class _CallbackServer(HTTPServer):
    expected_state: str
    callback_result: dict[str, str | None] | None


class XAIOAuthAuthenticator:
    def __init__(self, http_client: httpx.Client | HTTPHandler | None = None) -> None:
        self.token_dir = get_secret_str("XAI_OAUTH_TOKEN_DIR") or os.path.expanduser("~/.config/litellm/xai_oauth")
        self.auth_file = os.path.join(self.token_dir, get_secret_str("XAI_OAUTH_AUTH_FILE") or "auth.json")
        self.http_client = http_client

    def get_api_base(self) -> str:
        return get_secret_str("XAI_OAUTH_API_BASE") or get_secret_str("XAI_API_BASE") or XAI_API_BASE

    def get_access_token(self) -> str:
        auth_data: Final = self._read_auth_file()
        if not auth_data:
            raise XAIOAuthLoginRequiredError("xAI OAuth login required. Run `litellm xai-oauth login`.")

        access_token = auth_data.get("access_token")
        if access_token and not self._is_expired(auth_data):
            return access_token

        refresh_token: Final = auth_data.get("refresh_token")
        if not refresh_token:
            raise XAIOAuthLoginRequiredError("xAI OAuth refresh token missing. Run `litellm xai-oauth login`.")

        with _XAI_OAUTH_REFRESH_LOCK:
            locked_auth_data: Final = self._read_auth_file() or auth_data
            access_token = locked_auth_data.get("access_token")
            if access_token and not self._is_expired(locked_auth_data):
                return access_token

            refreshed: Final = self._refresh_tokens(locked_auth_data)
            return refreshed["access_token"]

    def login(self, force: bool = False, no_browser: bool = False) -> dict[str, Any]:
        existing: Final = self._read_auth_file()
        if existing and not force and existing.get("access_token"):
            if not self._is_expired(existing):
                return existing
            if existing.get("refresh_token"):
                try:
                    return self._refresh_tokens(existing)
                except XAIOAuthError:
                    pass

        discovery: Final = self._discover()
        verifier, challenge = self._pkce_pair()
        state: Final = uuid.uuid4().hex
        nonce: Final = uuid.uuid4().hex
        server, redirect_uri = self._start_callback_server(state)
        authorize_url: Final = self._build_authorize_url(
            authorization_endpoint=discovery["authorization_endpoint"],
            redirect_uri=redirect_uri,
            challenge=challenge,
            state=state,
            nonce=nonce,
        )

        if no_browser or not webbrowser.open(authorize_url):
            sys.stdout.write(f"Open this URL to authenticate with xAI:\n{authorize_url}\n")
            sys.stdout.flush()

        result: Final = self._wait_for_callback(server)
        if result.get("state") != state:
            raise XAIOAuthError("xAI OAuth state mismatch")
        if result.get("error"):
            description: Final = result.get("error_description") or result["error"]
            raise XAIOAuthError(f"xAI authorization failed: {description}")
        code: Final = result.get("code")
        if not code:
            raise XAIOAuthError("xAI authorization failed: no code returned")

        token_payload: Final = self._exchange_token(
            discovery["token_endpoint"],
            {
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": redirect_uri,
                "client_id": XAI_OAUTH_CLIENT_ID,
                "code_verifier": verifier,
            },
        )
        auth_data: Final = self._build_auth_record(token_payload, discovery["token_endpoint"])
        self._write_auth_file(auth_data)
        return auth_data

    def _client(self) -> httpx.Client | HTTPHandler:
        return self.http_client or _get_httpx_client()

    def _ensure_token_dir(self) -> None:
        os.makedirs(self.token_dir, mode=0o700, exist_ok=True)
        try:
            os.chmod(self.token_dir, 0o700)
        except OSError:
            verbose_logger.debug("Could not chmod xAI OAuth token directory")

    def _read_auth_file(self) -> dict[str, Any] | None:
        try:
            with open(self.auth_file, "r") as f:
                data: Final = json.load(f)
            return data if isinstance(data, dict) else None
        except (OSError, json.JSONDecodeError):
            return None

    def _write_auth_file(self, data: dict[str, Any]) -> None:
        self._ensure_token_dir()
        tmp_file: Final = os.path.join(
            self.token_dir,
            f".{os.path.basename(self.auth_file)}.{uuid.uuid4().hex}.tmp",
        )
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd: Final = os.open(tmp_file, flags, 0o600)
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(data, f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_file, self.auth_file)
            try:
                os.chmod(self.auth_file, 0o600)
            except OSError:
                verbose_logger.debug("Could not chmod xAI OAuth auth file")
        except Exception:
            try:
                os.close(fd)
            except OSError:
                pass
            try:
                os.unlink(tmp_file)
            except OSError:
                pass
            raise

    def _is_expired(self, auth_data: dict[str, Any]) -> bool:
        expires_at: Final = auth_data.get("expires_at")
        if expires_at is None:
            return True
        try:
            return time.time() >= float(expires_at) - XAI_OAUTH_EXPIRY_SKEW_SECONDS
        except (TypeError, ValueError):
            return True

    def _discover(self) -> dict[str, str]:
        try:
            response: Final = self._client().get(XAI_OAUTH_DISCOVERY_URL, headers={"Accept": "application/json"})
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise XAIOAuthError(
                f"xAI OAuth discovery request failed: {exc.response.status_code} {exc.response.text}"
            ) from exc
        try:
            data: Final = response.json()
        except ValueError as exc:
            raise XAIOAuthError("xAI OAuth discovery response was not valid JSON") from exc
        authorization_endpoint: Final = data.get("authorization_endpoint")
        token_endpoint: Final = data.get("token_endpoint")
        if not authorization_endpoint or not token_endpoint:
            raise XAIOAuthError("xAI OAuth discovery missing endpoints")
        return {
            "authorization_endpoint": self._validate_xai_endpoint(authorization_endpoint),
            "token_endpoint": self._validate_xai_endpoint(token_endpoint),
        }

    def _validate_xai_endpoint(self, url: str) -> str:
        parsed: Final = urlparse(url)
        host: Final = (parsed.hostname or "").lower()
        if parsed.scheme != "https" or (host != "x.ai" and not host.endswith(".x.ai")):
            raise XAIOAuthError(f"xAI OAuth discovery returned unexpected endpoint: {url}")
        return url

    def _pkce_pair(self) -> tuple[str, str]:
        verifier: Final = base64.urlsafe_b64encode(secrets.token_bytes(32)).rstrip(b"=").decode()
        challenge: Final = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
        return verifier, challenge

    def _start_callback_server(self, state: str) -> tuple[_CallbackServer, str]:
        last_error: OSError | None = None
        for port in (XAI_OAUTH_REDIRECT_PORT, 0):
            try:
                server = _CallbackServer((XAI_OAUTH_REDIRECT_HOST, port), _CallbackHandler)
                server.expected_state = state
                server.callback_result = None
                actual_port = server.server_address[1]
                redirect_uri = f"http://{XAI_OAUTH_REDIRECT_HOST}:{actual_port}{XAI_OAUTH_REDIRECT_PATH}"
                return server, redirect_uri
            except OSError as exc:
                last_error = exc
        raise XAIOAuthError(f"Could not start xAI OAuth callback server: {last_error}")

    def _build_authorize_url(
        self,
        authorization_endpoint: str,
        redirect_uri: str,
        challenge: str,
        state: str,
        nonce: str,
    ) -> str:
        params: Final = {
            "response_type": "code",
            "client_id": XAI_OAUTH_CLIENT_ID,
            "redirect_uri": redirect_uri,
            "scope": XAI_OAUTH_SCOPE,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "state": state,
            "nonce": nonce,
        }
        return f"{authorization_endpoint}?{urlencode(params)}"

    def _wait_for_callback(self, server: _CallbackServer) -> dict[str, str | None]:
        server.timeout = 1
        deadline: Final = time.time() + XAI_OAUTH_CALLBACK_TIMEOUT_SECONDS
        try:
            while time.time() < deadline:
                server.handle_request()
                if server.callback_result is not None:
                    return server.callback_result
        finally:
            server.server_close()
        raise XAIOAuthError("Timed out waiting for xAI OAuth callback")

    def _exchange_token(self, token_endpoint: str, data: dict[str, str]) -> dict[str, Any]:
        try:
            response: Final = self._client().post(
                token_endpoint,
                headers={
                    "Accept": "application/json",
                    "Content-Type": "application/x-www-form-urlencoded",
                },
                data=data,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise XAIOAuthError(
                f"xAI OAuth token request failed: {exc.response.status_code} {exc.response.text}"
            ) from exc
        try:
            body: Final = response.json()
        except ValueError as exc:
            raise XAIOAuthError("xAI OAuth token response was not valid JSON") from exc
        if not isinstance(body, dict):
            raise XAIOAuthError("xAI OAuth token response was not an object")
        return body

    def _build_auth_record(
        self,
        token_payload: dict[str, Any],
        token_endpoint: str,
        fallback_refresh_token: str | None = None,
    ) -> dict[str, Any]:
        access_token: Final = token_payload.get("access_token")
        refresh_token: Final = token_payload.get("refresh_token") or fallback_refresh_token
        if not access_token:
            raise XAIOAuthError("xAI OAuth token response missing access_token")
        if not refresh_token:
            raise XAIOAuthError("xAI OAuth token response missing refresh_token")
        expires_in: Final = token_payload.get("expires_in") or 3600
        try:
            expires_at = int(time.time() + int(expires_in))
        except (TypeError, ValueError):
            expires_at = int(time.time() + 3600)
        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "id_token": token_payload.get("id_token"),
            "token_type": token_payload.get("token_type") or "Bearer",
            "token_endpoint": token_endpoint,
            "expires_at": expires_at,
        }

    def _refresh_tokens(self, auth_data: dict[str, Any]) -> dict[str, Any]:
        token_endpoint = auth_data.get("token_endpoint")
        if not token_endpoint:
            token_endpoint = self._discover()["token_endpoint"]
        token_endpoint = self._validate_xai_endpoint(token_endpoint)
        refresh_token: Final = auth_data.get("refresh_token")
        if not refresh_token:
            raise XAIOAuthLoginRequiredError("xAI OAuth refresh token missing. Run `litellm xai-oauth login`.")

        token_payload: Final = self._exchange_token(
            token_endpoint,
            {
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "client_id": XAI_OAUTH_CLIENT_ID,
            },
        )
        refreshed: Final = self._build_auth_record(
            token_payload,
            token_endpoint,
            fallback_refresh_token=refresh_token,
        )
        self._write_auth_file(refreshed)
        return refreshed


def should_use_xai_oauth(litellm_params: dict[str, Any] | None) -> bool:
    return bool((litellm_params or {}).get("use_xai_oauth"))
