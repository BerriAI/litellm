import json
import os
import tempfile
import time
import webbrowser
import http.server
import html
import urllib.parse
import secrets
import hashlib
import base64
from typing import Any, Dict

import httpx

from litellm._logging import verbose_logger
from litellm.secret_managers.main import get_secret_str

GEMINI_SCOPES = [
    "https://www.googleapis.com/auth/cloud-platform",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
    "https://www.googleapis.com/auth/generative-language",
]

# Google OAuth URLs
GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"


class GeminiAuthenticator:
    def __init__(self) -> None:
        """Initialize the Gemini authenticator with configurable token paths."""
        # Token storage paths
        self.token_dir = os.getenv(
            "GEMINI_OAUTH_TOKEN_DIR",
            os.path.expanduser("~/.config/litellm/gemini_oauth"),
        )
        self.oauth_creds_file = os.path.join(
            self.token_dir,
            os.getenv("GEMINI_OAUTH_CREDS_FILE", "oauth_creds.json"),
        )
        self._ensure_token_dir()

    @staticmethod
    def _get_oauth_client_credentials() -> tuple[str, str]:
        client_id = get_secret_str("GEMINI_OAUTH_CLIENT_ID")
        client_secret = get_secret_str("GEMINI_OAUTH_CLIENT_SECRET")

        if not client_id or not client_secret:
            raise ValueError(
                "Missing Gemini OAuth client credentials. "
                "Set GEMINI_OAUTH_CLIENT_ID and GEMINI_OAUTH_CLIENT_SECRET."
            )
        return client_id, client_secret

    @staticmethod
    def _get_access_token_or_raise(creds: Dict[str, Any], source: str) -> str:
        token = creds.get("access_token")
        if not token or not isinstance(token, str):
            raise Exception(
                f"OAuth login failed: missing access_token in {source} response."
            )
        return token

    def get_token(self) -> str:
        """
        Get the OAuth token, refreshing if necessary.

        Returns:
            str: The Gemini access token.

        Raises:
            Exception: If unable to obtain or refresh an access token.
        """
        try:
            if os.path.exists(self.oauth_creds_file):
                with open(self.oauth_creds_file, "r") as f:
                    creds = json.load(f)

                    # Check if access_token exists and is valid (rough check)
                    if (
                        creds.get("access_token")
                        and creds.get("expires_at", 0) > time.time() + 60
                    ):
                        return creds.get("access_token")

                    # If expired but has refresh_token, try to refresh
                    if creds.get("refresh_token"):
                        verbose_logger.debug(
                            "Gemini access token expired, refreshing..."
                        )
                        return self._refresh_token(creds.get("refresh_token"))
        except Exception as e:
            verbose_logger.warning(f"Error reading Gemini OAuth credentials: {e}")

        # If we get here, we need to log in
        verbose_logger.info("Starting Gemini OAuth login flow...")
        creds = self._login()
        self._write_oauth_creds(creds)
        return self._get_access_token_or_raise(creds, "authorization_code")

    def _refresh_token(self, refresh_token: str) -> str:
        """Refresh the access token using the refresh token."""
        client_id, client_secret = self._get_oauth_client_credentials()
        data = {
            "client_id": client_id,
            "client_secret": client_secret,
            "refresh_token": refresh_token,
            "grant_type": "refresh_token",
        }
        resp = httpx.post(GOOGLE_TOKEN_URL, data=data)
        resp.raise_for_status()
        new_creds = resp.json()

        # Load existing creds to merge if possible
        creds = {}
        if os.path.exists(self.oauth_creds_file):
            try:
                with open(self.oauth_creds_file, "r") as f:
                    creds = json.load(f)
            except Exception:
                pass

        creds.update(new_creds)
        if "expires_in" in new_creds:
            creds["expires_at"] = time.time() + new_creds["expires_in"]

        self._write_oauth_creds(creds)

        return self._get_access_token_or_raise(creds, "refresh_token")

    def _ensure_token_dir(self) -> None:
        """Ensure the token directory exists."""
        if not os.path.exists(self.token_dir):
            os.makedirs(self.token_dir, mode=0o700, exist_ok=True)
        else:
            try:
                os.chmod(self.token_dir, 0o700)
            except OSError:
                pass

    def _write_oauth_creds(self, creds: Dict[str, Any]) -> None:
        """
        Write oauth credentials with user-only permissions.
        """
        fd, tmp_path = tempfile.mkstemp(dir=self.token_dir, prefix=".tmp_creds_")
        try:
            os.chmod(tmp_path, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(creds, f)
            os.replace(tmp_path, self.oauth_creds_file)
            try:
                os.chmod(self.oauth_creds_file, 0o600)
            except OSError:
                pass
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def _login(self) -> Dict[str, Any]:
        """Perform loopback flow login."""
        client_id, client_secret = self._get_oauth_client_credentials()
        # PKCE
        code_verifier = secrets.token_urlsafe(64)
        code_challenge = (
            base64.urlsafe_b64encode(hashlib.sha256(code_verifier.encode()).digest())
            .decode()
            .replace("=", "")
        )

        state = secrets.token_urlsafe(32)

        # Local web server for callback
        auth_code = None
        error = None

        class CallbackHandler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                nonlocal auth_code, error
                parsed_url = urllib.parse.urlparse(self.path)
                if parsed_url.path != "/oauth2callback":
                    self.send_response(404)
                    self.end_headers()
                    return

                query = parsed_url.query
                params = urllib.parse.parse_qs(query)

                if params.get("state", [None])[0] != state:
                    error = "state_mismatch"
                    self.send_response(400)
                    self.send_header("Content-type", "text/html")
                    self.end_headers()
                    self.wfile.write(
                        b"<html><body><h1>Authentication failed</h1><p>State mismatch. Possible CSRF attack.</p></body></html>"
                    )
                elif "code" in params:
                    auth_code = params["code"][0]
                    self.send_response(200)
                    self.send_header("Content-type", "text/html")
                    self.end_headers()
                    self.wfile.write(
                        b"<html><body><h1>Authentication successful!</h1><p>You can close this tab and return to the terminal.</p></body></html>"
                    )
                elif "error" in params:
                    error = params["error"][0]
                    self.send_response(200)
                    self.send_header("Content-type", "text/html")
                    self.end_headers()
                    self.wfile.write(
                        (
                            "<html><body><h1>Authentication failed</h1><p>"
                            f"{html.escape(error)}"
                            "</p></body></html>"
                        ).encode()
                    )
                else:
                    self.send_response(404)
                    self.end_headers()

            def log_message(self, format, *args):
                # Suppress logging to avoid noise in the terminal
                pass

        server = http.server.HTTPServer(("127.0.0.1", 0), CallbackHandler)
        port = server.server_port
        redirect_uri = f"http://127.0.0.1:{port}/oauth2callback"

        params = {
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "scope": " ".join(GEMINI_SCOPES),
            "state": state,
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
            "access_type": "offline",
            "prompt": "consent",
        }
        auth_url = f"{GOOGLE_AUTH_URL}?{urllib.parse.urlencode(params)}"

        print(  # noqa: T201
            f"Please visit the following URL to authenticate:\n\n{auth_url}\n"
        )
        webbrowser.open(auth_url)

        # Wait for callback; browsers may hit non-callback paths first (e.g. /favicon.ico).
        # Use a bounded loop so headless/CI environments fail fast instead of hanging forever.
        loopback_timeout_s = float(
            os.getenv("GEMINI_OAUTH_LOOPBACK_TIMEOUT_SECONDS", "120")
        )
        loopback_timeout_s = max(loopback_timeout_s, 1.0)
        deadline = time.monotonic() + loopback_timeout_s
        server.timeout = 1.0
        try:
            while auth_code is None and error is None:
                server.handle_request()
                if time.monotonic() >= deadline:
                    raise Exception(
                        "OAuth login timed out. Re-run `litellm-proxy gemini login` "
                        f"and complete the browser flow within {int(loopback_timeout_s)} seconds."
                    )
        finally:
            server.server_close()

        if error:
            raise Exception(f"OAuth login failed: {error}")
        if not auth_code:
            raise Exception("OAuth login failed: No code received")

        # Exchange code for token
        data = {
            "client_id": client_id,
            "client_secret": client_secret,
            "code": auth_code,
            "code_verifier": code_verifier,
            "grant_type": "authorization_code",
            "redirect_uri": redirect_uri,
        }
        resp = httpx.post(GOOGLE_TOKEN_URL, data=data)
        resp.raise_for_status()
        creds = resp.json()
        if "expires_in" in creds:
            creds["expires_at"] = time.time() + creds["expires_in"]

        return creds
