"""
Authorization Code + PKCE OAuth flow for the ChatGPT / Codex backend.

Kept in its own module so the upstream-facing ``authenticator.py`` (which
already hosts the device-code flow) stays nearly untouched — minimising
merge conflicts when syncing from BerriAI/litellm.
"""

import base64
import hashlib
import html
import http.server
import secrets
import threading
import urllib.parse
import webbrowser
from typing import TYPE_CHECKING, Any, Dict

import httpx

from litellm._logging import verbose_logger
from litellm.llms.custom_httpx.http_handler import _get_httpx_client

from .common_utils import (
    CHATGPT_AUTH_BASE,
    CHATGPT_CLIENT_ID,
    CHATGPT_OAUTH_TOKEN_URL,
    GetAccessTokenError,
    get_chatgpt_originator,
)

if TYPE_CHECKING:
    from .authenticator import Authenticator


OAUTH_AUTHORIZE_URL = f"{CHATGPT_AUTH_BASE}/oauth/authorize"
REDIRECT_HOST = "127.0.0.1"
REDIRECT_PORT = 1455
REDIRECT_PATH = "/auth/callback"
SCOPE = "openid profile email offline_access"
LOGIN_TIMEOUT_SECONDS = 10 * 60

_SUCCESS_HTML = (
    "<!doctype html><html><head><title>LiteLLM login complete</title></head>"
    '<body style="font-family: sans-serif; max-width: 480px; margin: 3rem auto;">'
    "<h2>Sign-in complete</h2>"
    "<p>You can close this tab and return to your terminal.</p>"
    "</body></html>"
)
_ERROR_HTML = (
    "<!doctype html><html><head><title>LiteLLM login failed</title></head>"
    '<body style="font-family: sans-serif; max-width: 480px; margin: 3rem auto;">'
    "<h2>Sign-in failed</h2>"
    "<p>{message}</p>"
    "<p>Return to your terminal for details.</p>"
    "</body></html>"
)


def login_pkce(
    authenticator: "Authenticator",
    open_browser: bool = True,
    port: int = REDIRECT_PORT,
    timeout_seconds: float = LOGIN_TIMEOUT_SECONDS,
) -> Dict[str, str]:
    """
    Run the full PKCE + loopback OAuth flow and persist tokens via the
    authenticator (reusing ``_build_auth_record``/``_write_auth_file`` so the
    on-disk format matches the device-code flow).
    """
    code_verifier = _generate_code_verifier()
    code_challenge = _generate_code_challenge(code_verifier)
    state = secrets.token_hex(16)
    redirect_uri = f"http://{REDIRECT_HOST}:{port}{REDIRECT_PATH}"
    authorize_url = _build_authorize_url(
        redirect_uri=redirect_uri,
        code_challenge=code_challenge,
        state=state,
    )

    result: Dict[str, Any] = {}
    completed = threading.Event()
    handler_cls = _make_handler(result, completed, expected_state=state)

    try:
        server = http.server.HTTPServer((REDIRECT_HOST, port), handler_cls)
    except OSError as exc:
        raise GetAccessTokenError(
            message=(
                f"Failed to bind loopback server on {REDIRECT_HOST}:{port}: {exc}. "
                "Pass a different port or close the process holding it."
            ),
            status_code=400,
        )

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        print(  # noqa: T201
            "Sign in with ChatGPT in your browser.\n"
            f"If the browser does not open, visit:\n{authorize_url}",
            flush=True,
        )
        if open_browser:
            try:
                webbrowser.open(authorize_url)
            except Exception as exc:
                verbose_logger.debug("webbrowser.open failed: %s", exc)

        if not completed.wait(timeout=timeout_seconds):
            raise GetAccessTokenError(
                message="Timed out waiting for OAuth callback",
                status_code=408,
            )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    if "error" in result:
        raise GetAccessTokenError(
            message=f"OAuth callback error: {result['error']}",
            status_code=400,
        )
    code = result.get("code")
    if not code:
        raise GetAccessTokenError(
            message="OAuth callback did not include an authorization code",
            status_code=400,
        )

    tokens = _exchange_code_for_tokens(
        code=code, code_verifier=code_verifier, redirect_uri=redirect_uri
    )
    auth_data = authenticator._build_auth_record(tokens)
    authenticator._write_auth_file(auth_data)
    return tokens


def _exchange_code_for_tokens(
    code: str, code_verifier: str, redirect_uri: str
) -> Dict[str, str]:
    try:
        client = _get_httpx_client()
        resp = client.post(
            CHATGPT_OAUTH_TOKEN_URL,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data={
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": redirect_uri,
                "client_id": CHATGPT_CLIENT_ID,
                "code_verifier": code_verifier,
            },
        )
        resp.raise_for_status()
        data = resp.json()
    except httpx.HTTPStatusError as exc:
        raise GetAccessTokenError(
            message=f"PKCE token exchange failed: {exc}",
            status_code=exc.response.status_code,
        )
    except Exception as exc:
        raise GetAccessTokenError(
            message=f"PKCE token exchange failed: {exc}",
            status_code=400,
        )

    if not all(key in data for key in ("access_token", "refresh_token", "id_token")):
        raise GetAccessTokenError(
            message=f"PKCE token response missing fields: {data}",
            status_code=400,
        )
    return {
        "access_token": data["access_token"],
        "refresh_token": data["refresh_token"],
        "id_token": data["id_token"],
    }


def _generate_code_verifier() -> str:
    """RFC 7636 PKCE verifier: 43-128 chars from the unreserved set."""
    return secrets.token_urlsafe(64)[:64]


def _generate_code_challenge(verifier: str) -> str:
    """S256 challenge: base64url(sha256(verifier)) without padding."""
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def _build_authorize_url(redirect_uri: str, code_challenge: str, state: str) -> str:
    params = {
        "response_type": "code",
        "client_id": CHATGPT_CLIENT_ID,
        "redirect_uri": redirect_uri,
        "scope": SCOPE,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
        "state": state,
        "id_token_add_organizations": "true",
        "codex_cli_simplified_flow": "true",
        "originator": get_chatgpt_originator(),
    }
    return f"{OAUTH_AUTHORIZE_URL}?{urllib.parse.urlencode(params)}"


def _make_handler(
    result: Dict[str, Any],
    completed: threading.Event,
    expected_state: str,
) -> type:
    class _Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path != REDIRECT_PATH:
                self.send_response(404)
                self.end_headers()
                return
            params = urllib.parse.parse_qs(parsed.query)
            error = params.get("error", [None])[0]
            if error:
                result["error"] = params.get("error_description", [error])[0]
                self._respond_error(result["error"])
                completed.set()
                return
            state = params.get("state", [None])[0]
            code = params.get("code", [None])[0]
            if state != expected_state:
                result["error"] = "state mismatch"
                self._respond_error("state mismatch")
                completed.set()
                return
            if not code:
                result["error"] = "missing code"
                self._respond_error("missing code")
                completed.set()
                return
            result["code"] = code
            self._respond_html(200, _SUCCESS_HTML)
            completed.set()

        def _respond_error(self, message: str) -> None:
            # ``message`` can originate from the IdP's ``error_description``
            # query param, so escape before interpolating into HTML.
            self._respond_html(400, _ERROR_HTML.format(message=html.escape(message)))

        def _respond_html(self, status: int, body: str) -> None:
            payload = body.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args: Any, **kwargs: Any) -> None:
            return

    return _Handler
