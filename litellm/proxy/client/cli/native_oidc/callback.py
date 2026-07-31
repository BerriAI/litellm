"""Loopback redirect listener for the Authorization Code flow (RFC 8252).

Hard requirements enforced here:

- binds only a numeric loopback interface, never 0.0.0.0 / ::
- takes an ephemeral port from the OS and builds the redirect URI from the
  address actually bound
- serves exactly one dedicated callback path
- rejects non-loopback peers and mismatched Host headers
- validates `state` in constant time before the authorization code is used
- never logs the request line, never renders the code or state into the page
"""

import socket
import socketserver
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlsplit

from litellm.litellm_core_utils.native_oidc_validation import is_numeric_loopback_host

from .errors import NativeOIDCError
from .pkce import states_match

DEFAULT_CALLBACK_PATH = "/oauth/callback"

# Provider-controlled text: bounded and reduced to printable ASCII before it is
# ever shown. The stable OAuth error code is preferred over the description.
MAX_ERROR_DESCRIPTION_LENGTH = 200

_SUCCESS_PAGE = (
    b"<!doctype html><html><head><title>LiteLLM</title></head><body>"
    b"<h1>Login complete</h1><p>You can close this window and return to the terminal.</p>"
    b"</body></html>"
)

_FAILURE_PAGE = (
    b"<!doctype html><html><head><title>LiteLLM</title></head><body>"
    b"<h1>Login failed</h1><p>Return to the terminal for details.</p>"
    b"</body></html>"
)


def sanitize_provider_error(error: str, description: str | None) -> str:
    """Build a bounded, printable message from a provider error response."""
    safe_error = "".join(c for c in error if 0x20 <= ord(c) < 0x7F)[:MAX_ERROR_DESCRIPTION_LENGTH]
    message = safe_error or "unknown_error"
    if description:
        safe_description = "".join(c for c in description if 0x20 <= ord(c) < 0x7F)[:MAX_ERROR_DESCRIPTION_LENGTH]
        if safe_description:
            message = f"{message}: {safe_description}"
    return message


def _single_value(params: dict, name: str) -> str | None:
    """Return the sole value for `name`, or raise if it appears more than once."""
    values = params.get(name)
    if not values:
        return None
    if len(values) != 1:
        raise NativeOIDCError(f"authorization response contained multiple '{name}' values")
    return values[0]


class _CallbackHandler(BaseHTTPRequestHandler):
    server_version = "LiteLLMCLI/1.0"

    def log_message(self, format: str, *args) -> None:
        """Suppress request logging.

        The default implementation writes the full request line to stderr, which
        would leak the authorization code and state.
        """

    def _respond(self, status: int, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        server: _CallbackServer = self.server  # type: ignore[assignment]

        peer_host = self.client_address[0]
        if not is_numeric_loopback_host(peer_host):
            self._respond(403, _FAILURE_PAGE)
            return

        host_header = self.headers.get("Host", "")
        if host_header and host_header != server.expected_host_header:
            self._respond(400, _FAILURE_PAGE)
            return

        if urlsplit(self.path).path != server.callback_path:
            self._respond(404, _FAILURE_PAGE)
            return

        if server.result is not None:
            # Exactly one authorization response is accepted.
            self._respond(400, _FAILURE_PAGE)
            return

        params = parse_qs(urlsplit(self.path).query, keep_blank_values=True)
        try:
            state = _single_value(params, "state")
            code = _single_value(params, "code")
            error = _single_value(params, "error")
            description = _single_value(params, "error_description")
        except NativeOIDCError as exc:
            server.result = _CallbackResult(error=str(exc))
            self._respond(400, _FAILURE_PAGE)
            return

        if state is None or not states_match(server.expected_state, state):
            # Validated before the code is looked at, let alone redeemed.
            server.result = _CallbackResult(error="authorization response state was missing or did not match")
            self._respond(400, _FAILURE_PAGE)
            return

        if error is not None:
            server.result = _CallbackResult(
                error="identity provider returned " + sanitize_provider_error(error, description)
            )
            self._respond(400, _FAILURE_PAGE)
            return

        if not code:
            server.result = _CallbackResult(error="authorization response did not contain a code")
            self._respond(400, _FAILURE_PAGE)
            return

        server.result = _CallbackResult(code=code)
        self._respond(200, _SUCCESS_PAGE)


class _CallbackResult:
    __slots__ = ("code", "error")

    def __init__(self, code: str | None = None, error: str | None = None) -> None:
        self.code = code
        self.error = error


class _CallbackServer(HTTPServer):
    # A fresh ephemeral port every time; never inherit a lingering socket.
    allow_reuse_address = False

    def __init__(self, server_address, handler, *, expected_state: str, callback_path: str):
        self.expected_state = expected_state
        self.callback_path = callback_path
        self.result: _CallbackResult | None = None
        self.expected_host_header = ""
        super().__init__(server_address, handler)

    def server_bind(self) -> None:
        # Skip HTTPServer.server_bind's socket.getfqdn() lookup: it is useless
        # for a loopback listener and can block on a slow resolver.
        socketserver.TCPServer.server_bind(self)
        host, port = self.server_address[:2]
        self.server_name = str(host)
        self.server_port = port

    def handle_error(self, request, client_address) -> None:
        """Swallow handler tracebacks; they can contain the request path."""


class _CallbackServerV6(_CallbackServer):
    address_family = socket.AF_INET6


class LoopbackCallbackListener:
    """A one-shot loopback listener bound to an OS-assigned ephemeral port."""

    def __init__(self, expected_state: str, callback_path: str = DEFAULT_CALLBACK_PATH):
        self._server = self._bind(expected_state, callback_path)
        host, port = self._server.server_address[:2]
        self.host = str(host)
        self.port = int(port)
        self._server.expected_host_header = self._format_authority()

    @staticmethod
    def _bind(expected_state: str, callback_path: str) -> _CallbackServer:
        """Prefer IPv4 loopback for predictable registration; fall back to ::1."""
        try:
            return _CallbackServer(
                ("127.0.0.1", 0),
                _CallbackHandler,
                expected_state=expected_state,
                callback_path=callback_path,
            )
        except OSError:
            try:
                return _CallbackServerV6(
                    ("::1", 0),
                    _CallbackHandler,
                    expected_state=expected_state,
                    callback_path=callback_path,
                )
            except OSError as error:
                raise NativeOIDCError("could not bind a loopback port for the login callback") from error

    def _format_authority(self) -> str:
        if ":" in self.host:
            return f"[{self.host}]:{self.port}"
        return f"{self.host}:{self.port}"

    @property
    def redirect_uri(self) -> str:
        """Built from the address actually bound, not from a guess."""
        return f"http://{self._format_authority()}{self._server.callback_path}"

    def wait_for_code(self, timeout: float) -> str:
        """Block until a valid authorization code arrives, or fail.

        Returns the code; the caller redeems it. The code is never logged here.
        """
        deadline = time.monotonic() + timeout
        self._server.timeout = 0.5
        while time.monotonic() < deadline:
            if self._server.result is not None:
                break
            self._server.handle_request()

        result = self._server.result
        if result is None:
            raise NativeOIDCError(f"timed out after {int(timeout)}s waiting for the browser login callback")
        if result.error is not None:
            raise NativeOIDCError(result.error)
        if not result.code:
            raise NativeOIDCError("authorization response did not contain a code")
        return result.code

    def close(self) -> None:
        self._server.server_close()

    def __enter__(self) -> "LoopbackCallbackListener":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()
