"""End-to-end native OIDC login against a mock identity provider.

Everything below runs over real loopback sockets: the proxy discovery document,
the OpenID provider configuration, the authorization redirect, the token
endpoint, and the `/v1/models` verification probe. Only the browser is
simulated -- `webbrowser.open` is replaced by a client that follows the
authorization redirect the way a real browser would.

These are the tests that would catch a regression the module-level suites
cannot: a wrong URL being derived, a header not being sent, a redirect being
followed that should not be, or a credential written in a shape the refresh
path cannot read back.
"""

import functools
import json
import threading
import time
from http.client import HTTPConnection
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

import pytest

from litellm.proxy.client.cli.native_oidc import browser_flow
from litellm.proxy.client.cli.native_oidc.credentials import (
    AUTH_TYPE_NATIVE_OIDC,
    TOKEN_SCHEMA_VERSION,
    is_native_credential,
    needs_refresh,
    refresh_native_credential,
    save_credential,
)
from litellm.proxy.client.cli.native_oidc.errors import (
    NativeOIDCAuthRejected,
    NativeOIDCError,
)
from litellm.proxy.client.cli.native_oidc.login import (
    FLOW_BROWSER,
    FLOW_DEVICE,
    run_native_login,
)
from litellm.proxy.client.cli.native_oidc.pkce import compute_code_challenge

CLIENT_ID = "litellm-cli"
SCOPES = ["openid", "profile"]
DEVICE_GRANT = "urn:ietf:params:oauth:grant-type:device_code"


class MockIdentityProvider:
    """A LiteLLM proxy and an OpenID provider sharing one loopback origin."""

    def __init__(self):
        self.issued_access_tokens = set()
        self.authorization_codes = {}
        self.refresh_tokens = {}
        self.requests = []
        self.device_polls = 0
        self.device_polls_until_approval = 1
        self.reject_at_litellm = False
        self.omit_refresh_token = False
        self.native_oidc = None  # set once the port is known
        self._counter = 0

    def next_value(self, prefix):
        self._counter += 1
        return f"{prefix}-{self._counter}"

    # -- documents ---------------------------------------------------------- #

    def discovery_document(self):
        return {
            "sso_enabled": True,
            "native_oidc": self.native_oidc,
        }

    def provider_document(self):
        return {
            "issuer": self.base_url,
            "authorization_endpoint": f"{self.base_url}/authorize",
            "token_endpoint": f"{self.base_url}/token",
            "device_authorization_endpoint": f"{self.base_url}/device",
            "response_types_supported": ["code"],
            "grant_types_supported": [
                "authorization_code",
                "refresh_token",
                DEVICE_GRANT,
            ],
            "code_challenge_methods_supported": ["S256"],
            "token_endpoint_auth_methods_supported": ["none"],
            "jwks_uri": f"{self.base_url}/jwks",
        }

    # -- grants ------------------------------------------------------------- #

    def issue_tokens(self):
        access_token = self.next_value("access-token")
        self.issued_access_tokens.add(access_token)
        body = {
            "access_token": access_token,
            "token_type": "Bearer",
            "expires_in": 3600,
        }
        if not self.omit_refresh_token:
            refresh_token = self.next_value("refresh-token")
            self.refresh_tokens[refresh_token] = access_token
            body["refresh_token"] = refresh_token
        return body

    def handle_token(self, form):
        grant_type = form.get("grant_type", [None])[0]
        if grant_type == "authorization_code":
            code = form.get("code", [None])[0]
            entry = self.authorization_codes.pop(code, None)
            if entry is None:
                return 400, {"error": "invalid_grant"}
            # PKCE is verified for real: the verifier must hash to the challenge.
            verifier = form.get("code_verifier", [""])[0]
            if compute_code_challenge(verifier) != entry["code_challenge"]:
                return 400, {"error": "invalid_grant"}
            if form.get("redirect_uri", [None])[0] != entry["redirect_uri"]:
                return 400, {"error": "invalid_grant"}
            return 200, self.issue_tokens()
        if grant_type == "refresh_token":
            token = form.get("refresh_token", [None])[0]
            if token not in self.refresh_tokens:
                return 400, {"error": "invalid_grant"}
            del self.refresh_tokens[token]
            return 200, self.issue_tokens()
        if grant_type == DEVICE_GRANT:
            self.device_polls += 1
            if self.device_polls < self.device_polls_until_approval:
                return 400, {"error": "authorization_pending"}
            return 200, self.issue_tokens()
        return 400, {"error": "unsupported_grant_type"}


class MockServer(ThreadingHTTPServer):
    # Handler threads must not outlive the test: with the default non-daemon
    # threads, `server_close()` blocks forever joining a keep-alive connection.
    daemon_threads = True


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    @property
    def idp(self):
        return self.server.idp

    def log_message(self, *args):
        pass

    def _send(self, status, payload=None, extra_headers=None):
        body = b"" if payload is None else json.dumps(payload).encode()
        self.send_response(status)
        if payload is not None:
            self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.close_connection = True
        for name, value in (extra_headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        if body:
            self.wfile.write(body)

    def do_GET(self):
        parsed = urlsplit(self.path)
        self.idp.requests.append(("GET", parsed.path))

        if parsed.path == "/.well-known/litellm-ui-config":
            return self._send(200, self.idp.discovery_document())
        if parsed.path == "/.well-known/openid-configuration":
            return self._send(200, self.idp.provider_document())
        if parsed.path == "/authorize":
            return self._authorize(parse_qs(parsed.query))
        if parsed.path == "/v1/models":
            return self._models()
        return self._send(404, {"error": "not_found"})

    def do_POST(self):
        parsed = urlsplit(self.path)
        self.idp.requests.append(("POST", parsed.path))
        length = int(self.headers.get("Content-Length", "0"))
        form = parse_qs(self.rfile.read(length).decode())

        if parsed.path == "/token":
            status, payload = self.idp.handle_token(form)
            return self._send(status, payload)
        if parsed.path == "/device":
            code = self.idp.next_value("device-code")
            return self._send(
                200,
                {
                    "device_code": code,
                    "user_code": "WXYZ-1234",
                    "verification_uri": f"{self.idp.base_url}/activate",
                    "expires_in": 600,
                    "interval": 1,
                },
            )
        return self._send(404, {"error": "not_found"})

    def _authorize(self, query):
        code = self.idp.next_value("auth-code")
        self.idp.authorization_codes[code] = {
            "code_challenge": query["code_challenge"][0],
            "redirect_uri": query["redirect_uri"][0],
        }
        location = f"{query['redirect_uri'][0]}?code={code}&state={query['state'][0]}"
        self._send(302, None, {"Location": location})

    def _models(self):
        authorization = self.headers.get("Authorization", "")
        token = (
            authorization[len("Bearer ") :]
            if authorization.startswith("Bearer ")
            else None
        )
        if self.idp.reject_at_litellm or token not in self.idp.issued_access_tokens:
            return self._send(401, {"error": "unauthorized"})
        return self._send(200, {"data": []})


@pytest.fixture
def idp(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))

    provider = MockIdentityProvider()
    server = MockServer(("127.0.0.1", 0), Handler)
    server.idp = provider
    provider.base_url = f"http://127.0.0.1:{server.server_address[1]}"
    provider.native_oidc = {
        "issuer": provider.base_url,
        "client_id": CLIENT_ID,
        "scopes": SCOPES,
    }

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield provider
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)


def fake_browser(monkeypatch, *, follow=True):
    """Replace `webbrowser.open` with a client that follows the redirect."""
    visited = []

    def open_url(url):
        def visit():
            parsed = urlsplit(url)
            connection = HTTPConnection(parsed.hostname, parsed.port, timeout=5)
            connection.request("GET", f"{parsed.path}?{parsed.query}")
            response = connection.getresponse()
            response.read()
            visited.append(response.status)
            location = response.getheader("Location")
            connection.close()
            if not follow or not location:
                return
            redirect = urlsplit(location)
            callback = HTTPConnection(redirect.hostname, redirect.port, timeout=5)
            callback.request("GET", f"{redirect.path}?{redirect.query}")
            callback.getresponse().read()
            callback.close()

        threading.Thread(target=visit, daemon=True).start()
        return True

    monkeypatch.setattr(browser_flow.webbrowser, "open", open_url)
    return visited


class TestBrowserLoginEndToEnd:
    def test_full_login_stores_a_usable_credential(self, idp, monkeypatch):
        visited = fake_browser(monkeypatch)
        lines = []

        credential = run_native_login(
            idp.base_url, flow=FLOW_BROWSER, echo=lines.append
        )

        assert visited == [302]
        assert credential["auth_type"] == AUTH_TYPE_NATIVE_OIDC
        assert credential["schema_version"] == TOKEN_SCHEMA_VERSION
        assert credential["issuer"] == idp.base_url
        assert credential["client_id"] == CLIENT_ID
        assert credential["scopes"] == SCOPES
        assert credential["key"] in idp.issued_access_tokens
        assert credential["refresh_token"] in idp.refresh_tokens
        assert is_native_credential(credential)
        assert not needs_refresh(credential)

    def test_every_expected_endpoint_was_actually_called(self, idp, monkeypatch):
        fake_browser(monkeypatch)
        run_native_login(idp.base_url, flow=FLOW_BROWSER, echo=lambda _: None)

        assert ("GET", "/.well-known/litellm-ui-config") in idp.requests
        assert ("GET", "/.well-known/openid-configuration") in idp.requests
        assert ("GET", "/authorize") in idp.requests
        assert ("POST", "/token") in idp.requests
        # The proxy is asked whether it accepts the token before anything is stored.
        assert ("GET", "/v1/models") in idp.requests

    def test_credential_file_is_owner_only_and_holds_no_id_token(
        self, idp, monkeypatch, tmp_path
    ):
        fake_browser(monkeypatch)
        run_native_login(idp.base_url, flow=FLOW_BROWSER, echo=lambda _: None)

        token_file = tmp_path / ".litellm" / "token.json"
        assert token_file.exists()
        assert oct(token_file.stat().st_mode & 0o777) == "0o600"
        stored = json.loads(token_file.read_text())
        assert "id_token" not in stored
        assert stored["key"] in idp.issued_access_tokens

    def test_pkce_verifier_is_checked_by_the_provider(self, idp, monkeypatch):
        # If the CLI ever sent the challenge instead of the verifier, or reused a
        # verifier from another exchange, the mock provider rejects the grant.
        fake_browser(monkeypatch)
        credential = run_native_login(
            idp.base_url, flow=FLOW_BROWSER, echo=lambda _: None
        )
        assert credential["key"] in idp.issued_access_tokens
        assert idp.authorization_codes == {}  # the code was consumed exactly once

    def test_login_fails_when_litellm_rejects_the_token(self, idp, monkeypatch):
        idp.reject_at_litellm = True
        fake_browser(monkeypatch)

        with pytest.raises(
            NativeOIDCAuthRejected, match="rejected the identity provider"
        ):
            run_native_login(idp.base_url, flow=FLOW_BROWSER, echo=lambda _: None)

    def test_nothing_is_written_when_verification_fails(
        self, idp, monkeypatch, tmp_path
    ):
        idp.reject_at_litellm = True
        fake_browser(monkeypatch)

        with pytest.raises(NativeOIDCAuthRejected):
            run_native_login(idp.base_url, flow=FLOW_BROWSER, echo=lambda _: None)

        assert not (tmp_path / ".litellm" / "token.json").exists()

    def test_abandoned_browser_login_times_out(self, idp, monkeypatch):
        # The user opens the page and never finishes: the redirect never arrives.
        fake_browser(monkeypatch, follow=False)
        monkeypatch.setattr(
            "litellm.proxy.client.cli.native_oidc.login.run_browser_flow",
            functools.partial(browser_flow.run_browser_flow, timeout=0.5),
        )

        with pytest.raises(NativeOIDCError, match="timed out"):
            run_native_login(idp.base_url, flow=FLOW_BROWSER, echo=lambda _: None)


class TestDeviceLoginEndToEnd:
    def test_full_device_login(self, idp, monkeypatch):
        idp.device_polls_until_approval = 3
        monkeypatch.setattr(
            "litellm.proxy.client.cli.native_oidc.device_flow.time.sleep",
            lambda _: None,
        )
        lines = []

        credential = run_native_login(
            idp.base_url, flow=FLOW_DEVICE, open_browser=False, echo=lines.append
        )

        assert idp.device_polls == 3
        assert credential["key"] in idp.issued_access_tokens
        assert ("POST", "/device") in idp.requests
        output = "\n".join(lines)
        assert "WXYZ-1234" in output
        assert f"{idp.base_url}/activate" in output


class TestRefreshEndToEnd:
    def login(self, idp, monkeypatch):
        fake_browser(monkeypatch)
        return run_native_login(idp.base_url, flow=FLOW_BROWSER, echo=lambda _: None)

    def login_then_expire(self, idp, monkeypatch):
        """Log in, then age out the credential on disk as well as in hand.

        The stored copy has to be stale too, otherwise the post-lock re-check
        legitimately short-circuits and no refresh is attempted at all.
        """
        credential = self.login(idp, monkeypatch)
        stale = {**credential, "expires_at": time.time() - 1}
        save_credential(stale)
        return stale

    def test_refresh_rotates_the_credential(self, idp, monkeypatch):
        original = self.login_then_expire(idp, monkeypatch)

        refreshed = refresh_native_credential(original)

        assert refreshed["key"] != original["key"]
        assert refreshed["key"] in idp.issued_access_tokens
        assert refreshed["refresh_token"] != original["refresh_token"]
        assert refreshed["issuer"] == original["issuer"]
        assert not needs_refresh(refreshed)

    def test_refresh_retains_the_previous_token_when_none_is_reissued(
        self, idp, monkeypatch
    ):
        original = self.login_then_expire(idp, monkeypatch)
        idp.omit_refresh_token = True

        refreshed = refresh_native_credential(original)

        assert refreshed["key"] != original["key"]
        assert refreshed["refresh_token"] == original["refresh_token"]

    def test_refresh_revalidates_the_whole_trust_chain(self, idp, monkeypatch):
        original = self.login_then_expire(idp, monkeypatch)
        # The proxy now points at a different client; the stored credential must
        # not be silently refreshed against it.
        idp.native_oidc = {**idp.native_oidc, "client_id": "some-other-client"}

        with pytest.raises(NativeOIDCError, match="different OIDC issuer or client id"):
            refresh_native_credential(original)

    def test_refresh_fails_when_litellm_stops_accepting_the_token(
        self, idp, monkeypatch
    ):
        original = self.login_then_expire(idp, monkeypatch)
        idp.reject_at_litellm = True

        with pytest.raises(NativeOIDCAuthRejected):
            refresh_native_credential(original)

    def test_a_still_fresh_credential_on_disk_short_circuits_the_refresh(
        self, idp, monkeypatch
    ):
        original = self.login(idp, monkeypatch)
        polls_before = len(idp.requests)

        # The on-disk credential is fresh, so a caller racing in with a stale copy
        # reuses it instead of burning the refresh token.
        result = refresh_native_credential({**original, "expires_at": time.time() - 1})

        assert result["key"] == original["key"]
        assert len(idp.requests) == polls_before

    def test_refresh_without_a_refresh_token_is_rejected(self, idp, monkeypatch):
        original = self.login_then_expire(idp, monkeypatch)
        stripped = {k: v for k, v in original.items() if k != "refresh_token"}

        with pytest.raises(NativeOIDCError, match="no refresh token"):
            refresh_native_credential(stripped)
