"""Tests for the ``lite login --pkce`` browser flow against a fake proxy and a real loopback listener."""

import hashlib
import json
import re
import socket
import threading
import time
import urllib.error
import urllib.request
from base64 import urlsafe_b64encode
from urllib.parse import parse_qs, urlparse

import pytest
import requests

from litellm.litellm_core_utils.cli_token_utils import is_cli_token_fresh
from litellm.proxy.client.cli.commands.pkce_login import (
    CallbackCode,
    CallbackDenied,
    CliAuthContract,
    LoopbackServer,
    PkceCredential,
    PkceFailure,
    RevocationUnavailable,
    _error_detail,
    authorize_url,
    discover_cli_auth,
    fresh_api_key,
    pkce_pair,
    pkce_token_record,
    redeem_code,
    refresh_credential,
    register_client,
    revoke_credential,
    revoke_stored_credential,
    run_pkce_login,
)

BASE = "https://llm.example.com"
DISCOVERY_URL = f"{BASE}/.well-known/litellm-cli-auth"
CONTRACT_DOC = {
    "contract_version": 1,
    "issuer": BASE,
    "authorization_endpoint": f"{BASE}/authorize",
    "token_endpoint": f"{BASE}/token",
    "registration_endpoint": f"{BASE}/register",
    "revocation_endpoint": f"{BASE}/revoke",
    "resource": BASE,
    "response_types_supported": ["code"],
    "grant_types_supported": ["authorization_code", "refresh_token"],
    "code_challenge_methods_supported": ["S256"],
    "token_endpoint_auth_methods_supported": ["none"],
    "revocation_endpoint_auth_methods_supported": ["none"],
}
CONTRACT = CliAuthContract.model_validate(CONTRACT_DOC)
TOKEN_JSON = {
    "access_token": "sk-cli-new",
    "token_type": "Bearer",
    "expires_in": 3600,
    "refresh_token": "llm_srefresh_new",
    "user_id": "u1",
    "team_id": "team-b",
}
STORED = {
    "base_url": BASE,
    "key": "sk-cli-old",
    "expires_at": 1_000_000.0,
    "refresh_token": "llm_srefresh_old",
    "client_id": "llm_dcrc_abc",
    "token_endpoint": f"{BASE}/token",
    "revocation_endpoint": f"{BASE}/revoke",
    "resource": BASE,
    "user_id": "user-1",
    "team_id": "team-a",
}


class _FakeResponse:
    def __init__(self, status_code, payload=None, text="", headers=None):
        self.status_code = status_code
        self.headers = headers or {}
        self._payload = payload
        self.text = json.dumps(payload) if payload is not None else text
        self.content = self.text.encode()

    def json(self):
        if self._payload is None:
            raise ValueError("not json")
        return self._payload


def _wire(body):
    return None if body is None else json.loads(json.dumps(body))


class _FakeHttp:
    def __init__(self, routes=None):
        self.routes = routes or {}
        self.calls = []

    def get(self, url, *, timeout):
        self.calls.append(("GET", url, None, None))
        return self._route("GET", url, None, None)

    def post(self, url, *, data=None, json=None, timeout, allow_redirects):
        self.calls.append(("POST", url, data, _wire(json)))
        response = self._route("POST", url, data, json)
        if allow_redirects and 300 <= response.status_code < 400:
            return self.post(response.headers["Location"], data=data, json=json, timeout=timeout, allow_redirects=True)
        return response

    def _route(self, method, url, data, json):
        handler = self.routes[(method, url)]
        if isinstance(handler, Exception):
            raise handler
        return handler(data, json) if callable(handler) else handler


def _get(url):
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            return response.status, response.read().decode()
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode()


def _fresh(token_data, save, http, reload=lambda: None, **options):
    return fresh_api_key(token_data, save, http, reload=reload, **options)


def _challenge_for(verifier: str) -> str:
    return urlsafe_b64encode(hashlib.sha256(verifier.encode("ascii")).digest()).rstrip(b"=").decode("ascii")


def test_loopback_settles_only_on_a_code_for_the_expected_state():
    with LoopbackServer("expected-state") as server:
        result = {}
        thread = threading.Thread(target=lambda: result.setdefault("outcome", server.wait(10)))
        thread.start()
        base = f"http://127.0.0.1:{server.server_address[1]}"
        assert server.redirect_uri == f"{base}/callback"
        assert _get(f"{base}/nope")[0] == 404
        status, body = _get(f"{base}/callback?state=other&code=stolen")
        assert status == 400
        assert "still waiting" in body
        status, body = _get(f"{base}/callback?state=expected-state")
        assert status == 400
        assert "no authorization code" in body
        assert thread.is_alive()
        status, body = _get(f"{base}/callback?state=expected-state&code=the-code")
        assert status == 200
        assert "Signed in to LiteLLM" in body
        thread.join(5)
    assert not thread.is_alive()
    assert result["outcome"] == CallbackCode(code="the-code")


def test_loopback_reports_a_denied_sign_in():
    with LoopbackServer("expected-state") as server:
        result = {}
        thread = threading.Thread(target=lambda: result.setdefault("outcome", server.wait(10)))
        thread.start()
        base = f"http://127.0.0.1:{server.server_address[1]}"
        status, body = _get(f"{base}/callback?state=expected-state&error=access_denied&error_description=nope")
        assert status == 200
        assert "not approved" in body
        thread.join(5)
    assert result["outcome"] == CallbackDenied(error="access_denied", description="nope")


def test_loopback_drops_an_idle_connection_instead_of_waiting_on_it():
    with LoopbackServer("expected-state", connection_timeout_seconds=0.2) as server:
        result = {}
        thread = threading.Thread(target=lambda: result.setdefault("outcome", server.wait(10)))
        thread.start()
        idle = socket.create_connection(server.server_address)
        try:
            status, body = _get(f"http://127.0.0.1:{server.server_address[1]}/callback?state=expected-state&code=c1")
        finally:
            idle.close()
        assert status == 200
        assert "Signed in to LiteLLM" in body
        thread.join(5)
    assert not thread.is_alive()
    assert result["outcome"] == CallbackCode(code="c1")


def test_loopback_wait_times_out_on_its_clock():
    ticks = iter([0.0, 0.5, 1.5])
    with LoopbackServer("expected-state") as server:
        server.timeout = 0.01
        outcome = server.wait(1.0, clock=lambda: next(ticks))
    assert outcome == PkceFailure("timed out waiting for the browser sign-in to finish")


def test_discover_reads_the_contract_from_the_well_known_path():
    http = _FakeHttp({("GET", DISCOVERY_URL): _FakeResponse(200, CONTRACT_DOC)})
    assert discover_cli_auth(f"{BASE}/", http) == CONTRACT


@pytest.mark.parametrize(
    "response, expected",
    [
        (_FakeResponse(404, text="not found"), "does not support `lite login --pkce`"),
        (_FakeResponse(200, {**CONTRACT_DOC, "contract_version": 2}), "unsupported discovery document"),
        (_FakeResponse(200, {"issuer": BASE}), "unsupported discovery document"),
        (_FakeResponse(200, text="<html>"), "unsupported discovery document"),
        (_FakeResponse(200, {**CONTRACT_DOC, "code_challenge_methods_supported": ["plain"]}), "PKCE S256"),
        (requests.ConnectionError("refused"), "could not reach"),
        (_FakeResponse(200, {**CONTRACT_DOC, "issuer": "https://evil.example.com"}), "is issued for https://evil"),
        (_FakeResponse(200, {**CONTRACT_DOC, "issuer": f"{BASE}/other"}), "is issued for"),
        (_FakeResponse(200, {**CONTRACT_DOC, "issuer": "http://llm.example.com"}), "is issued for"),
        (_FakeResponse(200, {**CONTRACT_DOC, "token_endpoint": "https://evil.example.com/token"}), "outside"),
        (_FakeResponse(200, {**CONTRACT_DOC, "registration_endpoint": f"{BASE}:8443/register"}), "outside"),
        (_FakeResponse(200, {**CONTRACT_DOC, "revocation_endpoint": "https://evil.example.com/revoke"}), "outside"),
        (_FakeResponse(200, {**CONTRACT_DOC, "authorization_endpoint": "http://llm.example.com/authorize"}), "outside"),
        (_FakeResponse(200, {**CONTRACT_DOC, "resource": "https://evil.example.com"}), "outside"),
        (_FakeResponse(200, {**CONTRACT_DOC, "token_endpoint": "/token"}), "outside"),
    ],
)
def test_discover_failures_name_the_cause(response, expected):
    result = discover_cli_auth(BASE, _FakeHttp({("GET", DISCOVERY_URL): response}))
    assert isinstance(result, PkceFailure)
    assert expected in result.reason


def test_discover_refuses_endpoints_on_another_origin_and_names_each_of_them():
    doc = {**CONTRACT_DOC, "token_endpoint": "https://evil.example.com/token", "resource": "https://evil.example.com"}
    result = discover_cli_auth(BASE, _FakeHttp({("GET", DISCOVERY_URL): _FakeResponse(200, doc)}))
    assert isinstance(result, PkceFailure)
    assert "https://evil.example.com/token" in result.reason
    assert "https://evil.example.com" in result.reason
    assert "refusing to send credentials" in result.reason


@pytest.mark.parametrize(
    "typed_base_url",
    ["https://LLM.example.com:443/", "https://llm.example.com", "HTTPS://llm.example.com/"],
)
def test_discover_accepts_the_same_proxy_however_the_user_spelled_it(typed_base_url):
    discovery_url = f"{typed_base_url.rstrip('/')}/.well-known/litellm-cli-auth"
    http = _FakeHttp({("GET", discovery_url): _FakeResponse(200, CONTRACT_DOC)})
    assert discover_cli_auth(typed_base_url, http) == CONTRACT


def test_discover_accepts_a_proxy_mounted_under_a_path_only_at_that_path():
    mounted = f"{BASE}/litellm"
    doc = {
        **CONTRACT_DOC,
        "issuer": mounted,
        "authorization_endpoint": f"{mounted}/authorize",
        "token_endpoint": f"{mounted}/token",
        "registration_endpoint": f"{mounted}/register",
        "revocation_endpoint": f"{mounted}/revoke",
        "resource": mounted,
    }
    routes = {
        ("GET", f"{mounted}/.well-known/litellm-cli-auth"): _FakeResponse(200, doc),
        ("GET", DISCOVERY_URL): _FakeResponse(200, doc),
    }
    assert discover_cli_auth(f"{mounted}/", _FakeHttp(routes)) == CliAuthContract.model_validate(doc)
    rejected = discover_cli_auth(BASE, _FakeHttp(routes))
    assert isinstance(rejected, PkceFailure)
    assert f"is issued for {mounted}" in rejected.reason


def test_register_sends_a_public_loopback_client_and_returns_its_id():
    http = _FakeHttp({("POST", f"{BASE}/register"): _FakeResponse(201, {"client_id": "llm_dcrc_abc"})})
    assert register_client(CONTRACT, "http://127.0.0.1:5/callback", http) == "llm_dcrc_abc"
    assert http.calls == [
        (
            "POST",
            f"{BASE}/register",
            None,
            {
                "client_name": "litellm-cli",
                "redirect_uris": ["http://127.0.0.1:5/callback"],
                "grant_types": ["authorization_code", "refresh_token"],
                "response_types": ["code"],
                "token_endpoint_auth_method": "none",
            },
        )
    ]


@pytest.mark.parametrize(
    "response, expected",
    [
        (
            _FakeResponse(400, {"error": "invalid_redirect_uri", "error_description": "loopback only"}),
            "400: loopback only",
        ),
        (_FakeResponse(201, {}), "unexpected body"),
        (requests.ConnectionError("refused"), "registration failed"),
    ],
)
def test_register_failures_name_the_cause(response, expected):
    result = register_client(
        CONTRACT, "http://127.0.0.1:5/callback", _FakeHttp({("POST", f"{BASE}/register"): response})
    )
    assert isinstance(result, PkceFailure)
    assert expected in result.reason


def test_pkce_pair_is_a_high_entropy_s256_pair():
    verifier, challenge = pkce_pair()
    assert 43 <= len(verifier) <= 128
    assert challenge == _challenge_for(verifier)
    assert pkce_pair()[0] != verifier


def test_authorize_url_carries_every_oauth_parameter_and_the_proxy_resource():
    url = authorize_url(CONTRACT, "llm_dcrc_abc", "http://127.0.0.1:5/callback", "state-1", "challenge-1")
    parsed = urlparse(url)
    assert f"{parsed.scheme}://{parsed.netloc}{parsed.path}" == f"{BASE}/authorize"
    assert parse_qs(parsed.query) == {
        "response_type": ["code"],
        "client_id": ["llm_dcrc_abc"],
        "redirect_uri": ["http://127.0.0.1:5/callback"],
        "state": ["state-1"],
        "code_challenge": ["challenge-1"],
        "code_challenge_method": ["S256"],
        "resource": [BASE],
    }


def test_redeem_code_posts_the_verifier_and_binds_the_credential_to_the_contract():
    http = _FakeHttp({("POST", f"{BASE}/token"): _FakeResponse(200, TOKEN_JSON)})
    credential = redeem_code(
        CONTRACT, "llm_dcrc_abc", "http://127.0.0.1:5/callback", "the-code", "the-verifier", http, now=lambda: 100.0
    )
    assert credential == PkceCredential(
        access_token="sk-cli-new",
        refresh_token="llm_srefresh_new",
        expires_at=3700.0,
        client_id="llm_dcrc_abc",
        token_endpoint=f"{BASE}/token",
        revocation_endpoint=f"{BASE}/revoke",
        resource=BASE,
        user_id="u1",
        team_id="team-b",
    )
    assert http.calls[0][2] == {
        "grant_type": "authorization_code",
        "code": "the-code",
        "redirect_uri": "http://127.0.0.1:5/callback",
        "client_id": "llm_dcrc_abc",
        "code_verifier": "the-verifier",
        "resource": BASE,
    }


def test_refresh_posts_the_refresh_grant_for_the_same_client_and_resource():
    http = _FakeHttp({("POST", f"{BASE}/token"): _FakeResponse(200, TOKEN_JSON)})
    credential = refresh_credential(
        f"{BASE}/token", f"{BASE}/revoke", BASE, "llm_dcrc_abc", "llm_srefresh_old", http, now=lambda: 5.0
    )
    assert isinstance(credential, PkceCredential)
    assert credential.expires_at == 3605.0
    assert credential.refresh_token == "llm_srefresh_new"
    assert http.calls[0][2] == {
        "grant_type": "refresh_token",
        "refresh_token": "llm_srefresh_old",
        "client_id": "llm_dcrc_abc",
        "resource": BASE,
    }


@pytest.mark.parametrize(
    "response, expected",
    [
        (_FakeResponse(400, {"error": "invalid_grant", "error_description": "already used"}), "400: already used"),
        (_FakeResponse(200, {**TOKEN_JSON, "refresh_token": ""}), "unexpected body"),
        (_FakeResponse(200, {**TOKEN_JSON, "expires_in": 0}), "unexpected body"),
        (_FakeResponse(200, text="<html>"), "unexpected body"),
        (requests.ConnectionError("refused"), "token request failed"),
    ],
)
def test_token_request_failures_name_the_cause(response, expected):
    result = redeem_code(CONTRACT, "c", "r", "code", "v", _FakeHttp({("POST", f"{BASE}/token"): response}))
    assert isinstance(result, PkceFailure)
    assert expected in result.reason


def test_revoke_posts_an_rfc7009_request_for_the_refresh_token():
    http = _FakeHttp({("POST", f"{BASE}/revoke"): _FakeResponse(200, {})})
    assert revoke_credential(f"{BASE}/revoke", "llm_dcrc_abc", "llm_srefresh_old", http) is None
    assert http.calls == [
        (
            "POST",
            f"{BASE}/revoke",
            {"token": "llm_srefresh_old", "token_type_hint": "refresh_token", "client_id": "llm_dcrc_abc"},
            None,
        )
    ]


@pytest.mark.parametrize(
    "response, expected",
    [
        (_FakeResponse(401, {"error": "invalid_client"}), "401: invalid_client"),
        (requests.ConnectionError("refused"), "revocation request failed"),
    ],
)
def test_revoke_failures_name_the_cause(response, expected):
    result = revoke_credential(f"{BASE}/revoke", "llm_dcrc_abc", "t", _FakeHttp({("POST", f"{BASE}/revoke"): response}))
    assert isinstance(result, PkceFailure)
    assert expected in result.reason


def test_revoke_reports_a_503_as_unavailable_so_the_caller_can_retry():
    response = _FakeResponse(
        503,
        {
            "error": "temporarily_unavailable",
            "error_description": "the single-use record is unavailable right now; try again shortly",
        },
    )
    result = revoke_credential(f"{BASE}/revoke", "llm_dcrc_abc", "t", _FakeHttp({("POST", f"{BASE}/revoke"): response}))
    assert result == RevocationUnavailable(
        "revocation failed with 503: the single-use record is unavailable right now; try again shortly"
    )


@pytest.mark.parametrize("status", [302, 307, 308])
def test_posts_to_the_proxy_never_follow_a_redirect(status):
    elsewhere = "https://evil.example.net/oauth"
    http = _FakeHttp(
        {
            ("POST", f"{BASE}/register"): _FakeResponse(status, headers={"Location": elsewhere}),
            ("POST", f"{BASE}/token"): _FakeResponse(status, headers={"Location": elsewhere}),
            ("POST", f"{BASE}/revoke"): _FakeResponse(status, headers={"Location": elsewhere}),
            ("POST", elsewhere): _FakeResponse(200, {"client_id": "llm_dcrc_evil", **TOKEN_JSON}),
        }
    )
    outcomes = (
        register_client(CONTRACT, "http://127.0.0.1:5/callback", http),
        redeem_code(CONTRACT, "llm_dcrc_abc", "http://127.0.0.1:5/callback", "code", "verifier", http),
        refresh_credential(f"{BASE}/token", f"{BASE}/revoke", BASE, "llm_dcrc_abc", "llm_srefresh_old", http),
        revoke_credential(f"{BASE}/revoke", "llm_dcrc_abc", "llm_srefresh_old", http),
    )
    assert [type(outcome) for outcome in outcomes] == [PkceFailure] * 4
    assert all(f"redirected to {elsewhere}; refusing to follow it" in outcome.reason for outcome in outcomes)
    posted_to = [url for _, url, _, _ in http.calls]
    assert posted_to == [f"{BASE}/register", f"{BASE}/token", f"{BASE}/token", f"{BASE}/revoke"]


@pytest.mark.parametrize(
    "response, expected",
    [
        (_FakeResponse(400, {"error": "invalid_grant", "error_description": "the code expired"}), "the code expired"),
        (_FakeResponse(400, {"error": "invalid_grant"}), "invalid_grant"),
        (_FakeResponse(404, {"detail": "Not Found"}), "Not Found"),
        (_FakeResponse(502, text="<html>bad gateway</html>"), "<html>bad gateway</html>"),
        (_FakeResponse(500, text="x" * 500), "x" * 200),
    ],
)
def test_error_detail_prefers_the_oauth_description(response, expected):
    assert _error_detail(response) == expected


def _blocking_browser(seen, suffix):
    seen["done"] = threading.Event()

    def open_browser(url):
        query = parse_qs(urlparse(url).query)
        seen["authorize"] = query
        seen["callback"] = _get(f"{query['redirect_uri'][0]}?state={query['state'][0]}&{suffix}")
        seen["done"].set()

    return open_browser


def test_run_pkce_login_end_to_end_against_a_fake_proxy():
    seen = {}

    def register(data, json):
        seen["register"] = json
        return _FakeResponse(201, {"client_id": "llm_dcrc_abc"})

    def token(data, json):
        seen["token"] = data
        return _FakeResponse(200, TOKEN_JSON)

    http = _FakeHttp(
        {
            ("GET", DISCOVERY_URL): _FakeResponse(200, CONTRACT_DOC),
            ("POST", f"{BASE}/register"): register,
            ("POST", f"{BASE}/token"): token,
        }
    )
    echoed = []
    credential = run_pkce_login(
        BASE, http, open_browser=_blocking_browser(seen, "code=the-code"), echo=echoed.append, timeout_seconds=10
    )
    assert isinstance(credential, PkceCredential)
    assert credential.access_token == "sk-cli-new"
    assert credential.refresh_token == "llm_srefresh_new"
    assert credential.client_id == "llm_dcrc_abc"
    assert credential.team_id == "team-b"
    redirect_uri = seen["register"]["redirect_uris"][0]
    assert re.fullmatch(r"http://127\.0\.0\.1:\d+/callback", redirect_uri)
    assert seen["authorize"]["redirect_uri"] == [redirect_uri]
    assert seen["authorize"]["client_id"] == ["llm_dcrc_abc"]
    assert seen["authorize"]["code_challenge_method"] == ["S256"]
    assert seen["authorize"]["resource"] == [BASE]
    form = seen["token"]
    assert form["code"] == "the-code"
    assert form["redirect_uri"] == redirect_uri
    assert form["client_id"] == "llm_dcrc_abc"
    assert _challenge_for(form["code_verifier"]) == seen["authorize"]["code_challenge"][0]
    assert echoed[0].startswith(f"Opening browser to: {BASE}/authorize?")
    assert echoed[1] == "Approve the sign-in in your browser. Waiting..."
    assert seen["done"].wait(5)
    assert seen["callback"][0] == 200


def test_run_pkce_login_reports_a_denied_sign_in_without_touching_the_token_endpoint():
    seen = {}
    http = _FakeHttp(
        {
            ("GET", DISCOVERY_URL): _FakeResponse(200, CONTRACT_DOC),
            ("POST", f"{BASE}/register"): _FakeResponse(201, {"client_id": "llm_dcrc_abc"}),
        }
    )
    result = run_pkce_login(
        BASE,
        http,
        open_browser=_blocking_browser(seen, "error=access_denied&error_description=nope"),
        echo=lambda _: None,
        timeout_seconds=10,
    )
    assert result == PkceFailure("sign-in was not approved (access_denied): nope")
    assert [call[1] for call in http.calls] == [DISCOVERY_URL, f"{BASE}/register"]


def test_run_pkce_login_stops_before_the_browser_when_registration_fails():
    opened = []
    http = _FakeHttp(
        {
            ("GET", DISCOVERY_URL): _FakeResponse(200, CONTRACT_DOC),
            ("POST", f"{BASE}/register"): _FakeResponse(400, {"error": "invalid_client_metadata"}),
        }
    )
    result = run_pkce_login(BASE, http, open_browser=opened.append, echo=lambda _: None, timeout_seconds=1)
    assert isinstance(result, PkceFailure)
    assert "invalid_client_metadata" in result.reason
    assert opened == []


def test_pkce_token_record_keeps_every_refresh_input_next_to_the_key():
    credential = PkceCredential(
        access_token="sk-cli-new",
        refresh_token="llm_srefresh_new",
        expires_at=3700.0,
        client_id="llm_dcrc_abc",
        token_endpoint=f"{BASE}/token",
        revocation_endpoint=f"{BASE}/revoke",
        resource=BASE,
        user_id=None,
        team_id=None,
    )
    record = pkce_token_record(f"{BASE}/", credential)
    assert record["base_url"] == BASE
    assert record["key"] == "sk-cli-new"
    assert record["user_id"] == "cli-user"
    assert record["expires_at"] == 3700.0
    assert record["refresh_token"] == "llm_srefresh_new"
    assert record["client_id"] == "llm_dcrc_abc"
    assert record["token_endpoint"] == f"{BASE}/token"
    assert record["revocation_endpoint"] == f"{BASE}/revoke"
    assert record["resource"] == BASE
    assert record["team_id"] is None
    assert record["auth_header_name"] == "Authorization"


def test_fresh_api_key_returns_a_classic_or_still_fresh_key_without_network():
    http = _FakeHttp()
    saved = []
    assert _fresh({"key": "sk-classic"}, saved.append, http) == "sk-classic"
    assert _fresh(STORED, saved.append, http, now=lambda: 999_000.0) == "sk-cli-old"
    assert _fresh({}, saved.append, http) is None
    assert _fresh({"key": ""}, saved.append, http) is None
    assert http.calls == []
    assert saved == []


def test_fresh_api_key_refreshes_near_expiry_and_saves_before_returning():
    http = _FakeHttp({("POST", f"{BASE}/token"): _FakeResponse(200, TOKEN_JSON)})
    saved = []
    assert _fresh(STORED, saved.append, http, now=lambda: 999_950.0) == "sk-cli-new"
    assert len(saved) == 1
    assert saved[0]["key"] == "sk-cli-new"
    assert saved[0]["refresh_token"] == "llm_srefresh_new"
    assert saved[0]["expires_at"] == 999_950.0 + 3600
    assert saved[0]["base_url"] == BASE
    assert saved[0]["team_id"] == "team-b"
    assert http.calls[0][2]["refresh_token"] == "llm_srefresh_old"


@pytest.mark.parametrize(("seconds_left", "refreshes"), [(200, True), (600, False)])
def test_fresh_api_key_refreshes_exactly_when_is_cli_token_fresh_stops_calling_the_key_fresh(
    seconds_left, refreshes
):
    """`lite up` and `lite auth print-token` ask `is_cli_token_fresh` and then `fresh_api_key`; a key
    the first calls stale that the second would not renew left `lite up` refusing a renewable login."""
    record = {**STORED, "expires_at": time.time() + seconds_left}
    http = _FakeHttp({("POST", f"{BASE}/token"): _FakeResponse(200, TOKEN_JSON)})
    saved = []

    key = _fresh(record, saved.append, http)

    assert is_cli_token_fresh(record) is (not refreshes)
    assert key == ("sk-cli-new" if refreshes else "sk-cli-old")
    assert len(saved) == (1 if refreshes else 0)


def test_fresh_api_key_never_hands_out_a_rotated_key_it_could_not_save():
    http = _FakeHttp({("POST", f"{BASE}/token"): _FakeResponse(200, TOKEN_JSON)})

    def save(_record):
        raise OSError("disk full")

    with pytest.raises(OSError, match="disk full"):
        _fresh(STORED, save, http, now=lambda: 999_950.0)


def test_fresh_api_key_falls_back_to_the_old_key_only_while_it_is_still_valid():
    failing = _FakeHttp({("POST", f"{BASE}/token"): _FakeResponse(503, {"error": "temporarily_unavailable"})})
    saved = []
    assert _fresh(STORED, saved.append, failing, now=lambda: 999_950.0) == "sk-cli-old"
    assert _fresh(STORED, saved.append, failing, now=lambda: 1_000_001.0) is None
    assert saved == []


def test_fresh_api_key_reports_why_a_renewal_failed_unless_a_sibling_rotated_first():
    failing = _FakeHttp({("POST", f"{BASE}/token"): _FakeResponse(503, {"error": "temporarily_unavailable"})})
    rotated = {**STORED, "key": "sk-cli-sibling", "refresh_token": "llm_srefresh_sibling", "expires_at": 1_003_600.0}
    warnings = []
    assert _fresh(STORED, lambda _: None, failing, now=lambda: 990_000.0, warn=warnings.append) == "sk-cli-old"
    assert warnings == []
    assert _fresh(STORED, lambda _: None, failing, now=lambda: 999_950.0, warn=warnings.append) == "sk-cli-old"
    assert _fresh(STORED, lambda _: None, failing, now=lambda: 1_000_001.0, warn=warnings.append) is None
    assert warnings == ["Could not renew the key: token request failed with 503: temporarily_unavailable"] * 2
    sibling = _fresh(STORED, lambda _: None, failing, reload=lambda: rotated, now=lambda: 1_000_001.0, warn=warnings.append)
    assert sibling == "sk-cli-sibling"
    assert len(warnings) == 2


def test_fresh_api_key_uses_a_sibling_rotation_when_its_own_refresh_loses_the_race():
    rotated = {**STORED, "key": "sk-cli-sibling", "refresh_token": "llm_srefresh_sibling", "expires_at": 1_003_600.0}
    http = _FakeHttp({("POST", f"{BASE}/token"): _FakeResponse(400, {"error": "invalid_grant"})})
    saved = []
    assert _fresh(STORED, saved.append, http, reload=lambda: rotated, now=lambda: 999_950.0) == "sk-cli-sibling"
    assert _fresh(STORED, saved.append, http, reload=lambda: rotated, now=lambda: 1_000_001.0) == "sk-cli-sibling"
    assert _fresh(STORED, saved.append, http, reload=lambda: STORED, now=lambda: 1_000_001.0) is None
    assert _fresh(STORED, saved.append, http, reload=lambda: None, now=lambda: 1_000_001.0) is None
    assert saved == []


@pytest.mark.parametrize(
    "sibling_record",
    [
        {"base_url": "https://other.example.com", "token_endpoint": "https://other.example.com/token"},
        {"base_url": "https://other.example.com", "resource": "https://other.example.com"},
        {"base_url": "https://other.example.com"},
        {"token_endpoint": "https://other.example.com/token"},
        {"resource": "https://other.example.com"},
        {"user_id": "someone-else"},
        {"team_id": "another-team"},
        {"team_id": None},
        {"expires_at": 999_000.0},
        {"expires_at": None},
        {"key": ""},
    ],
    ids=[
        "other-proxy",
        "other-resource-line",
        "other-base-url",
        "other-token-endpoint",
        "other-resource",
        "other-user",
        "other-team",
        "no-team",
        "expired",
        "no-expiry",
        "empty-key",
    ],
)
def test_fresh_api_key_never_substitutes_a_sibling_record_for_another_proxy_identity_or_a_dead_key(sibling_record):
    rotated = {**STORED, "key": "sk-cli-sibling", "refresh_token": "llm_srefresh_sibling", "expires_at": 1_003_600.0}
    foreign = {**rotated, **sibling_record}
    http = _FakeHttp({("POST", f"{BASE}/token"): _FakeResponse(400, {"error": "invalid_grant"})})
    assert _fresh(STORED, lambda _: None, http, reload=lambda: foreign, now=lambda: 1_000_001.0) is None
    assert _fresh(STORED, lambda _: None, http, reload=lambda: foreign, now=lambda: 999_950.0) == "sk-cli-old"


def test_fresh_api_key_without_refresh_inputs_expires_like_a_classic_key():
    http = _FakeHttp()
    no_refresh = {key: value for key, value in STORED.items() if key != "refresh_token"}
    assert _fresh(no_refresh, lambda _: None, http, now=lambda: 999_950.0) == "sk-cli-old"
    assert _fresh(no_refresh, lambda _: None, http, now=lambda: 1_000_001.0) is None
    assert http.calls == []


def test_revoke_stored_credential_revokes_only_pkce_records():
    http = _FakeHttp({("POST", f"{BASE}/revoke"): _FakeResponse(200, {})})
    assert revoke_stored_credential({"key": "sk-classic"}, http) is None
    assert http.calls == []
    assert revoke_stored_credential(STORED, http) is None
    assert http.calls[0][2] == {
        "token": "llm_srefresh_old",
        "token_type_hint": "refresh_token",
        "client_id": "llm_dcrc_abc",
    }
