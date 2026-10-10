from __future__ import annotations

import base64
import json
import os
import re
import uuid
from collections.abc import Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from pathlib import Path
from typing import Final
from urllib.parse import parse_qs, urlencode, urlparse

import httpx
import jwt
import pytest
import yaml
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID
from onelogin.saml2.utils import OneLogin_Saml2_Utils

from tests.integration._support.client import JSON_OBJECT, Gateway, Scenario, gateway_from_environment, string_value
from tests.integration._support.database import read_rows, write_rows
from tests.integration._support.database_relay import dropped_connection_relay
from tests.integration._support.process import OwnedProxy, owned_proxy_process
from tests.integration._support.wire import Reply, Request, Wire, wire_server

pytestmark: Final = pytest.mark.timeout(300)

CLIENT_ID: Final = "integration-entra-client"
CLIENT_SECRET: Final = "integration-entra-secret"
TENANT: Final = "integration-entra-tenant"
CLI_SOURCE: Final = "litellm-cli"
POLL_SECRET_HEADER: Final = "x-litellm-cli-poll-secret"
CLI_LOGIN_PAGE_TITLE: Final = "<title>LiteLLM CLI Login</title>"
CLI_SUCCESS_PAGE_TITLE: Final = "<title>CLI Authentication Successful - LiteLLM</title>"
REFUSED: Final = "SSO login failed: this sign-in did not resolve to a user id"
LOOKUP_FAILED: Final = "Failed to retrieve user information from SSO"
BLANK: Final = r"^\s*$"
COMPLETE_TOKEN_FIELD: Final = re.compile(r'name="browser_complete_token" value="([^"]+)"')
COMPLETE_FORM_ACTION: Final = re.compile(r'action="([^"]+/sso/cli/complete/[^"]+)"')
GRAPH_ERROR: Final = json.dumps(
    {"error": {"code": "Authentication_MissingOrMalformed", "message": "Access Token missing or malformed."}}
).encode()
CUSTOM_SSO_MODULE: Final = """
async def custom_sso(result: object) -> dict[str, object] | None:
    email: str = getattr(result, "email", None) or ""
    for prefix, user_id in (("mapped-blank-", " "), ("mapped-none-", None)):
        if email.startswith(prefix):
            return {
                "models": [],
                "user_id": user_id,
                "user_email": email,
                "max_budget": None,
                "user_role": None,
                "budget_duration": None,
            }
    return None
"""


@dataclass(frozen=True, slots=True)
class Graph:
    """Entra sign-in plus the Graph `/me` lookup, one scripted reply per login hint."""

    wire: Wire
    profiles: dict[str, Reply]

    def profile(self, reply: Reply) -> str:
        hint: Final = f"hint-{uuid.uuid4().hex}"
        self.profiles[hint] = reply
        return hint


@dataclass(frozen=True, slots=True)
class CliSession:
    login_id: str
    poll_secret: str
    user_code: str


def _between(value: str, prefix: str) -> str:
    return value.removeprefix(prefix).rsplit("-", 1)[0]


def _graph_reply(profiles: Mapping[str, Reply], request: Request) -> Reply:
    target: Final = urlparse(request.target)
    if target.path == "/authorize":
        query: Final = parse_qs(target.query)
        code: Final = f"code-{query['login_hint'][0]}-{uuid.uuid4().hex}"
        state: Final = {"state": query["state"][0]} if "state" in query else {}
        location: Final = f"{query['redirect_uri'][0]}?{urlencode({'code': code, **state})}"
        return Reply(status=302, body=b"", headers={"location": location})
    if target.path == "/token":
        form: Final = parse_qs(request.body.decode())
        code_value: Final = form.get("code", [""])[0]
        if form.get("grant_type") != ["authorization_code"] or not code_value.startswith("code-"):
            return Reply(status=400, body=b'{"error": "invalid_grant"}')
        token: Final = f"graph-{_between(code_value, 'code-')}-{uuid.uuid4().hex}"
        return Reply(body=json.dumps({"access_token": token, "token_type": "Bearer", "expires_in": 3600}).encode())
    bearer: Final = request.headers.get("authorization", "")
    if not bearer.startswith("Bearer graph-"):
        return Reply(status=401, body=GRAPH_ERROR)
    hint: Final = _between(bearer, "Bearer graph-")
    if target.path == "/v1.0/me":
        return profiles[hint]
    if target.path.startswith("/v1.0/me/memberOf"):
        failed: Final = profiles[hint].status != 200
        return Reply(status=401, body=GRAPH_ERROR) if failed else Reply(body=b'{"value": []}')
    return Reply(status=404, body=b'{"error": {"code": "Request_ResourceNotFound"}}')


def _profile(user_id: str, email: str) -> Reply:
    return Reply(
        body=json.dumps(
            {
                "id": user_id,
                "userPrincipalName": email,
                "mail": email,
                "displayName": "Synthetic Person",
                "givenName": "Synthetic",
                "surname": "Person",
            }
        ).encode()
    )


def _entra_environment(graph_url: str) -> Mapping[str, str]:
    return {
        "MICROSOFT_CLIENT_ID": CLIENT_ID,
        "MICROSOFT_CLIENT_SECRET": CLIENT_SECRET,
        "MICROSOFT_TENANT": TENANT,
        "MICROSOFT_AUTHORIZATION_ENDPOINT": f"{graph_url}/authorize",
        "MICROSOFT_TOKEN_ENDPOINT": f"{graph_url}/token",
        "MICROSOFT_USERINFO_ENDPOINT": f"{graph_url}/v1.0/me",
        "MICROSOFT_GRAPH_ENDPOINT": f"{graph_url}/v1.0",
        "OAUTHLIB_INSECURE_TRANSPORT": "1",
    }


SAML_SP_ENTITY: Final = "litellm-integration-sp"
SAML_IDP_ENTITY: Final = "https://idp.integration.invalid/metadata"
SAML_STATE_COOKIE: Final = "litellm_saml_authn"


@dataclass(frozen=True, slots=True)
class SamlIdp:
    """A signing SAML IdP the test speaks for: its metadata and signed answers to a proxy login request"""

    key_pem: str
    cert_pem: str

    def metadata(self) -> str:
        body: Final = "".join(line for line in self.cert_pem.splitlines() if "CERTIFICATE" not in line)
        return (
            f'<EntityDescriptor xmlns="urn:oasis:names:tc:SAML:2.0:metadata" entityID="{SAML_IDP_ENTITY}">'
            '<IDPSSODescriptor protocolSupportEnumeration="urn:oasis:names:tc:SAML:2.0:protocol">'
            '<KeyDescriptor use="signing"><KeyInfo xmlns="http://www.w3.org/2000/09/xmldsig#">'
            f"<X509Data><X509Certificate>{body}</X509Certificate></X509Data></KeyInfo></KeyDescriptor>"
            '<SingleSignOnService Binding="urn:oasis:names:tc:SAML:2.0:bindings:HTTP-Redirect" '
            'Location="https://idp.integration.invalid/sso"/>'
            "</IDPSSODescriptor></EntityDescriptor>"
        )

    def response(self, acs: str, request_id: str, name_id: str, email: str | None) -> str:
        now: Final = datetime.now(timezone.utc)
        issued: Final = (now - timedelta(seconds=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
        not_before: Final = (now - timedelta(seconds=60)).strftime("%Y-%m-%dT%H:%M:%SZ")
        expires: Final = (now + timedelta(seconds=300)).strftime("%Y-%m-%dT%H:%M:%SZ")
        attributes: Final = (
            '<saml:Attribute Name="givenName"><saml:AttributeValue>Saml</saml:AttributeValue></saml:Attribute>'
            if email is None
            else f'<saml:Attribute Name="email"><saml:AttributeValue>{email}</saml:AttributeValue></saml:Attribute>'
        )
        assertion: Final = (
            '<saml:Assertion xmlns:saml="urn:oasis:names:tc:SAML:2.0:assertion" '
            f'ID="_assertion_{uuid.uuid4().hex}" Version="2.0" IssueInstant="{issued}">'
            f"<saml:Issuer>{SAML_IDP_ENTITY}</saml:Issuer>"
            '<saml:Subject><saml:NameID Format="urn:oasis:names:tc:SAML:1.1:nameid-format:unspecified">'
            f"{name_id}</saml:NameID>"
            '<saml:SubjectConfirmation Method="urn:oasis:names:tc:SAML:2.0:cm:bearer">'
            f'<saml:SubjectConfirmationData InResponseTo="{request_id}" NotOnOrAfter="{expires}" Recipient="{acs}"/>'
            "</saml:SubjectConfirmation></saml:Subject>"
            f'<saml:Conditions NotBefore="{not_before}" NotOnOrAfter="{expires}">'
            f"<saml:AudienceRestriction><saml:Audience>{SAML_SP_ENTITY}</saml:Audience>"
            "</saml:AudienceRestriction></saml:Conditions>"
            f'<saml:AuthnStatement AuthnInstant="{issued}" SessionIndex="_session">'
            "<saml:AuthnContext><saml:AuthnContextClassRef>"
            "urn:oasis:names:tc:SAML:2.0:ac:classes:Password"
            "</saml:AuthnContextClassRef></saml:AuthnContext></saml:AuthnStatement>"
            f"<saml:AttributeStatement>{attributes}</saml:AttributeStatement>"
            "</saml:Assertion>"
        )
        signed: Final = OneLogin_Saml2_Utils.add_sign(assertion, self.key_pem, self.cert_pem)
        signed_text: Final = (signed.decode() if isinstance(signed, bytes) else signed).replace(
            '<?xml version="1.0"?>', ""
        )
        document: Final = (
            '<samlp:Response xmlns:samlp="urn:oasis:names:tc:SAML:2.0:protocol" '
            'xmlns:saml="urn:oasis:names:tc:SAML:2.0:assertion" '
            f'ID="_response_{uuid.uuid4().hex}" Version="2.0" IssueInstant="{issued}" '
            f'Destination="{acs}" InResponseTo="{request_id}">'
            f"<saml:Issuer>{SAML_IDP_ENTITY}</saml:Issuer>"
            '<samlp:Status><samlp:StatusCode Value="urn:oasis:names:tc:SAML:2.0:status:Success"/></samlp:Status>'
            f"{signed_text}</samlp:Response>"
        )
        return base64.b64encode(document.encode()).decode()


@pytest.fixture(scope="module")
def saml_idp() -> SamlIdp:
    key: Final = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name: Final = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "idp.integration.invalid")])
    now: Final = datetime.now(timezone.utc)
    cert: Final = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(days=1))
        .not_valid_after(now + timedelta(days=1))
        .sign(key, hashes.SHA256())
    )
    return SamlIdp(
        key_pem=key.private_bytes(
            serialization.Encoding.PEM, serialization.PrivateFormat.TraditionalOpenSSL, serialization.NoEncryption()
        ).decode(),
        cert_pem=cert.public_bytes(serialization.Encoding.PEM).decode(),
    )


@pytest.fixture(scope="module")
def saml(saml_idp: SamlIdp, gateway_module: Gateway, tmp_path_factory: pytest.TempPathFactory) -> Iterator[OwnedProxy]:
    with owned_proxy_process(
        gateway_module,
        tmp_path_factory.mktemp("sso-blank-identity-saml"),
        {"SAML_IDP_METADATA_XML": saml_idp.metadata(), "SAML_SP_ENTITY_ID": SAML_SP_ENTITY},
        remove_environment=("PROXY_BASE_URL",),
        workers=2,
    ) as owned:
        yield owned


def _saml_sign_in(proxy: Gateway, idp: SamlIdp, name_id: str, email: str | None) -> httpx.Response:
    acs: Final = f"{_proxy_url(proxy)}/sso/saml/callback"
    with _browser() as browser:
        login: Final = browser.get(f"{_proxy_url(proxy)}/sso/saml/login")
        assert login.is_redirect, f"{login.status_code} {login.text}"
        request_id: Final = login.cookies[SAML_STATE_COOKIE]
        return browser.post(acs, data={"SAMLResponse": idp.response(acs, request_id, name_id, email)})


@pytest.fixture(scope="module")
def graph() -> Iterator[Graph]:
    profiles: Final[dict[str, Reply]] = {}
    with wire_server(lambda request: _graph_reply(profiles, request)) as wire:
        yield Graph(wire, profiles)


@pytest.fixture(scope="module")
def entra(graph: Graph, gateway_module: Gateway, tmp_path_factory: pytest.TempPathFactory) -> Iterator[OwnedProxy]:
    directory: Final = tmp_path_factory.mktemp("sso-blank-identity")
    with owned_proxy_process(
        gateway_module,
        directory,
        _entra_environment(graph.wire.url),
        remove_environment=("PROXY_BASE_URL",),
        workers=2,
    ) as owned:
        yield owned


def _custom_mapping_config(directory: Path) -> Path:
    stock: Final = yaml.safe_load((Path(__file__).resolve().parents[1] / "proxy_config.yaml").read_text())
    (directory / "blank_identity_custom_sso.py").write_text(CUSTOM_SSO_MODULE)
    config: Final = directory / "custom_sso_config.yaml"
    config.write_text(
        json.dumps(
            {
                **stock,
                "general_settings": {
                    **stock["general_settings"],
                    "custom_sso": "blank_identity_custom_sso.custom_sso",
                },
            }
        )
    )
    return config


@pytest.fixture(scope="module")
def entra_custom_mapping(
    graph: Graph, gateway_module: Gateway, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[OwnedProxy]:
    directory: Final = tmp_path_factory.mktemp("sso-blank-identity-custom")
    with owned_proxy_process(
        gateway_module,
        directory,
        _entra_environment(graph.wire.url),
        config=_custom_mapping_config(directory),
        remove_environment=("PROXY_BASE_URL",),
        workers=2,
    ) as owned:
        yield owned


@pytest.fixture(scope="module")
def gateway_module() -> Iterator[Gateway]:
    with gateway_from_environment() as value:
        yield value


def _blank_accounts() -> tuple[str, ...]:
    rows: Final = read_rows('SELECT user_id FROM "LiteLLM_UserTable" WHERE user_id ~ %s', (BLANK,))
    return tuple(repr(row["user_id"]) for row in rows)


def _drop_blank_accounts() -> None:
    write_rows('DELETE FROM "LiteLLM_VerificationToken" WHERE user_id ~ %s', (BLANK,))
    write_rows('DELETE FROM "LiteLLM_UserTable" WHERE user_id ~ %s', (BLANK,))


def _forget_sign_in(scenario: Scenario, email: str) -> None:
    """Delete the account a sign-in creates for ``email``, and its session keys, whether or not the test passes."""
    scenario.cleanups.callback(write_rows, 'DELETE FROM "LiteLLM_UserTable" WHERE user_email = %s', (email,))
    scenario.cleanups.callback(
        write_rows,
        'DELETE FROM "LiteLLM_VerificationToken" WHERE user_id IN '
        '(SELECT user_id FROM "LiteLLM_UserTable" WHERE user_email = %s)',
        (email,),
    )


def _seed_account(user_id: str, email: str) -> None:
    write_rows(
        'INSERT INTO "LiteLLM_UserTable" (user_id, user_email, user_role, spend, models) '
        "VALUES (%s, %s, 'internal_user', 0, '{}')",
        (user_id, email),
    )


@pytest.fixture
def clean_slate() -> Iterator[None]:
    """Blank-identity accounts never belong in the database; a refused sign-in must not leave one behind."""
    _drop_blank_accounts()
    yield
    _drop_blank_accounts()


def _proxy_url(proxy: Gateway) -> str:
    return str(proxy.client.base_url).rstrip("/")


def _browser() -> httpx.Client:
    return httpx.Client(follow_redirects=False, trust_env=False, timeout=30)


def _through_graph(proxy: Gateway, graph: Graph, browser: httpx.Client, entry: str, hint: str) -> httpx.Response:
    link: Final = browser.get(entry)
    assert link.is_redirect, f"{link.status_code} {link.text}"
    authorize: Final = link.headers["location"]
    assert authorize.startswith(f"{graph.wire.url}/authorize?"), authorize
    assert parse_qs(urlparse(authorize).query)["redirect_uri"] == [f"{_proxy_url(proxy)}/sso/callback"], authorize
    at_graph: Final = browser.get(f"{authorize}&{urlencode({'login_hint': hint})}")
    assert at_graph.status_code == 302, f"{at_graph.status_code} {at_graph.text}"
    return browser.get(at_graph.headers["location"])


def _dashboard_sign_in(proxy: Gateway, graph: Graph, hint: str) -> httpx.Response:
    with _browser() as browser:
        return _through_graph(proxy, graph, browser, f"{_proxy_url(proxy)}/sso/key/generate", hint)


def _session(callback: httpx.Response) -> Mapping[str, object] | None:
    cookie: Final = callback.cookies.get("token")
    return None if cookie is None else jwt.decode(cookie, options={"verify_signature": False})


def _list_keys(proxy: Gateway, session_key: str) -> httpx.Response:
    return proxy.request("GET", "/key/list", key=session_key, params={"return_full_object": "true"})


def _listed_tokens(listed: httpx.Response) -> frozenset[str]:
    keys: Final = JSON_OBJECT.validate_json(listed.content)["keys"]
    assert isinstance(keys, list), listed.text
    return frozenset(string_value(JSON_OBJECT.validate_python(item)["token"]) for item in keys)


def _visible_tokens(proxy: Gateway, session_key: str) -> frozenset[str]:
    listed: Final = _list_keys(proxy, session_key)
    assert listed.status_code == 200, f"{listed.status_code} {listed.text}"
    return _listed_tokens(listed)


def _refusal_report(proxy: Gateway, callback: httpx.Response, victim_token: str) -> str:
    session: Final = _session(callback)
    if session is None:
        return f"{callback.status_code} {callback.text}"
    issued: Final = (
        f"{callback.status_code}: a dashboard session was issued for user_id={session['user_id']!r} "
        f"with role {session.get('user_role')!r}"
    )
    listed: Final = _list_keys(proxy, str(session["key"]))
    if listed.status_code != 200:
        return f"{issued}; its /key/list answered {listed.status_code}"
    visible: Final = _listed_tokens(listed)
    return f"{issued}; it lists {len(visible)} keys and the other account's key is visible={victim_token in visible}"


def _victim(scenario: Scenario) -> str:
    owner: Final = scenario.user(user_role="internal_user")
    return sha256(scenario.key(user_id=owner, key_alias=f"victim-{uuid.uuid4().hex}").encode()).hexdigest()


def _start_cli_login(proxy: Gateway) -> CliSession:
    response: Final = proxy.client.post("/sso/cli/start")
    assert response.status_code == 200, response.text
    body: Final = JSON_OBJECT.validate_json(response.content)
    return CliSession(
        string_value(body["login_id"]), string_value(body["poll_secret"]), string_value(body["user_code"])
    )


def _cli_callback(
    proxy: Gateway, graph: Graph, browser: httpx.Client, session: CliSession, hint: str
) -> httpx.Response:
    entry: Final = f"{_proxy_url(proxy)}/sso/key/generate?{urlencode({'source': CLI_SOURCE, 'key': session.login_id})}"
    return _through_graph(proxy, graph, browser, entry, hint)


def _complete_cli(browser: httpx.Client, callback: httpx.Response, session: CliSession) -> httpx.Response:
    assert callback.status_code == 200 and CLI_LOGIN_PAGE_TITLE in callback.text, (
        f"{callback.status_code} {callback.text}"
    )
    token: Final = COMPLETE_TOKEN_FIELD.search(callback.text)
    action: Final = COMPLETE_FORM_ACTION.search(callback.text)
    assert token is not None and action is not None, callback.text
    return browser.post(
        action.group(1), data={"user_code": session.user_code, "browser_complete_token": token.group(1)}
    )


def _poll(proxy: Gateway, session: CliSession) -> Mapping[str, object]:
    response: Final = proxy.client.get(
        f"/sso/cli/poll/{session.login_id}", headers={POLL_SECRET_HEADER: session.poll_secret}
    )
    assert response.status_code == 200, f"{response.status_code} {response.text}"
    return JSON_OBJECT.validate_json(response.content)


def _cli_report(proxy: Gateway, browser: httpx.Client, callback: httpx.Response, session: CliSession) -> str:
    if callback.status_code != 200:
        return f"{callback.status_code} {callback.text}"
    done: Final = _complete_cli(browser, callback, session)
    polled: Final = _poll(proxy, session)
    return (
        f"CLI login completed ({done.status_code}) and poll returned {polled.get('status')!r} "
        f"for user_id={polled.get('user_id')!r}"
    )


def test_dashboard_sign_in_is_refused_when_the_graph_profile_lookup_fails(
    entra: OwnedProxy, graph: Graph, gateway_module: Gateway, clean_slate: None
) -> None:
    proxy: Final = entra.gateway
    with gateway_module.scenario() as scenario:
        victim_token: Final = _victim(scenario)
        callback: Final = _dashboard_sign_in(proxy, graph, graph.profile(Reply(status=401, body=GRAPH_ERROR)))
        assert callback.status_code == 401 and _session(callback) is None, _refusal_report(
            proxy, callback, victim_token
        )
    assert JSON_OBJECT.validate_json(callback.content)["detail"] == REFUSED, callback.text
    assert _blank_accounts() == (), _blank_accounts()


def test_cli_sign_in_is_refused_when_the_graph_profile_lookup_fails(
    entra: OwnedProxy, graph: Graph, clean_slate: None
) -> None:
    proxy: Final = entra.gateway
    session: Final = _start_cli_login(proxy)
    with _browser() as browser:
        callback: Final = _cli_callback(
            proxy, graph, browser, session, graph.profile(Reply(status=401, body=GRAPH_ERROR))
        )
        assert callback.status_code == 401, _cli_report(proxy, browser, callback, session)
    assert JSON_OBJECT.validate_json(callback.content)["detail"] == REFUSED, callback.text
    assert _poll(proxy, session) == {"status": "pending"}
    assert _blank_accounts() == (), _blank_accounts()


def test_dashboard_sign_in_with_a_valid_profile_sees_only_its_own_keys(
    entra: OwnedProxy, graph: Graph, gateway_module: Gateway
) -> None:
    proxy: Final = entra.gateway
    subject: Final = str(uuid.uuid4())
    email: Final = f"valid-{uuid.uuid4().hex[:12]}@example.com"
    with gateway_module.scenario() as scenario:
        victim_token: Final = _victim(scenario)
        _forget_sign_in(scenario, email)
        callback: Final = _dashboard_sign_in(proxy, graph, graph.profile(_profile(subject, email)))
        assert callback.status_code == 303, f"{callback.status_code} {callback.text}"
        session: Final = _session(callback)
        assert session is not None and session["user_id"] == subject, session
        visible: Final = _visible_tokens(proxy, str(session["key"]))
        assert victim_token not in visible, visible
        rows: Final = read_rows('SELECT user_email FROM "LiteLLM_UserTable" WHERE user_id = %s', (subject,))
        assert [row["user_email"] for row in rows] == [email], rows


def test_cli_sign_in_with_a_valid_profile_completes(entra: OwnedProxy, graph: Graph, gateway_module: Gateway) -> None:
    proxy: Final = entra.gateway
    subject: Final = str(uuid.uuid4())
    email: Final = f"valid-cli-{uuid.uuid4().hex[:12]}@example.com"
    session: Final = _start_cli_login(proxy)
    with gateway_module.scenario() as scenario, _browser() as browser:
        _forget_sign_in(scenario, email)
        callback: Final = _cli_callback(proxy, graph, browser, session, graph.profile(_profile(subject, email)))
        done: Final = _complete_cli(browser, callback, session)
        assert done.status_code == 200 and CLI_SUCCESS_PAGE_TITLE in done.text, f"{done.status_code} {done.text}"
        ready: Final = _poll(proxy, session)
        assert ready["status"] == "ready" and ready["user_id"] == subject, ready


@pytest.mark.parametrize("blank_row", [False, True], ids=["no-blank-row", "blank-row-exists"])
def test_blank_provider_id_signs_in_as_the_mail_address(
    entra: OwnedProxy, graph: Graph, gateway_module: Gateway, clean_slate: None, blank_row: bool
) -> None:
    proxy: Final = entra.gateway
    email: Final = f"blank-id-{uuid.uuid4().hex[:12]}@example.com"
    if blank_row:
        _seed_account("   ", f"someone-else-{uuid.uuid4().hex[:12]}@example.com")
    with gateway_module.scenario() as scenario:
        _forget_sign_in(scenario, email)
        callback: Final = _dashboard_sign_in(proxy, graph, graph.profile(_profile("   ", email)))
        session: Final = _session(callback)
        assert callback.status_code == 303 and session is not None, f"{callback.status_code} {callback.text}"
        assert session["user_id"] == email, f"session issued for user_id={session['user_id']!r}"
    assert _blank_accounts() == (("'   '",) if blank_row else ()), _blank_accounts()


def test_dashboard_sign_in_that_resolves_to_a_blank_account_row_is_refused(
    entra: OwnedProxy, graph: Graph, gateway_module: Gateway, clean_slate: None
) -> None:
    proxy: Final = entra.gateway
    email: Final = f"shared-row-{uuid.uuid4().hex[:12]}@example.com"
    _seed_account("", email)
    with gateway_module.scenario() as scenario:
        victim_token: Final = _victim(scenario)
        callback: Final = _dashboard_sign_in(proxy, graph, graph.profile(_profile(str(uuid.uuid4()), email)))
        assert callback.status_code == 401 and _session(callback) is None, _refusal_report(
            proxy, callback, victim_token
        )
    assert JSON_OBJECT.validate_json(callback.content)["detail"] == REFUSED, callback.text
    tokens: Final = read_rows('SELECT token FROM "LiteLLM_VerificationToken" WHERE user_id ~ %s', (BLANK,))
    assert tokens == [], tokens


@pytest.mark.parametrize("blank_id", ["", "  "], ids=["empty", "whitespace"])
def test_cli_sign_in_that_resolves_to_a_blank_account_row_is_refused(
    entra: OwnedProxy, graph: Graph, clean_slate: None, blank_id: str
) -> None:
    proxy: Final = entra.gateway
    email: Final = f"blank-row-{uuid.uuid4().hex[:12]}@example.com"
    _seed_account(blank_id, email)
    session: Final = _start_cli_login(proxy)
    with _browser() as browser:
        callback: Final = _cli_callback(
            proxy, graph, browser, session, graph.profile(_profile(str(uuid.uuid4()), email))
        )
        assert callback.status_code == 401, _cli_report(proxy, browser, callback, session)
    assert JSON_OBJECT.validate_json(callback.content)["detail"] == REFUSED, callback.text
    assert _poll(proxy, session) == {"status": "pending"}


def test_cli_sign_in_completes_for_an_existing_account_when_the_custom_mapping_returns_none(
    entra_custom_mapping: OwnedProxy, graph: Graph, gateway_module: Gateway
) -> None:
    proxy: Final = entra_custom_mapping.gateway
    email: Final = f"unmapped-{uuid.uuid4().hex[:12]}@example.com"
    with gateway_module.scenario() as scenario, _browser() as browser:
        subject: Final = scenario.user(user_email=email, user_role="internal_user")
        session: Final = _start_cli_login(proxy)
        callback: Final = _cli_callback(proxy, graph, browser, session, graph.profile(_profile(subject, email)))
        done: Final = _complete_cli(browser, callback, session)
        assert done.status_code == 200 and CLI_SUCCESS_PAGE_TITLE in done.text, f"{done.status_code} {done.text}"
        ready: Final = _poll(proxy, session)
        assert ready["status"] == "ready" and ready["user_id"] == subject, ready


def test_sign_in_is_refused_when_the_custom_mapping_returns_a_blank_user_id(
    entra_custom_mapping: OwnedProxy, graph: Graph, gateway_module: Gateway, clean_slate: None
) -> None:
    proxy: Final = entra_custom_mapping.gateway
    with gateway_module.scenario() as scenario:
        victim_token: Final = _victim(scenario)
        hint: Final = graph.profile(_profile(str(uuid.uuid4()), f"mapped-blank-{uuid.uuid4().hex[:12]}@example.com"))
        callback: Final = _dashboard_sign_in(proxy, graph, hint)
        assert callback.status_code == 401 and _session(callback) is None, _refusal_report(
            proxy, callback, victim_token
        )
    assert JSON_OBJECT.validate_json(callback.content)["detail"] == REFUSED, callback.text
    session: Final = _start_cli_login(proxy)
    with _browser() as browser:
        cli_hint: Final = graph.profile(
            _profile(str(uuid.uuid4()), f"mapped-blank-{uuid.uuid4().hex[:12]}@example.com")
        )
        cli_callback: Final = _cli_callback(proxy, graph, browser, session, cli_hint)
        assert cli_callback.status_code == 401, _cli_report(proxy, browser, cli_callback, session)
    assert JSON_OBJECT.validate_json(cli_callback.content)["detail"] == REFUSED, cli_callback.text
    assert _poll(proxy, session) == {"status": "pending"}
    assert _blank_accounts() == (), _blank_accounts()


@pytest.mark.parametrize("prefix", ["mapped-none-", "mapped-blank-"])
def test_sign_in_uses_the_matching_account_when_the_custom_mapping_leaves_the_user_id_blank(
    entra_custom_mapping: OwnedProxy, graph: Graph, gateway_module: Gateway, prefix: str
) -> None:
    proxy: Final = entra_custom_mapping.gateway
    email: Final = f"{prefix}{uuid.uuid4().hex[:12]}@example.com"
    with gateway_module.scenario() as scenario, _browser() as browser:
        account: Final = scenario.user(user_email=email, user_role="internal_user")
        callback: Final = _dashboard_sign_in(proxy, graph, graph.profile(_profile(str(uuid.uuid4()), email)))
        session: Final = _session(callback)
        assert callback.status_code == 303 and session is not None, f"{callback.status_code} {callback.text}"
        assert session["user_id"] == account, session
        login: Final = _start_cli_login(proxy)
        cli_hint: Final = graph.profile(_profile(str(uuid.uuid4()), email))
        cli_callback: Final = _cli_callback(proxy, graph, browser, login, cli_hint)
        done: Final = _complete_cli(browser, cli_callback, login)
        assert done.status_code == 200 and CLI_SUCCESS_PAGE_TITLE in done.text, f"{done.status_code} {done.text}"
        ready: Final = _poll(proxy, login)
        assert ready["status"] == "ready" and ready["user_id"] == account, ready


def test_sign_in_creates_an_account_when_the_custom_mapping_leaves_the_user_id_unset(
    entra_custom_mapping: OwnedProxy, graph: Graph, gateway_module: Gateway
) -> None:
    proxy: Final = entra_custom_mapping.gateway
    email: Final = f"mapped-none-{uuid.uuid4().hex[:12]}@example.com"
    with gateway_module.scenario() as scenario:
        _forget_sign_in(scenario, email)
        callback: Final = _dashboard_sign_in(proxy, graph, graph.profile(_profile(str(uuid.uuid4()), email)))
        session: Final = _session(callback)
        assert callback.status_code == 303 and session is not None, f"{callback.status_code} {callback.text}"
        created: Final = read_rows('SELECT user_id FROM "LiteLLM_UserTable" WHERE user_email = %s', (email,))
        assert [row["user_id"] for row in created] == [session["user_id"]], created
        assert str(session["user_id"]).strip(), session


def test_sign_in_is_refused_while_the_graph_profile_service_is_down(
    entra: OwnedProxy, graph: Graph, clean_slate: None
) -> None:
    proxy: Final = entra.gateway
    outage: Final = Reply(
        status=503, body=b'{"error": {"code": "serviceNotAvailable", "message": "Service unavailable"}}'
    )
    callback: Final = _dashboard_sign_in(proxy, graph, graph.profile(outage))
    assert callback.status_code == 401 and _session(callback) is None, f"{callback.status_code} {callback.text}"
    session: Final = _start_cli_login(proxy)
    with _browser() as browser:
        cli_callback: Final = _cli_callback(proxy, graph, browser, session, graph.profile(outage))
        assert cli_callback.status_code == 401, _cli_report(proxy, browser, cli_callback, session)
    assert JSON_OBJECT.validate_json(cli_callback.content)["detail"] == REFUSED, cli_callback.text
    assert _poll(proxy, session) == {"status": "pending"}
    assert _blank_accounts() == (), _blank_accounts()


def test_concurrent_failed_profile_lookups_mint_no_session(entra: OwnedProxy, graph: Graph, clean_slate: None) -> None:
    proxy: Final = entra.gateway
    hints: Final = [graph.profile(Reply(status=401, body=GRAPH_ERROR)) for _ in range(8)]
    with ThreadPoolExecutor(max_workers=len(hints)) as pool:
        callbacks: Final = list(pool.map(lambda hint: _dashboard_sign_in(proxy, graph, hint), hints))
    assert [(callback.status_code, _session(callback)) for callback in callbacks] == [(401, None)] * len(hints), [
        f"{callback.status_code} {callback.text[:120]}" for callback in callbacks
    ]
    tokens: Final = read_rows('SELECT token FROM "LiteLLM_VerificationToken" WHERE user_id ~ %s', (BLANK,))
    assert tokens == [], f"{len(tokens)} blank-identity session keys were minted"
    assert _blank_accounts() == (), _blank_accounts()


def test_custom_mapped_sign_in_fails_with_500_when_the_account_lookup_loses_the_database(
    graph: Graph, gateway_module: Gateway, tmp_path: Path
) -> None:
    outage_marker: Final = uuid.uuid4().hex[:12]
    dashboard_email: Final = f"mapped-none-{outage_marker}-dashboard@example.com"
    cli_email: Final = f"mapped-none-{outage_marker}-cli@example.com"
    with (
        dropped_connection_relay(os.environ["DATABASE_URL"], outage_marker.encode()) as (relay, relayed_url),
        owned_proxy_process(
            gateway_module,
            tmp_path,
            {
                **_entra_environment(graph.wire.url),
                "DATABASE_URL": relayed_url,
                "PRISMA_HEALTH_WATCHDOG_ENABLED": "false",
            },
            config=_custom_mapping_config(tmp_path),
            remove_environment=("PROXY_BASE_URL", "DATABASE_URL_READ_REPLICA"),
            workers=2,
        ) as owned,
        gateway_module.scenario() as scenario,
    ):
        scenario.user(user_email=dashboard_email, user_role="internal_user")
        scenario.user(user_email=cli_email, user_role="internal_user")
        cli_login: Final = _start_cli_login(owned.gateway)
        relay.arm()
        callback: Final = _dashboard_sign_in(
            owned.gateway, graph, graph.profile(_profile(str(uuid.uuid4()), dashboard_email))
        )
        dashboard_lookup_dropped: Final = relay.dropped.is_set()
        relay.dropped.clear()
        with _browser() as browser:
            cli_callback: Final = _cli_callback(
                owned.gateway, graph, browser, cli_login, graph.profile(_profile(str(uuid.uuid4()), cli_email))
            )
        relay.disarm()
        assert dashboard_lookup_dropped and relay.dropped.is_set(), "an account lookup never reached the database"
        session: Final = _session(callback)
        assert callback.status_code == 500 and session is None, (
            f"{callback.status_code}: session for user_id={None if session is None else session['user_id']!r}"
        )
        assert JSON_OBJECT.validate_json(callback.content)["detail"] == LOOKUP_FAILED, callback.text
        assert cli_callback.status_code == 500, f"{cli_callback.status_code} {cli_callback.text}"
        assert JSON_OBJECT.validate_json(cli_callback.content)["detail"] == LOOKUP_FAILED, cli_callback.text
        assert _poll(owned.gateway, cli_login) == {"status": "pending"}


def test_saml_sign_in_with_a_whitespace_name_id_and_no_email_is_refused(
    saml: OwnedProxy, saml_idp: SamlIdp, gateway_module: Gateway, clean_slate: None
) -> None:
    proxy: Final = saml.gateway
    with gateway_module.scenario() as scenario:
        victim_token: Final = _victim(scenario)
        callback: Final = _saml_sign_in(proxy, saml_idp, "   ", None)
        assert callback.status_code == 401 and _session(callback) is None, _refusal_report(
            proxy, callback, victim_token
        )
    assert JSON_OBJECT.validate_json(callback.content)["detail"] == REFUSED, callback.text
    assert _blank_accounts() == (), _blank_accounts()


@pytest.mark.parametrize("blank_name_id", [True, False], ids=["whitespace-name-id", "name-id"])
def test_saml_sign_in_with_an_email_signs_in_as_a_real_account(
    saml: OwnedProxy, saml_idp: SamlIdp, gateway_module: Gateway, clean_slate: None, blank_name_id: bool
) -> None:
    proxy: Final = saml.gateway
    email: Final = f"saml-{uuid.uuid4().hex[:12]}@example.com"
    name_id: Final = "   " if blank_name_id else f"saml-subject-{uuid.uuid4().hex[:12]}"
    expected: Final = email if blank_name_id else name_id
    with gateway_module.scenario() as scenario:
        _forget_sign_in(scenario, email)
        callback: Final = _saml_sign_in(proxy, saml_idp, name_id, email)
        session: Final = _session(callback)
        assert callback.status_code == 303 and session is not None, f"{callback.status_code} {callback.text}"
        assert session["user_id"] == expected, f"session issued for user_id={session['user_id']!r}"
    assert _blank_accounts() == (), _blank_accounts()
