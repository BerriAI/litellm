"""
Regression tests for SAML 2.0 SSO (SP- and IdP-initiated) on the admin UI.

These exercise the real OneLogin python3-saml validation by generating signed
SAML responses with a freshly minted IdP keypair, so a mutation that weakens
signature, signing-requirement, expiry, replay or attribute-mapping handling
makes a test fail.
"""

import base64
import datetime
import time

import pytest
from fastapi import HTTPException, Request

pytest.importorskip(
    "onelogin", reason="python3-saml (saml extra) is required for SAML SSO tests"
)

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID
from onelogin.saml2.utils import OneLogin_Saml2_Utils
from starlette.datastructures import URL

from typing import cast

from litellm.caching.dual_cache import DualCache
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.caching.redis_cache import RedisCache
from litellm.proxy._types import LitellmUserRoles
from litellm.proxy.management_endpoints.sso.saml_sso import (
    _SAML_AUTHN_REQUEST_CACHE_PREFIX,
    _SAML_AUTHN_STATE_COOKIE,
    _SAML_MAX_POST_BYTES,
    _SAML_REPLAY_GUARD_DEFAULT_TTL_SECONDS,
    _SAML_REPLAY_GUARD_MAX_TTL_SECONDS,
    SAMLAuthHandler,
)


def _shared_cache(store=None):
    """A DualCache whose replay guard is backed by a shared, atomic store.

    An InMemoryCache instance stands in for Redis; passing the same instance to
    two DualCaches simulates two workers sharing one atomic backend."""
    return DualCache(redis_cache=cast(RedisCache, store or InMemoryCache()))

IDP_ENTITY = "https://idp.example.com/metadata"
SP_ENTITY = "https://proxy.example.com/sso/saml/metadata"
ACS = "https://proxy.example.com/sso/saml/callback"
SSO_URL = "https://idp.example.com/sso"
PROXY_BASE_URL = "https://proxy.example.com"


def _make_idp_keypair():
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "idp.example.com")])
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime.utcnow() - datetime.timedelta(days=1))
        .not_valid_after(datetime.datetime.utcnow() + datetime.timedelta(days=365))
        .sign(key, hashes.SHA256())
    )
    key_pem = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.TraditionalOpenSSL,
        serialization.NoEncryption(),
    ).decode()
    cert_pem = cert.public_bytes(serialization.Encoding.PEM).decode()
    return key_pem, cert_pem


def _idp_metadata_xml(cert_pem):
    cert_body = "".join(
        line for line in cert_pem.splitlines() if "CERTIFICATE" not in line
    )
    return (
        '<?xml version="1.0"?>'
        f'<EntityDescriptor xmlns="urn:oasis:names:tc:SAML:2.0:metadata" entityID="{IDP_ENTITY}">'
        '<IDPSSODescriptor protocolSupportEnumeration="urn:oasis:names:tc:SAML:2.0:protocol">'
        '<KeyDescriptor use="signing"><KeyInfo xmlns="http://www.w3.org/2000/09/xmldsig#">'
        f"<X509Data><X509Certificate>{cert_body}</X509Certificate></X509Data>"
        "</KeyInfo></KeyDescriptor>"
        '<SingleSignOnService Binding="urn:oasis:names:tc:SAML:2.0:bindings:HTTP-Redirect" '
        f'Location="{SSO_URL}"/>'
        "</IDPSSODescriptor></EntityDescriptor>"
    )


def _saml_time(delta_seconds):
    t = datetime.datetime.utcnow() + datetime.timedelta(seconds=delta_seconds)
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def _build_signed_response(
    key_pem,
    cert_pem,
    *,
    in_response_to=None,
    response_level_in_response_to=True,
    email="alice@example.com",
    attributes=None,
    not_before_delta=-60,
    not_on_or_after_delta=300,
    sign=True,
):
    if attributes is None:
        attributes = {
            "email": [email],
            "givenName": ["Alice"],
            "sn": ["Smith"],
            "role": ["internal_user"],
        }
    assertion_id = "_assertion_" + OneLogin_Saml2_Utils.generate_unique_id()
    response_id = "_response_" + OneLogin_Saml2_Utils.generate_unique_id()
    not_before = _saml_time(not_before_delta)
    not_on_or_after = _saml_time(not_on_or_after_delta)
    issue_instant = _saml_time(-1)
    irt = f'InResponseTo="{in_response_to}"' if in_response_to else ""
    response_irt = irt if response_level_in_response_to else ""

    attr_xml = "".join(
        f'<saml:Attribute Name="{name}">'
        + "".join(f"<saml:AttributeValue>{v}</saml:AttributeValue>" for v in values)
        + "</saml:Attribute>"
        for name, values in attributes.items()
    )

    assertion = (
        '<saml:Assertion xmlns:saml="urn:oasis:names:tc:SAML:2.0:assertion" '
        f'ID="{assertion_id}" Version="2.0" IssueInstant="{issue_instant}">'
        f"<saml:Issuer>{IDP_ENTITY}</saml:Issuer>"
        "<saml:Subject>"
        '<saml:NameID Format="urn:oasis:names:tc:SAML:1.1:nameid-format:emailAddress">'
        f"{email}</saml:NameID>"
        '<saml:SubjectConfirmation Method="urn:oasis:names:tc:SAML:2.0:cm:bearer">'
        f'<saml:SubjectConfirmationData {irt} NotOnOrAfter="{not_on_or_after}" Recipient="{ACS}"/>'
        "</saml:SubjectConfirmation></saml:Subject>"
        f'<saml:Conditions NotBefore="{not_before}" NotOnOrAfter="{not_on_or_after}">'
        f"<saml:AudienceRestriction><saml:Audience>{SP_ENTITY}</saml:Audience>"
        "</saml:AudienceRestriction></saml:Conditions>"
        f'<saml:AuthnStatement AuthnInstant="{issue_instant}" SessionIndex="_session">'
        "<saml:AuthnContext><saml:AuthnContextClassRef>"
        "urn:oasis:names:tc:SAML:2.0:ac:classes:Password"
        "</saml:AuthnContextClassRef></saml:AuthnContext></saml:AuthnStatement>"
        f"<saml:AttributeStatement>{attr_xml}</saml:AttributeStatement>"
        "</saml:Assertion>"
    )

    if sign:
        signed = OneLogin_Saml2_Utils.add_sign(assertion, key_pem, cert_pem)
        assertion = (signed.decode() if isinstance(signed, bytes) else signed).replace(
            '<?xml version="1.0"?>', ""
        )

    return (
        '<?xml version="1.0"?>'
        '<samlp:Response xmlns:samlp="urn:oasis:names:tc:SAML:2.0:protocol" '
        'xmlns:saml="urn:oasis:names:tc:SAML:2.0:assertion" '
        f'ID="{response_id}" Version="2.0" IssueInstant="{issue_instant}" '
        f'Destination="{ACS}" {response_irt}>'
        f"<saml:Issuer>{IDP_ENTITY}</saml:Issuer>"
        '<samlp:Status><samlp:StatusCode Value="urn:oasis:names:tc:SAML:2.0:status:Success"/>'
        "</samlp:Status>"
        f"{assertion}</samlp:Response>"
    )


def _b64(xml):
    return base64.b64encode(xml.encode()).decode()


def _fake_request(cookies=None):
    return type(
        "Req",
        (),
        {
            "base_url": URL(PROXY_BASE_URL + "/"),
            "query_params": {},
            "cookies": cookies or {},
        },
    )()


async def _acs(b64, cache, cookies=None):
    return await SAMLAuthHandler.handle_acs(
        _fake_request(cookies), cache, {"SAMLResponse": b64}
    )


@pytest.fixture
def saml_env(monkeypatch):
    key_pem, cert_pem = _make_idp_keypair()
    monkeypatch.setenv("SAML_IDP_METADATA_XML", _idp_metadata_xml(cert_pem))
    monkeypatch.setenv("SAML_SP_ENTITY_ID", SP_ENTITY)
    monkeypatch.setenv("PROXY_BASE_URL", PROXY_BASE_URL)
    for var in (
        "SAML_IDP_METADATA_URL",
        "SAML_ATTRIBUTE_EMAIL",
        "SAML_ATTRIBUTE_TEAM_IDS",
        "SAML_ALLOW_UNSOLICITED",
        "ALLOWED_EMAIL_DOMAINS",
    ):
        monkeypatch.delenv(var, raising=False)
    return key_pem, cert_pem


@pytest.fixture
def saml_env_idp_initiated(saml_env, monkeypatch):
    monkeypatch.setenv("SAML_ALLOW_UNSOLICITED", "true")
    return saml_env


@pytest.mark.asyncio
async def test_valid_idp_initiated_login_maps_assertion_to_user(saml_env_idp_initiated):
    key_pem, cert_pem = saml_env_idp_initiated
    resp = _build_signed_response(key_pem, cert_pem)

    result = await _acs(_b64(resp), _shared_cache())

    assert result.email == "alice@example.com"
    assert result.id == "alice@example.com"
    assert result.first_name == "Alice"
    assert result.last_name == "Smith"
    assert result.user_role == LitellmUserRoles.INTERNAL_USER
    assert result.provider == "saml"


@pytest.mark.asyncio
async def test_tampered_assertion_is_rejected(saml_env):
    key_pem, cert_pem = saml_env
    resp = _build_signed_response(key_pem, cert_pem)
    tampered = resp.replace("alice@example.com", "attacker@example.com")

    with pytest.raises(HTTPException) as exc:
        await _acs(_b64(tampered), DualCache())
    assert exc.value.status_code == 401


@pytest.mark.asyncio
async def test_unsigned_assertion_is_rejected(saml_env):
    key_pem, cert_pem = saml_env
    resp = _build_signed_response(key_pem, cert_pem, sign=False)

    with pytest.raises(HTTPException) as exc:
        await _acs(_b64(resp), DualCache())
    assert exc.value.status_code == 401


@pytest.mark.asyncio
async def test_signature_from_untrusted_key_is_rejected(saml_env):
    _, cert_pem = saml_env
    attacker_key, attacker_cert = _make_idp_keypair()
    resp = _build_signed_response(attacker_key, attacker_cert)

    with pytest.raises(HTTPException) as exc:
        await _acs(_b64(resp), DualCache())
    assert exc.value.status_code == 401


@pytest.mark.asyncio
async def test_expired_assertion_is_rejected(saml_env):
    key_pem, cert_pem = saml_env
    resp = _build_signed_response(
        key_pem, cert_pem, not_before_delta=-7200, not_on_or_after_delta=-3600
    )

    with pytest.raises(HTTPException) as exc:
        await _acs(_b64(resp), DualCache())
    assert exc.value.status_code == 401


@pytest.mark.asyncio
async def test_sp_initiated_unknown_in_response_to_is_rejected(saml_env):
    key_pem, cert_pem = saml_env
    resp = _build_signed_response(key_pem, cert_pem, in_response_to="_never_issued")

    with pytest.raises(HTTPException) as exc:
        await _acs(_b64(resp), DualCache())
    assert exc.value.status_code == 401


@pytest.mark.asyncio
async def test_sp_initiated_known_request_succeeds_once_then_replay_rejected(saml_env):
    key_pem, cert_pem = saml_env
    cache = DualCache()
    request_id = "_authn_req_known"
    cache.set_cache(
        key=f"{_SAML_AUTHN_REQUEST_CACHE_PREFIX}:{request_id}", value="1", ttl=600
    )
    resp = _build_signed_response(key_pem, cert_pem, in_response_to=request_id)
    cookies = {_SAML_AUTHN_STATE_COOKIE: request_id}

    result = await _acs(_b64(resp), cache, cookies=cookies)
    assert result.email == "alice@example.com"

    with pytest.raises(HTTPException) as exc:
        await _acs(_b64(resp), cache, cookies=cookies)
    assert exc.value.status_code == 401


@pytest.mark.asyncio
async def test_sp_initiated_response_not_bound_to_browser_is_rejected(saml_env):
    key_pem, cert_pem = saml_env
    cache = DualCache()
    request_id = "_authn_req_known"
    cache.set_cache(
        key=f"{_SAML_AUTHN_REQUEST_CACHE_PREFIX}:{request_id}", value="1", ttl=600
    )
    resp = _build_signed_response(key_pem, cert_pem, in_response_to=request_id)

    with pytest.raises(HTTPException) as exc:
        await _acs(_b64(resp), cache)
    assert exc.value.status_code == 401

    with pytest.raises(HTTPException) as exc:
        await _acs(
            _b64(resp), cache, cookies={_SAML_AUTHN_STATE_COOKIE: "_attacker_request"}
        )
    assert exc.value.status_code == 401


@pytest.mark.asyncio
async def test_subjectconfirmation_only_in_response_to_without_cookie_is_rejected(
    saml_env_idp_initiated,
):
    """An IdP that stamps InResponseTo only on the SubjectConfirmationData (not the
    Response element) is still solicited and must be browser-bound: with unsolicited
    explicitly allowed, a missing cookie must still 401 rather than slip through."""
    key_pem, cert_pem = saml_env_idp_initiated
    cache = DualCache()
    request_id = "_authn_req_known"
    cache.set_cache(
        key=f"{_SAML_AUTHN_REQUEST_CACHE_PREFIX}:{request_id}", value="1", ttl=600
    )
    resp = _build_signed_response(
        key_pem,
        cert_pem,
        in_response_to=request_id,
        response_level_in_response_to=False,
    )

    with pytest.raises(HTTPException) as exc:
        await _acs(_b64(resp), cache)
    assert exc.value.status_code == 401


@pytest.mark.asyncio
async def test_subjectconfirmation_only_in_response_to_with_cookie_succeeds(saml_env):
    key_pem, cert_pem = saml_env
    cache = DualCache()
    request_id = "_authn_req_known"
    cache.set_cache(
        key=f"{_SAML_AUTHN_REQUEST_CACHE_PREFIX}:{request_id}", value="1", ttl=600
    )
    resp = _build_signed_response(
        key_pem,
        cert_pem,
        in_response_to=request_id,
        response_level_in_response_to=False,
    )

    result = await _acs(
        _b64(resp), cache, cookies={_SAML_AUTHN_STATE_COOKIE: request_id}
    )
    assert result.email == "alice@example.com"


@pytest.mark.asyncio
async def test_unsolicited_response_rejected_by_default(saml_env):
    key_pem, cert_pem = saml_env
    resp = _build_signed_response(key_pem, cert_pem)

    with pytest.raises(HTTPException) as exc:
        await _acs(_b64(resp), DualCache())
    assert exc.value.status_code == 401


@pytest.mark.asyncio
async def test_idp_initiated_assertion_replay_is_rejected(saml_env_idp_initiated):
    key_pem, cert_pem = saml_env_idp_initiated
    cache = _shared_cache()
    resp = _build_signed_response(key_pem, cert_pem, email="bob@example.com")

    first = await _acs(_b64(resp), cache)
    assert first.email == "bob@example.com"

    with pytest.raises(HTTPException) as exc:
        await _acs(_b64(resp), cache)
    assert exc.value.status_code == 401


@pytest.mark.asyncio
async def test_assertion_without_id_is_rejected(saml_env_idp_initiated):
    """An assertion with no ID attribute has no stable replay key. On the unsolicited
    path there is no browser binding, so the consumed-assertion guard is the only replay
    defense; a missing ID must be rejected rather than silently skipping the guard."""

    class _AuthNoAssertionId:
        def get_last_response_in_response_to(self):
            return None

        def get_last_response_xml(self):
            return None

        def get_last_assertion_id(self):
            return None

    with pytest.raises(HTTPException) as exc:
        await SAMLAuthHandler._enforce_response_binding(
            _AuthNoAssertionId(), _shared_cache(), None
        )
    assert exc.value.status_code == 401
    assert "ID" in exc.value.detail


@pytest.mark.asyncio
async def test_unsolicited_response_rejected_when_disabled(saml_env, monkeypatch):
    key_pem, cert_pem = saml_env
    monkeypatch.setenv("SAML_ALLOW_UNSOLICITED", "false")
    resp = _build_signed_response(key_pem, cert_pem)

    with pytest.raises(HTTPException) as exc:
        await _acs(_b64(resp), DualCache())
    assert exc.value.status_code == 401


@pytest.mark.asyncio
async def test_invalid_email_in_assertion_is_rejected_cleanly(saml_env_idp_initiated):
    key_pem, cert_pem = saml_env_idp_initiated
    resp = _build_signed_response(
        key_pem,
        cert_pem,
        email="not-an-email",
        attributes={"email": ["not-an-email"], "givenName": ["X"]},
    )

    with pytest.raises(HTTPException) as exc:
        await _acs(_b64(resp), _shared_cache())
    assert exc.value.status_code == 401
    assert "invalid subject or email" in exc.value.detail


@pytest.mark.asyncio
async def test_email_less_assertion_rejected_when_domain_restriction_configured(
    saml_env_idp_initiated, monkeypatch
):
    key_pem, cert_pem = saml_env_idp_initiated
    monkeypatch.setenv("ALLOWED_EMAIL_DOMAINS", "example.com")
    resp = _build_signed_response(
        key_pem,
        cert_pem,
        email="opaque-persistent-id-123",
        attributes={"givenName": ["Alice"]},
    )

    with pytest.raises(HTTPException) as exc:
        await _acs(_b64(resp), _shared_cache())
    assert exc.value.status_code == 401
    assert "ALLOWED_EMAIL_DOMAINS" in exc.value.detail


@pytest.mark.asyncio
async def test_email_less_assertion_allowed_without_domain_restriction(saml_env_idp_initiated):
    key_pem, cert_pem = saml_env_idp_initiated
    resp = _build_signed_response(
        key_pem,
        cert_pem,
        email="opaque-persistent-id-123",
        attributes={"givenName": ["Alice"]},
    )

    result = await _acs(_b64(resp), _shared_cache())
    assert result.email is None
    assert result.id == "opaque-persistent-id-123"


@pytest.mark.asyncio
async def test_custom_email_attribute_override(saml_env_idp_initiated, monkeypatch):
    key_pem, cert_pem = saml_env_idp_initiated
    monkeypatch.setenv("SAML_ATTRIBUTE_EMAIL", "corpMail")
    resp = _build_signed_response(
        key_pem,
        cert_pem,
        email="ignored@example.com",
        attributes={
            "corpMail": ["real@corp.example.com"],
            "givenName": ["Real"],
        },
    )

    result = await _acs(_b64(resp), _shared_cache())
    assert result.email == "real@corp.example.com"


@pytest.mark.asyncio
async def test_team_ids_extracted_from_groups_attribute(saml_env_idp_initiated):
    key_pem, cert_pem = saml_env_idp_initiated
    resp = _build_signed_response(
        key_pem,
        cert_pem,
        attributes={
            "email": ["carol@example.com"],
            "groups": ["team-a", "team-b"],
        },
    )

    result = await _acs(_b64(resp), _shared_cache())
    assert result.team_ids == ["team-a", "team-b"]


@pytest.mark.asyncio
async def test_build_login_redirect_targets_idp_and_caches_request_id(saml_env):
    cache = DualCache()
    redirect = await SAMLAuthHandler.build_login_redirect(_fake_request(), cache)

    location = redirect.headers["location"]
    assert location.startswith(SSO_URL)
    assert "SAMLRequest=" in location
    cached = [
        k
        for k in cache.in_memory_cache.cache_dict
        if k.startswith(_SAML_AUTHN_REQUEST_CACHE_PREFIX)
    ]
    assert len(cached) == 1

    request_id = cached[0].split(":", 1)[1]
    set_cookie = redirect.headers["set-cookie"]
    assert f"{_SAML_AUTHN_STATE_COOKIE}={request_id}" in set_cookie
    assert "httponly" in set_cookie.lower()


@pytest.mark.asyncio
async def test_sp_metadata_contains_acs_and_entity_id(saml_env):
    metadata = await SAMLAuthHandler.build_sp_metadata(_fake_request(), DualCache())
    assert ACS in metadata
    assert SP_ENTITY in metadata
    assert "AssertionConsumerService" in metadata


def test_replay_guard_ttl_tracks_assertion_validity():
    class _Auth:
        def __init__(self, not_on_or_after):
            self._not_on_or_after = not_on_or_after

        def get_last_assertion_not_on_or_after(self):
            return self._not_on_or_after

    now = int(time.time())

    long_lived = SAMLAuthHandler._replay_guard_ttl(_Auth(now + 7200))
    assert long_lived >= 7200

    short_lived = SAMLAuthHandler._replay_guard_ttl(_Auth(now + 60))
    assert short_lived == _SAML_REPLAY_GUARD_DEFAULT_TTL_SECONDS

    missing = SAMLAuthHandler._replay_guard_ttl(_Auth(None))
    assert missing == _SAML_REPLAY_GUARD_DEFAULT_TTL_SECONDS

    capped = SAMLAuthHandler._replay_guard_ttl(_Auth(now + 10 * 86400))
    assert capped == _SAML_REPLAY_GUARD_MAX_TTL_SECONDS


def test_is_saml_configured_reflects_env(monkeypatch):
    monkeypatch.delenv("SAML_IDP_METADATA_URL", raising=False)
    monkeypatch.delenv("SAML_IDP_METADATA_XML", raising=False)
    assert SAMLAuthHandler.is_saml_configured() is False

    monkeypatch.setenv("SAML_IDP_METADATA_URL", "https://idp.example.com/metadata.xml")
    assert SAMLAuthHandler.is_saml_configured() is True


@pytest.mark.asyncio
async def test_idp_initiated_rejected_without_shared_cache(saml_env_idp_initiated):
    key_pem, cert_pem = saml_env_idp_initiated
    resp = _build_signed_response(key_pem, cert_pem)

    with pytest.raises(HTTPException) as exc:
        await _acs(_b64(resp), DualCache())
    assert exc.value.status_code == 401
    assert "shared Redis cache" in exc.value.detail


@pytest.mark.asyncio
async def test_idp_initiated_replay_rejected_across_workers(saml_env_idp_initiated):
    key_pem, cert_pem = saml_env_idp_initiated
    shared_store = InMemoryCache()
    worker_one = _shared_cache(shared_store)
    worker_two = _shared_cache(shared_store)
    resp = _build_signed_response(key_pem, cert_pem, email="bob@example.com")

    first = await _acs(_b64(resp), worker_one)
    assert first.email == "bob@example.com"

    with pytest.raises(HTTPException) as exc:
        await _acs(_b64(resp), worker_two)
    assert exc.value.status_code == 401


class _FakeChunkedRequest:
    def __init__(self, chunks, content_length=None):
        self._chunks = chunks
        self.headers = {} if content_length is None else {"content-length": content_length}

    async def stream(self):
        for chunk in self._chunks:
            yield chunk


@pytest.mark.asyncio
async def test_read_acs_post_data_parses_form():
    body = b"SAMLResponse=abc123&RelayState=%2Fui%2F"
    request = _FakeChunkedRequest([body], content_length=str(len(body)))

    post_data = await SAMLAuthHandler.read_acs_post_data(cast(Request, request))

    assert post_data == {"SAMLResponse": "abc123", "RelayState": "/ui/"}


@pytest.mark.asyncio
async def test_read_acs_post_data_rejects_oversized_content_length():
    request = _FakeChunkedRequest([b""], content_length=str(_SAML_MAX_POST_BYTES + 1))

    with pytest.raises(HTTPException) as exc:
        await SAMLAuthHandler.read_acs_post_data(cast(Request, request))
    assert exc.value.status_code == 413


@pytest.mark.asyncio
async def test_read_acs_post_data_rejects_oversized_stream_without_content_length():
    chunk = b"a" * (1024 * 1024)
    chunk_count = _SAML_MAX_POST_BYTES // len(chunk) + 2
    request = _FakeChunkedRequest([chunk] * chunk_count)

    with pytest.raises(HTTPException) as exc:
        await SAMLAuthHandler.read_acs_post_data(cast(Request, request))
    assert exc.value.status_code == 413
