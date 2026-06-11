from __future__ import annotations

import base64
import datetime
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import pytest
from fastapi import FastAPI, Security
from fastapi.testclient import TestClient

xmlsec1 = shutil.which("xmlsec1")
pytestmark = pytest.mark.skipif(
    xmlsec1 is None, reason="SAML SP requires the xmlsec1 binary on PATH"
)

SP_ENTITY_ID = "https://sp.test.litellm.ai/auth/saml/metadata"
ACS_URL = "https://sp.test.litellm.ai/auth/saml/acs"
IDP_ENTITY_ID = "https://idp.test.litellm.ai/idp"
IDP_SSO_URL = "https://idp.test.litellm.ai/sso"


def _gen_cert(directory: Path, prefix: str) -> tuple[str, str]:
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, prefix)])
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime(2020, 1, 1))
        .not_valid_after(datetime.datetime(2035, 1, 1))
        .sign(key, hashes.SHA256())
    )
    key_path = directory / f"{prefix}.key"
    cert_path = directory / f"{prefix}.crt"
    key_path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.TraditionalOpenSSL,
            serialization.NoEncryption(),
        )
    )
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    return str(key_path), str(cert_path)


@dataclass
class SamlEnv:
    config: Any
    idp: Any  # saml2.server.Server
    name_id_value: str = "alice@example.com"

    def mint_response(
        self,
        *,
        identity: Optional[Dict[str, Any]] = None,
        sign_assertion: bool = True,
    ) -> str:
        from saml2.authn_context import PASSWORD
        from saml2.saml import NAMEID_FORMAT_EMAILADDRESS, NameID

        name_id = NameID(format=NAMEID_FORMAT_EMAILADDRESS, text=self.name_id_value)
        response = self.idp.create_authn_response(
            identity=identity
            or {
                "email": ["alice@example.com"],
                "displayName": ["Alice Anderson"],
                "groups": ["eng", "admins"],
            },
            in_response_to=None,
            destination=ACS_URL,
            sp_entity_id=SP_ENTITY_ID,
            name_id=name_id,
            sign_assertion=sign_assertion,
            authn={"class_ref": PASSWORD, "authn_auth": IDP_ENTITY_ID},
        )
        return base64.b64encode(str(response).encode()).decode()


@pytest.fixture
def saml_env(tmp_path: Path) -> SamlEnv:
    from saml2 import BINDING_HTTP_POST, BINDING_HTTP_REDIRECT
    from saml2.config import IdPConfig, SPConfig
    from saml2.metadata import entity_descriptor
    from saml2.saml import NAMEID_FORMAT_EMAILADDRESS
    from saml2.server import Server

    from litellm.proxy.auth_v2.config import SamlConfig

    idp_key, idp_cert = _gen_cert(tmp_path, "idp")
    sp_key, sp_cert = _gen_cert(tmp_path, "sp")

    sp_conf = SPConfig()
    sp_conf.load(
        {
            "entityid": SP_ENTITY_ID,
            "service": {
                "sp": {
                    "endpoints": {
                        "assertion_consumer_service": [(ACS_URL, BINDING_HTTP_POST)]
                    },
                    "allow_unsolicited": True,
                    "authn_requests_signed": False,
                    "want_assertions_signed": True,
                    "want_response_signed": False,
                }
            },
            "allow_unknown_attributes": True,
            "xmlsec_binary": xmlsec1,
        }
    )
    sp_metadata_path = tmp_path / "sp_metadata.xml"
    sp_metadata_path.write_text(str(entity_descriptor(sp_conf)))

    idp_conf = IdPConfig()
    idp_conf.load(
        {
            "entityid": IDP_ENTITY_ID,
            "service": {
                "idp": {
                    "endpoints": {
                        "single_sign_on_service": [(IDP_SSO_URL, BINDING_HTTP_REDIRECT)]
                    },
                    "name_id_format": [NAMEID_FORMAT_EMAILADDRESS],
                }
            },
            "metadata": {"local": [str(sp_metadata_path)]},
            "key_file": idp_key,
            "cert_file": idp_cert,
            "xmlsec_binary": xmlsec1,
        }
    )
    idp = Server(config=idp_conf)
    idp_metadata = str(entity_descriptor(idp.config))

    config = SamlConfig(
        enabled=True,
        entity_id=SP_ENTITY_ID,
        acs_url=ACS_URL,
        idp_metadata=idp_metadata,
        sp_key_file=sp_key,
        sp_cert_file=sp_cert,
        xmlsec_binary=xmlsec1,
        # this harness mints IdP-initiated (unsolicited) responses; pin the config
        # explicitly so the suite is independent of the allow_unsolicited default
        allow_unsolicited=True,
    )
    return SamlEnv(config=config, idp=idp)


def _build_app(saml_env: SamlEnv):
    from litellm.proxy.auth_v2.config import AuthConfig
    from litellm.proxy.auth_v2.models import Principal
    from litellm.proxy.auth_v2.resolver import InMemoryIdentityStore
    from litellm.proxy.auth_v2.security import get_current_principal, install_auth

    app = FastAPI()
    store = InMemoryIdentityStore()
    install_auth(
        app,
        AuthConfig(saml=saml_env.config),
        store,
        mount_scim=False,
        mount_oidc=False,
        mount_saml=True,
    )

    @app.get("/whoami")
    async def whoami(
        principal: "Principal" = Security(get_current_principal),
    ):
        return {
            "subject": principal.subject,
            "auth_method": principal.auth_method.value,
            "email": principal.user.email if principal.user else None,
        }

    return app, store


# --------------------------------------------------------------------------- #
# Metadata + login redirect
# --------------------------------------------------------------------------- #


def test_metadata_endpoint_serves_sp_descriptor(saml_env):
    app, _ = _build_app(saml_env)
    client = TestClient(app)
    response = client.get("/auth/saml/metadata")
    assert response.status_code == 200
    assert "EntityDescriptor" in response.text
    assert SP_ENTITY_ID in response.text
    assert ACS_URL in response.text


def test_login_redirects_to_idp_sso(saml_env):
    app, _ = _build_app(saml_env)
    client = TestClient(app)
    response = client.get("/auth/saml/login", follow_redirects=False)
    assert response.status_code == 303
    assert response.headers["location"].startswith(IDP_SSO_URL)


# --------------------------------------------------------------------------- #
# ACS: signed assertion provisions + authenticates; tampering is rejected
# --------------------------------------------------------------------------- #


def test_acs_accepts_signed_assertion_and_provisions_user(saml_env):
    app, store = _build_app(saml_env)
    client = TestClient(app)
    saml_response = saml_env.mint_response()

    acs = client.post(
        "/auth/saml/acs",
        data={"SAMLResponse": saml_response},
        follow_redirects=False,
    )
    assert acs.status_code == 303
    assert "saml_session" in acs.cookies

    # user was provisioned into the ProvisioningStore via the shared upsert seam
    users = list(store._users.values())
    assert len(users) == 1
    assert users[0].external_id == "alice@example.com"
    assert users[0].emails[0].value == "alice@example.com"


def test_session_cookie_authenticates_with_saml_method(saml_env):
    app, _ = _build_app(saml_env)
    client = TestClient(app)
    acs = client.post(
        "/auth/saml/acs",
        data={"SAMLResponse": saml_env.mint_response()},
        follow_redirects=False,
    )
    client.cookies.set("saml_session", acs.cookies["saml_session"])

    whoami = client.get("/whoami")
    assert whoami.status_code == 200
    body = whoami.json()
    assert body["auth_method"] == "saml"
    assert body["subject"] == "alice@example.com"
    assert body["email"] == "alice@example.com"


def test_acs_rejects_tampered_assertion(saml_env):
    app, store = _build_app(saml_env)
    client = TestClient(app)
    valid = saml_env.mint_response()
    decoded = base64.b64decode(valid).decode()
    tampered = decoded.replace("alice@example.com", "attacker@evil.com")
    tampered_b64 = base64.b64encode(tampered.encode()).decode()

    response = client.post(
        "/auth/saml/acs",
        data={"SAMLResponse": tampered_b64},
        follow_redirects=False,
    )
    assert response.status_code == 401
    assert store._users == {}


def test_acs_rejects_unsigned_assertion(saml_env):
    app, store = _build_app(saml_env)
    client = TestClient(app)
    unsigned = saml_env.mint_response(sign_assertion=False)
    response = client.post(
        "/auth/saml/acs",
        data={"SAMLResponse": unsigned},
        follow_redirects=False,
    )
    assert response.status_code == 401
    assert store._users == {}


def test_acs_missing_response_is_rejected(saml_env):
    app, _ = _build_app(saml_env)
    client = TestClient(app)
    response = client.post("/auth/saml/acs", data={}, follow_redirects=False)
    assert response.status_code == 400


def test_acs_rejects_garbage_response(saml_env):
    app, store = _build_app(saml_env)
    client = TestClient(app)
    response = client.post(
        "/auth/saml/acs",
        data={"SAMLResponse": "this-is-not-a-saml-response"},
        follow_redirects=False,
    )
    assert response.status_code == 401
    assert store._users == {}


def test_acs_ignores_untrusted_form_relay_state(saml_env):
    # the redirect target is bound server-side to the originating AuthnRequest, so a
    # client-supplied form RelayState on an (unsolicited) response is NOT trusted and
    # the ACS falls back to default_redirect_path
    app, _ = _build_app(saml_env)
    client = TestClient(app)
    acs = client.post(
        "/auth/saml/acs",
        data={"SAMLResponse": saml_env.mint_response(), "RelayState": "/dashboard"},
        follow_redirects=False,
    )
    assert acs.status_code == 303
    assert acs.headers["location"] == "/"


def test_acs_never_redirects_to_attacker_relay_state(saml_env):
    app, _ = _build_app(saml_env)
    client = TestClient(app)
    acs = client.post(
        "/auth/saml/acs",
        data={
            "SAMLResponse": saml_env.mint_response(),
            "RelayState": "https://evil.example.com/phish",
        },
        follow_redirects=False,
    )
    assert acs.status_code == 303
    assert "evil.example.com" not in acs.headers["location"]
    assert acs.headers["location"] == "/"


def test_login_threads_safe_next_as_relay_state(saml_env):
    app, _ = _build_app(saml_env)
    client = TestClient(app)
    response = client.get("/auth/saml/login?next=/dashboard", follow_redirects=False)
    assert response.status_code == 303
    assert "RelayState=%2Fdashboard" in response.headers["location"]


def test_login_rejects_open_redirect_next(saml_env):
    app, _ = _build_app(saml_env)
    client = TestClient(app)
    response = client.get(
        "/auth/saml/login?next=https://evil.example.com", follow_redirects=False
    )
    assert response.status_code == 303
    location = response.headers["location"]
    assert "evil.example.com" not in location
    # falls back to default_redirect_path ("/") as the RelayState
    assert "RelayState=%2F&" in location or location.endswith("RelayState=%2F")


# --------------------------------------------------------------------------- #
# Pure helpers (no xmlsec1 required) - attribute mapping + open-redirect guard
# --------------------------------------------------------------------------- #


def test_map_attributes_applies_attribute_map():
    from litellm.proxy.auth_v2.config import DEFAULT_SAML_ATTRIBUTE_MAP
    from litellm.proxy.auth_v2.saml import _map_attributes

    ava = {
        "email": ["alice@example.com"],
        "givenName": ["Alice"],
        "surname": ["Anderson"],
        "groups": ["eng", "admins"],
    }
    mapped = _map_attributes(ava, dict(DEFAULT_SAML_ATTRIBUTE_MAP))
    assert mapped["email"] == "alice@example.com"
    assert mapped["given_name"] == "Alice"
    assert mapped["family_name"] == "Anderson"
    assert mapped["groups"] == ["eng", "admins"]


def test_user_from_mapped_builds_name_and_email():
    from litellm.proxy.auth_v2.saml import _user_from_mapped

    user = _user_from_mapped(
        "alice@example.com",
        {
            "given_name": "Alice",
            "family_name": "Anderson",
            "email": "alice@example.com",
        },
    )
    assert user.external_id == "alice@example.com"
    assert user.display_name == "Alice Anderson"
    assert user.emails[0].value == "alice@example.com"
    assert user.name.given_name == "Alice"


@pytest.mark.parametrize(
    "candidate,expected",
    [
        ("/dashboard", "/dashboard"),
        ("//evil.com", "/"),
        ("https://evil.com", "/"),
        ("/path\\with-backslash", "/"),
        (None, "/"),
    ],
)
def test_safe_relay_state_blocks_open_redirects(candidate, expected):
    from litellm.proxy.auth_v2.saml import _safe_relay_state

    assert _safe_relay_state(candidate, "/") == expected


@pytest.mark.parametrize(
    "metadata,expected_key",
    [
        ("<EntityDescriptor/>", "inline"),
        ("https://idp.example.com/metadata", "remote"),
        ("/etc/saml/idp.xml", "local"),
    ],
)
def test_metadata_source_classifies_input(metadata, expected_key):
    from litellm.proxy.auth_v2.saml import _metadata_source

    assert expected_key in _metadata_source(metadata)


def test_acs_session_cookie_is_secure(saml_env):
    app, _ = _build_app(saml_env)
    client = TestClient(app)
    acs = client.post(
        "/auth/saml/acs",
        data={"SAMLResponse": saml_env.mint_response()},
        follow_redirects=False,
    )
    assert "saml_session" in acs.cookies
    assert "secure" in acs.headers["set-cookie"].lower()


# --------------------------------------------------------------------------- #
# SamlSessionStore TTL + size eviction (no xmlsec1 needed)
# --------------------------------------------------------------------------- #


def test_session_store_expires_entries():
    from litellm.proxy.auth_v2.saml import SamlSessionStore

    store = SamlSessionStore(ttl_seconds=0)
    session_id = store.create_session({"name_id": "alice@example.com"})
    # ttl of 0 means the entry is already past its expiry on the next read
    assert store.get(session_id) is None


def test_session_store_evicts_when_over_capacity():
    from litellm.proxy.auth_v2.saml import SamlSessionStore

    store = SamlSessionStore(max_size=3)
    ids = [store.create_session({"name_id": f"user-{i}"}) for i in range(5)]
    live = [sid for sid in ids if store.get(sid) is not None]
    assert len(live) <= 3
