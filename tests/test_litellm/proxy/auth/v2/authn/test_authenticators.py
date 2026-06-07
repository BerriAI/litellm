import pytest

from litellm.proxy.auth.v2.authn import authenticators
from litellm.proxy.auth.v2.authn.authenticators import (
    OAuth2IntrospectionAuthenticator,
    JWTAuthenticator,
    MasterKeyAuthenticator,
    VirtualKeyAuthenticator,
    _jwks_provider,
)
from litellm.proxy.auth.v2.context import AuthMethod

JWT_SHAPED = "header.payload.signature"


def test_each_authenticator_advertises_its_method():
    # The method is recorded on the auth context for telemetry, so the chain can
    # tag how a request authenticated without re-deriving it.
    assert MasterKeyAuthenticator().method is AuthMethod.MASTER_KEY
    assert VirtualKeyAuthenticator().method is AuthMethod.VIRTUAL_KEY
    assert JWTAuthenticator().method is AuthMethod.JWT
    assert OAuth2IntrospectionAuthenticator().method is AuthMethod.OAUTH2


def test_jwks_provider_is_reused_per_uri():
    # Regression: a fresh provider per request makes the TTL cache dead and refetches
    # the JWKS every JWT auth. The same uri must return the same long-lived provider.
    authenticators._jwks_providers.clear()
    first = _jwks_provider("https://idp.example/.well-known/jwks.json")
    again = _jwks_provider("https://idp.example/.well-known/jwks.json")
    other = _jwks_provider("https://other.example/jwks")
    assert first is again
    assert first is not other


MASTER = "sk-master-secret-123"


@pytest.fixture
def master_key_set(monkeypatch):
    # proxy_server.master_key is a module global; set it for the exact-compare path.
    monkeypatch.setattr("litellm.proxy.proxy_server.master_key", MASTER, raising=False)


def test_master_key_matches_only_exact(master_key_set):
    auth = MasterKeyAuthenticator()
    assert auth.can_handle(MASTER) is True
    # A prefix / near-match must NOT authenticate as admin (constant-time exact compare).
    assert auth.can_handle(MASTER + "x") is False
    assert auth.can_handle("sk-master-secret-12") is False
    assert auth.can_handle("sk-something-else") is False


def test_master_key_rejects_non_strings_and_empty(master_key_set):
    auth = MasterKeyAuthenticator()
    assert auth.can_handle(None) is False
    assert auth.can_handle(b"bytes") is False
    assert auth.can_handle("") is False


def test_master_key_authenticator_inert_when_unconfigured(monkeypatch):
    monkeypatch.setattr("litellm.proxy.proxy_server.master_key", None, raising=False)
    assert MasterKeyAuthenticator().can_handle("sk-anything") is False


def test_virtual_key_handles_sk_prefix_but_not_master_first():
    vk = VirtualKeyAuthenticator()
    assert vk.can_handle("sk-abc123") is True
    assert vk.can_handle("not-a-key") is False
    assert vk.can_handle(None) is False


def test_jwt_authenticator_inert_when_unconfigured(monkeypatch):
    # No auth_v2_jwt config and no env var: a JWT-shaped token must fall through
    # (can_handle False) so the chain ends in a clean 401, not a 500.
    monkeypatch.setattr(
        "litellm.proxy.proxy_server.general_settings", {}, raising=False
    )
    monkeypatch.delenv("AUTH_V2_JWKS_URI", raising=False)
    assert JWTAuthenticator().can_handle(JWT_SHAPED) is False


def test_jwt_authenticator_handles_jwt_shape_when_configured(monkeypatch):
    monkeypatch.setattr(
        "litellm.proxy.proxy_server.general_settings",
        {"auth_v2_jwt": {"jwks_uri": "https://idp.example/jwks"}},
        raising=False,
    )
    auth = JWTAuthenticator()
    assert auth.can_handle(JWT_SHAPED) is True
    # Still only claims JWT-shaped tokens, never virtual keys or non-3-part values.
    assert auth.can_handle("sk-abc.def.ghi") is False
    assert auth.can_handle("not.a.jwt.token") is False
    assert auth.can_handle(None) is False
