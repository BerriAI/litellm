import pytest

from litellm.proxy.auth.v2.authenticators import (
    JWTAuthenticator,
    MasterKeyAuthenticator,
    VirtualKeyAuthenticator,
)

JWT_SHAPED = "header.payload.signature"

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
