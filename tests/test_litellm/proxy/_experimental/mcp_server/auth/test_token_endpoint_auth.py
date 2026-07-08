"""Tests for token-endpoint client authentication (client_secret_basic vs client_secret_post)."""

import base64

import pytest

from litellm.proxy._experimental.mcp_server.auth.token_endpoint_auth import (
    TokenEndpointAuthConfigError,
    build_token_endpoint_client_auth,
    normalize_token_endpoint_auth_method,
)


def _expected_basic(client_id: str, client_secret: str) -> str:
    return "Basic " + base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()


def test_basic_puts_credentials_in_header_and_not_body():
    auth = build_token_endpoint_client_auth(auth_method="client_secret_basic", client_id="cid", client_secret="sec")
    assert auth.headers == {"Authorization": _expected_basic("cid", "sec")}
    assert "client_secret" not in auth.body
    assert auth.body == {}


def test_basic_form_urlencodes_reserved_characters():
    """RFC 6749 2.3.1: client_id and client_secret are form-urlencoded before the ':' join, so reserved
    characters survive base64 transport instead of corrupting the username/password split."""
    auth = build_token_endpoint_client_auth(
        auth_method="client_secret_basic", client_id="client:one", client_secret="sec+ret:two"
    )
    decoded = base64.b64decode(auth.headers["Authorization"].removeprefix("Basic ")).decode()
    assert decoded == "client%3Aone:sec%2Bret%3Atwo"


def test_post_default_puts_credentials_in_body_and_no_auth_header():
    auth = build_token_endpoint_client_auth(auth_method="client_secret_post", client_id="cid", client_secret="sec")
    assert auth.headers == {}
    assert auth.body == {"client_id": "cid", "client_secret": "sec"}


def test_none_method_defaults_to_post():
    auth = build_token_endpoint_client_auth(auth_method=None, client_id="cid", client_secret="sec")
    assert auth.headers == {}
    assert auth.body == {"client_id": "cid", "client_secret": "sec"}


def test_explicit_basic_without_secret_raises():
    """client_secret_basic is a confidential-client method; a missing secret is a misconfiguration
    that must surface, not silently downgrade to a body request (RFC 6749; the no-silent-fallback rule)."""
    with pytest.raises(TokenEndpointAuthConfigError):
        build_token_endpoint_client_auth(auth_method="client_secret_basic", client_id="cid", client_secret=None)


def test_explicit_basic_without_client_id_raises():
    with pytest.raises(TokenEndpointAuthConfigError):
        build_token_endpoint_client_auth(auth_method="client_secret_basic", client_id=None, client_secret="sec")


def test_default_method_without_secret_is_public_client_post():
    """A secretless client_id under the default method is the legitimate public-client / PKCE case:
    client_id goes in the body, no secret, no error."""
    auth = build_token_endpoint_client_auth(auth_method=None, client_id="cid", client_secret=None)
    assert auth.headers == {}
    assert auth.body == {"client_id": "cid"}


def test_explicit_post_without_secret_does_not_raise():
    """Unlike basic, explicit client_secret_post degrades to a valid public-client request, so it
    does not error on a missing secret."""
    auth = build_token_endpoint_client_auth(auth_method="client_secret_post", client_id="cid", client_secret=None)
    assert auth.headers == {}
    assert auth.body == {"client_id": "cid"}


def test_normalize_only_accepts_known_methods():
    assert normalize_token_endpoint_auth_method("client_secret_basic") == "client_secret_basic"
    assert normalize_token_endpoint_auth_method("client_secret_post") == "client_secret_post"
    assert normalize_token_endpoint_auth_method("private_key_jwt") is None
    assert normalize_token_endpoint_auth_method(None) is None
    assert normalize_token_endpoint_auth_method(123) is None
