"""Tests for the v1 -> v2 bridge.

`to_server_spec` maps the migrated modes (none + the static-header family, shared-key) and
defers everything else to v1 by returning None; `to_subject` maps the principal; `raise_public`
maps each CredError onto its HTTP status. These pin the parity-critical mapping before the graft.
"""

import base64
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from litellm.proxy._experimental.mcp_server.outbound_credentials.adapter import (
    oauth_protected_resource_path,
    raise_public,
    raise_user_oauth_challenge,
    to_server_spec,
    to_subject,
)
from litellm.proxy._experimental.mcp_server.outbound_credentials.types import (
    ApiKeyConfig,
    AuthorizationCodeConfig,
    CredError,
    NoneConfig,
    SharedKey,
    TokenExchangeConfig,
)
from litellm.types.mcp import MCPAuth, MCPTransport
from litellm.types.mcp_server.mcp_server_manager import MCPServer


def _server(**kwargs) -> MCPServer:
    return MCPServer(server_id="s", name="n", transport=MCPTransport.http, **kwargs)


def test_none_maps_to_none_config():
    spec = to_server_spec(_server(auth_type=None))
    assert spec is not None
    assert isinstance(spec.config, NoneConfig)


def test_api_key_maps_to_x_api_key_shared():
    spec = to_server_spec(_server(auth_type=MCPAuth.api_key, authentication_token="k"))
    assert spec is not None and isinstance(spec.config, ApiKeyConfig)
    assert spec.config.header_name == "X-API-Key"
    assert spec.config.value_prefix == ""
    assert isinstance(spec.config.key_source, SharedKey)
    assert spec.config.key_source.value.get_secret_value() == "k"


@pytest.mark.parametrize(
    "auth_type, prefix",
    [
        (MCPAuth.bearer_token, "Bearer"),
        (MCPAuth.token, "token"),
        (MCPAuth.authorization, ""),
    ],
)
def test_authorization_schemes_map_with_their_prefix(auth_type, prefix):
    spec = to_server_spec(_server(auth_type=auth_type, authentication_token="t"))
    assert spec is not None and isinstance(spec.config, ApiKeyConfig)
    assert spec.config.header_name == "Authorization"
    assert spec.config.value_prefix == prefix
    assert spec.config.key_source.value.get_secret_value() == "t"


def test_basic_scheme_base64_encodes_the_token():
    spec = to_server_spec(_server(auth_type=MCPAuth.basic, authentication_token="user:pass"))
    assert spec is not None and isinstance(spec.config, ApiKeyConfig)
    assert spec.config.value_prefix == "Basic"
    expected = base64.b64encode(b"user:pass").decode()
    assert spec.config.key_source.value.get_secret_value() == expected


@pytest.mark.parametrize(
    "oauth2_flow",
    [None, "authorization_code"],
)
def test_oauth2_user_token_maps_to_authorization_code(oauth2_flow):
    # oauth2 without client_credentials is the per-user authorization_code mode.
    spec = to_server_spec(_server(auth_type=MCPAuth.oauth2, oauth2_flow=oauth2_flow))
    assert spec is not None and isinstance(spec.config, AuthorizationCodeConfig)


@pytest.mark.parametrize(
    "server",
    [
        _server(auth_type=MCPAuth.api_key),  # no token configured
        _server(auth_type=MCPAuth.bearer_token),  # no token configured
        _server(auth_type=MCPAuth.oauth2, oauth2_flow="client_credentials"),  # M2M -> v1
        _server(auth_type=MCPAuth.oauth2, delegate_auth_to_upstream=True),  # delegated upstream OAuth -> v1
        _server(auth_type=MCPAuth.oauth2_token_exchange),  # no endpoint/client creds -> incomplete -> v1
        _server(
            auth_type=MCPAuth.oauth2_token_exchange,
            token_exchange_endpoint="https://idp/token",
            client_id="cid",
        ),  # missing client_secret -> incomplete -> v1
        _server(auth_type=MCPAuth.aws_sigv4),
        _server(auth_type=None, oauth_passthrough=True, extra_headers=["Authorization"]),
    ],
)
def test_unmigrated_modes_defer_to_v1(server):
    # A None spec is the defer signal; the caller falls back to v1.
    assert to_server_spec(server) is None


def test_token_exchange_maps_full_config():
    spec = to_server_spec(
        _server(
            auth_type=MCPAuth.oauth2_token_exchange,
            url="https://up.example.com/mcp",
            token_exchange_endpoint="https://idp.example.com/token",
            audience="https://up.example.com",
            client_id="cid",
            client_secret="csec",
            subject_token_type="urn:ietf:params:oauth:token-type:jwt",
            scopes=["a", "b"],
        )
    )
    assert spec is not None
    config = spec.config
    assert isinstance(config, TokenExchangeConfig)
    assert config.token_exchange_endpoint == "https://idp.example.com/token"
    assert config.audience == "https://up.example.com"
    assert config.client_id == "cid"
    assert config.client_secret is not None
    assert config.client_secret.get_secret_value() == "csec"
    assert config.subject_token_type == "urn:ietf:params:oauth:token-type:jwt"
    assert config.scopes == ("a", "b")


def test_token_exchange_falls_back_to_token_url_when_no_exchange_endpoint():
    spec = to_server_spec(
        _server(
            auth_type=MCPAuth.oauth2_token_exchange,
            token_url="https://idp.example.com/token",
            client_id="cid",
            client_secret="csec",
        )
    )
    assert spec is not None
    assert isinstance(spec.config, TokenExchangeConfig)
    assert spec.config.token_exchange_endpoint == "https://idp.example.com/token"


def test_token_exchange_with_creds_but_no_endpoint_is_owned_for_fail_closed():
    # An OBO server with client credentials but no endpoint is still owned by v2 (spec, not None) so
    # it fails closed at the exchanger (412) rather than silently deferring to v1 and connecting
    # unauthenticated. The endpoint stays None for the exchanger to reject.
    spec = to_server_spec(
        _server(
            auth_type=MCPAuth.oauth2_token_exchange,
            client_id="cid",
            client_secret="csec",
        )
    )
    assert spec is not None
    assert isinstance(spec.config, TokenExchangeConfig)
    assert spec.config.token_exchange_endpoint is None


def test_token_exchange_maps_entra_obo_profile():
    spec = to_server_spec(
        _server(
            auth_type=MCPAuth.oauth2_token_exchange,
            token_exchange_endpoint="https://login.microsoftonline.com/tid/oauth2/v2.0/token",
            client_id="cid",
            client_secret="csec",
            token_exchange_profile="entra_obo",
            scopes=["api://target/.default"],
        )
    )
    assert spec is not None and isinstance(spec.config, TokenExchangeConfig)
    assert spec.config.profile == "entra_obo"


def test_token_exchange_defaults_to_rfc8693_profile():
    spec = to_server_spec(
        _server(
            auth_type=MCPAuth.oauth2_token_exchange,
            token_exchange_endpoint="https://idp/token",
            client_id="cid",
            client_secret="csec",
        )
    )
    assert spec is not None and isinstance(spec.config, TokenExchangeConfig)
    assert spec.config.profile == "rfc8693"


@pytest.mark.parametrize("bogus", ["", "RFC8693", "jwt_bearer", "entra", "unknown"])
def test_token_exchange_unknown_profile_normalizes_to_rfc8693(bogus):
    # A bad DB/config value must normalize to rfc8693, not raise a ValidationError building the spec.
    spec = to_server_spec(
        _server(
            auth_type=MCPAuth.oauth2_token_exchange,
            token_exchange_endpoint="https://idp/token",
            client_id="cid",
            client_secret="csec",
            token_exchange_profile=bogus,
        )
    )
    assert spec is not None and isinstance(spec.config, TokenExchangeConfig)
    assert spec.config.profile == "rfc8693"


def test_token_exchange_omits_audience_when_unset():
    spec = to_server_spec(
        _server(
            auth_type=MCPAuth.oauth2_token_exchange,
            token_exchange_endpoint="https://idp.example.com/token",
            client_id="cid",
            client_secret="csec",
        )
    )
    assert spec is not None
    assert isinstance(spec.config, TokenExchangeConfig)
    assert spec.config.audience is None


def test_token_exchange_empty_subject_token_type_normalizes_to_default():
    # Parity with v1: a falsy subject_token_type must not be sent verbatim to the IdP.
    spec = to_server_spec(
        _server(
            auth_type=MCPAuth.oauth2_token_exchange,
            token_exchange_endpoint="https://idp.example.com/token",
            client_id="cid",
            client_secret="csec",
            subject_token_type="",
        )
    )
    assert spec is not None
    assert isinstance(spec.config, TokenExchangeConfig)
    assert spec.config.subject_token_type == "urn:ietf:params:oauth:token-type:access_token"


@pytest.mark.parametrize(
    "server",
    [
        _server(auth_type=MCPAuth.api_key, is_byok=True),
        # BYOK rides on auth_type, so it must defer for every scheme, not just api_key. A stray
        # static token must not route a BYOK server to a v2 shared-key spec with the wrong value.
        _server(auth_type=MCPAuth.bearer_token, is_byok=True, authentication_token="x"),
        _server(auth_type=MCPAuth.basic, is_byok=True, authentication_token="x"),
        _server(auth_type=MCPAuth.authorization, is_byok=True, authentication_token="x"),
        _server(auth_type=MCPAuth.token, is_byok=True, authentication_token="x"),
        _server(auth_type=None, is_byok=True),
    ],
)
def test_byok_defers_regardless_of_auth_type(server):
    assert to_server_spec(server) is None


def test_to_subject_unauthenticated_is_empty_with_inbound_token():
    subject = to_subject(None, "inbound-jwt")
    assert subject.tenant_id == ""
    assert subject.subject_id == ""
    assert subject.inbound_token is not None
    assert subject.inbound_token.get_secret_value() == "inbound-jwt"


def test_to_subject_maps_principal_fields():
    principal = SimpleNamespace(org_id="org1", team_id="team1", user_id="user1")
    subject = to_subject(principal, None)
    assert subject.tenant_id == "org1"
    assert subject.subject_id == "user1"
    assert subject.inbound_token is None


@pytest.mark.parametrize(
    "error, status",
    [
        (CredError.of_unauthorized("x"), 401),
        (CredError.of_misconfigured("x"), 500),
        (CredError.of_upstream_unavailable("x"), 503),
        (CredError.of_unsupported_mode("x"), 500),
        (CredError.of_precondition_required("x"), 412),
        (CredError.of_not_implemented("x"), 501),
    ],
)
def test_raise_public_maps_each_error_to_its_status(error, status):
    with pytest.raises(HTTPException) as exc_info:
        raise_public(error)
    assert exc_info.value.status_code == status


def test_raise_public_emits_unauthorized_challenge():
    body = {"error": "byok_auth_required", "server_id": "s1"}
    error = CredError.of_unauthorized("needs key", www_authenticate='Bearer resource_metadata="/x"', body=body)
    with pytest.raises(HTTPException) as exc_info:
        raise_public(error)
    exc = exc_info.value
    assert exc.status_code == 401
    assert exc.detail == body
    assert exc.headers is not None
    assert exc.headers["WWW-Authenticate"] == 'Bearer resource_metadata="/x"'


def test_raise_public_plain_unauthorized_has_no_challenge():
    with pytest.raises(HTTPException) as exc_info:
        raise_public(CredError.of_unauthorized("nope"))
    exc = exc_info.value
    assert exc.status_code == 401
    assert exc.detail == "unauthorized: nope"
    assert exc.headers is None


@pytest.mark.parametrize(
    "root_path, expected_prefix",
    [
        ("/", ""),  # "/" means no prefix
        ("", ""),  # empty means no prefix
        ("/api/v1", "/api/v1"),  # a real root path is prepended verbatim
    ],
)
def test_oauth_protected_resource_path_honors_root_path(root_path, expected_prefix):
    path = oauth_protected_resource_path(root_path, _server(alias="my-srv"))
    assert path == f"/.well-known/oauth-protected-resource{expected_prefix}/mcp/my-srv"


@pytest.mark.parametrize(
    "kwargs, expected_name",
    [
        ({"alias": "a", "server_name": "sn"}, "a"),  # alias wins
        ({"server_name": "sn"}, "sn"),  # then server_name
        ({}, "n"),  # then the name field (server_id is the last fallback)
    ],
)
def test_oauth_protected_resource_path_name_fallback(kwargs, expected_name):
    assert oauth_protected_resource_path("/", _server(**kwargs)).endswith(f"/mcp/{expected_name}")


def test_raise_user_oauth_challenge_points_at_per_server_prm():
    with pytest.raises(HTTPException) as exc_info:
        raise_user_oauth_challenge(_server(alias="my-srv"), root_path="/")
    exc = exc_info.value
    assert exc.status_code == 401
    assert (
        exc.headers["WWW-Authenticate"] == 'Bearer resource_metadata="/.well-known/oauth-protected-resource/mcp/my-srv"'
    )


def test_raise_user_oauth_challenge_includes_server_root_path():
    with pytest.raises(HTTPException) as exc_info:
        raise_user_oauth_challenge(_server(alias="my-srv"), root_path="/api/v1")
    assert (
        exc_info.value.headers["WWW-Authenticate"]
        == 'Bearer resource_metadata="/.well-known/oauth-protected-resource/api/v1/mcp/my-srv"'
    )


def test_raise_token_exchange_challenge_is_rfc9728_invalid_token():
    from litellm.proxy._experimental.mcp_server.outbound_credentials.adapter import (
        raise_token_exchange_challenge,
    )

    with pytest.raises(HTTPException) as exc_info:
        raise_token_exchange_challenge(_server(alias="obo-srv"), root_path="/")
    exc = exc_info.value
    www = exc.headers["WWW-Authenticate"]
    assert exc.status_code == 401
    # RFC 9728 resource_metadata so the client can discover the IdP, plus RFC 6750 invalid_token.
    assert 'resource_metadata="/.well-known/oauth-protected-resource/mcp/obo-srv"' in www
    assert 'error="invalid_token"' in www
    assert "error_description=" in www


def test_raise_token_exchange_challenge_includes_server_root_path():
    from litellm.proxy._experimental.mcp_server.outbound_credentials.adapter import (
        raise_token_exchange_challenge,
    )

    with pytest.raises(HTTPException) as exc_info:
        raise_token_exchange_challenge(_server(alias="obo-srv"), root_path="/api/v1")
    www = exc_info.value.headers["WWW-Authenticate"]
    assert 'resource_metadata="/.well-known/oauth-protected-resource/api/v1/mcp/obo-srv"' in www
