"""Tests for the v1 -> v2 bridge.

`to_server_spec` maps the migrated modes (none + the static-header family, shared-key) and
defers everything else to v1 by returning None; `to_subject` maps the principal; `raise_public`
maps each CredError onto its HTTP status. These pin the parity-critical mapping before the graft.
"""

import base64
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi import HTTPException

from litellm.proxy._experimental.mcp_server.outbound_credentials.adapter import (
    raise_public,
    raise_user_oauth_challenge,
    to_server_spec,
    to_subject,
)
from litellm.proxy._experimental.mcp_server.outbound_credentials.types import (
    ApiKeyConfig,
    AuthorizationCodeConfig,
    ClientSecretAuth,
    CredError,
    IdJagConfig,
    NoneConfig,
    PrivateKeyJwtAuth,
    SharedKey,
)
from litellm.types.mcp import MCPAuth, MCPTransport
from litellm.types.mcp_server.mcp_server_manager import MCPServer


def _server(**kwargs) -> MCPServer:
    return MCPServer(server_id="s", name="n", transport=MCPTransport.http, **kwargs)


def _id_jag_server(**overrides) -> MCPServer:
    defaults = dict(
        auth_type=MCPAuth.oauth2_id_jag,
        url="https://mcp.example.com/mcp",
        client_id="litellm-client-id",
        client_secret="litellm-client-secret",
        token_exchange_endpoint="https://idp.example.com/token",
        id_jag_resource_token_endpoint="https://mcp-as.example.com/token",
        audience="api://mcp-server",
        scopes=["mcp.read", "mcp.write"],
    )
    defaults.update(overrides)
    return _server(**defaults)


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
    spec = to_server_spec(
        _server(auth_type=MCPAuth.basic, authentication_token="user:pass")
    )
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
        _server(
            auth_type=MCPAuth.oauth2, oauth2_flow="client_credentials"
        ),  # M2M -> v1
        _server(
            auth_type=MCPAuth.oauth2, delegate_auth_to_upstream=True
        ),  # delegated upstream OAuth -> v1
        _server(auth_type=MCPAuth.oauth2_token_exchange),
        _server(auth_type=MCPAuth.aws_sigv4),
        _server(
            auth_type=None, oauth_passthrough=True, extra_headers=["Authorization"]
        ),
    ],
)
def test_unmigrated_modes_defer_to_v1(server):
    # A None spec is the defer signal; the caller falls back to v1.
    assert to_server_spec(server) is None


@pytest.mark.parametrize(
    "server",
    [
        _server(auth_type=MCPAuth.api_key, is_byok=True),
        # BYOK rides on auth_type, so it must defer for every scheme, not just api_key. A stray
        # static token must not route a BYOK server to a v2 shared-key spec with the wrong value.
        _server(auth_type=MCPAuth.bearer_token, is_byok=True, authentication_token="x"),
        _server(auth_type=MCPAuth.basic, is_byok=True, authentication_token="x"),
        _server(
            auth_type=MCPAuth.authorization, is_byok=True, authentication_token="x"
        ),
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
    error = CredError.of_unauthorized(
        "needs key", www_authenticate='Bearer resource_metadata="/x"', body=body
    )
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


_ROOT_PATH = "litellm.proxy.utils.get_server_root_path"


def test_raise_user_oauth_challenge_points_at_per_server_prm():
    with patch(_ROOT_PATH, return_value="/"), pytest.raises(HTTPException) as exc_info:
        raise_user_oauth_challenge(_server(alias="my-srv"))
    exc = exc_info.value
    assert exc.status_code == 401
    assert (
        exc.headers["WWW-Authenticate"]
        == 'Bearer resource_metadata="/.well-known/oauth-protected-resource/mcp/my-srv"'
    )


def test_raise_user_oauth_challenge_includes_server_root_path():
    with (
        patch(_ROOT_PATH, return_value="/api/v1"),
        pytest.raises(HTTPException) as exc_info,
    ):
        raise_user_oauth_challenge(_server(alias="my-srv"))
    assert (
        exc_info.value.headers["WWW-Authenticate"]
        == 'Bearer resource_metadata="/.well-known/oauth-protected-resource/api/v1/mcp/my-srv"'
    )


@pytest.mark.parametrize(
    "kwargs, expected_name",
    [
        ({"alias": "a", "server_name": "sn"}, "a"),  # alias wins
        ({"server_name": "sn"}, "sn"),  # then server_name
        ({}, "n"),  # then the name field (server_id is the last fallback)
    ],
)
def test_raise_user_oauth_challenge_name_fallback(kwargs, expected_name):
    with patch(_ROOT_PATH, return_value="/"), pytest.raises(HTTPException) as exc_info:
        raise_user_oauth_challenge(_server(**kwargs))
    assert f'/mcp/{expected_name}"' in exc_info.value.headers["WWW-Authenticate"]


def test_id_jag_client_secret_maps_to_config():
    spec = to_server_spec(_id_jag_server())
    assert spec is not None and isinstance(spec.config, IdJagConfig)
    assert spec.config.org_token_endpoint == "https://idp.example.com/token"
    assert spec.config.resource_token_endpoint == "https://mcp-as.example.com/token"
    assert spec.config.client_id == "litellm-client-id"
    assert spec.config.audience == "api://mcp-server"
    assert spec.config.scopes == ("mcp.read", "mcp.write")
    # ID-JAG asserts the user's id_token; the access_token default maps to id_token.
    assert spec.config.subject_token_type == "urn:ietf:params:oauth:token-type:id_token"
    assert isinstance(spec.config.client_auth, ClientSecretAuth)
    assert spec.config.client_auth.client_secret.get_secret_value() == (
        "litellm-client-secret"
    )


def test_id_jag_private_key_maps_to_private_key_jwt_auth():
    spec = to_server_spec(
        _id_jag_server(
            client_secret=None,
            client_private_key="PEM-DATA",
            client_private_key_id="kid-1",
            client_assertion_signing_alg="RS384",
        )
    )
    assert spec is not None and isinstance(spec.config, IdJagConfig)
    assert isinstance(spec.config.client_auth, PrivateKeyJwtAuth)
    assert spec.config.client_auth.private_key.get_secret_value() == "PEM-DATA"
    assert spec.config.client_auth.key_id == "kid-1"
    assert spec.config.client_auth.signing_alg == "RS384"


def test_id_jag_private_key_wins_over_client_secret():
    spec = to_server_spec(_id_jag_server(client_private_key="PEM-DATA"))
    assert spec is not None and isinstance(spec.config, IdJagConfig)
    assert isinstance(spec.config.client_auth, PrivateKeyJwtAuth)


def test_id_jag_honors_explicit_subject_token_type():
    spec = to_server_spec(
        _id_jag_server(subject_token_type="urn:ietf:params:oauth:token-type:saml2")
    )
    assert spec is not None and isinstance(spec.config, IdJagConfig)
    assert spec.config.subject_token_type == "urn:ietf:params:oauth:token-type:saml2"


@pytest.mark.parametrize(
    "server",
    [
        _id_jag_server(token_exchange_endpoint=None),
        _id_jag_server(id_jag_resource_token_endpoint=None),
        _id_jag_server(client_id=None),
        _id_jag_server(client_secret=None, client_private_key=None),
    ],
)
def test_id_jag_half_configured_defers_to_v1(server):
    # A half-configured server must defer (None) rather than 500 at IdJagConfig construction.
    assert to_server_spec(server) is None
