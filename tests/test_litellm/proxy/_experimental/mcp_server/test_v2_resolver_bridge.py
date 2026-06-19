"""Parity tests for the v1->v2 MCP resolver graft (none + api_key).

When the flag is on, resolve_mcp_auth routes none/api_key through the v2 resolver; the upstream
headers must be byte-identical to what v1 produces, and every other mode (or any v2 error) must
fall back to v1 unchanged.
"""

import httpx
import pytest

from litellm.experimental_mcp_client.client import MCPClient
from litellm.proxy._experimental.mcp_server.oauth2_token_cache import resolve_mcp_auth
from litellm.proxy._experimental.mcp_server.v2_resolver_bridge import (
    _to_subject,
    resolve_v2_auth_value,
    resolve_v2_aws_auth,
)
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.mcp import MCPAuth, MCPTransport
from litellm.types.mcp_server.mcp_server_manager import MCPServer

pytestmark = pytest.mark.asyncio

FLAG = "LITELLM_USE_V2_MCP_RESOLVER"


def _server(auth_type, token=None):
    return MCPServer(
        server_id="s1",
        name="s1",
        transport=MCPTransport.http,
        url="https://up.example/mcp",
        auth_type=auth_type,
        authentication_token=token,
    )


def _v1_headers(auth_type, auth_value):
    """The final upstream headers v1's MCPClient would send for this resolved value."""
    return MCPClient(auth_type=auth_type, auth_value=auth_value)._get_auth_headers()


@pytest.fixture
def v2_on(monkeypatch):
    monkeypatch.setenv(FLAG, "true")


@pytest.fixture
def v2_off(monkeypatch):
    monkeypatch.delenv(FLAG, raising=False)


async def test_flag_off_defers_to_v1(v2_off):
    assert await resolve_v2_auth_value(_server(MCPAuth.api_key, "k")) is None


async def test_api_key_parity(v2_on):
    token = "up-secret"
    server = _server(MCPAuth.api_key, token)
    v2_value = await resolve_v2_auth_value(server)
    assert v2_value == {"X-API-Key": token}
    # byte-identical to v1's final upstream headers
    assert _v1_headers(MCPAuth.api_key, token) == _v1_headers(MCPAuth.api_key, v2_value)


async def test_none_attaches_no_auth(v2_on):
    server = _server(MCPAuth.none)
    v2_value = await resolve_v2_auth_value(server)
    assert v2_value == {}
    # v1 none -> no auth header; v2 -> empty dict merged -> no auth header
    assert _v1_headers(MCPAuth.none, None) == _v1_headers(MCPAuth.none, v2_value) == {}


async def test_non_grafted_mode_defers_to_v1(v2_on):
    # bearer_token is not grafted yet -> v2 returns None so v1 handles it
    assert await resolve_v2_auth_value(_server(MCPAuth.bearer_token, "k")) is None


async def test_api_key_without_token_defers_to_v1(v2_on):
    assert await resolve_v2_auth_value(_server(MCPAuth.api_key, None)) is None


async def test_resolve_mcp_auth_hook_routes_api_key_when_on(v2_on):
    server = _server(MCPAuth.api_key, "up-secret")
    assert await resolve_mcp_auth(server) == {"X-API-Key": "up-secret"}


async def test_resolve_mcp_auth_hook_uses_v1_when_off(v2_off):
    # v1 returns the raw token string; MCPClient then maps it to the X-API-Key header
    server = _server(MCPAuth.api_key, "up-secret")
    assert await resolve_mcp_auth(server) == "up-secret"


def _m2m_server():
    return MCPServer(
        server_id="m2m",
        name="m2m",
        transport=MCPTransport.http,
        url="https://up.example/mcp",
        auth_type=MCPAuth.oauth2,
        oauth2_flow="client_credentials",
        client_id="cid",
        client_secret="csecret",
        token_url="https://idp/token",
        scopes=["a", "b"],
    )


async def test_client_credentials_maps_to_config():
    from litellm.proxy._experimental.mcp_server.v2_resolver_bridge import (
        _to_server_spec,
    )
    from litellm.proxy.gateway.mcp.outbound_credentials.types import (
        ClientCredentialsConfig,
    )

    spec = _to_server_spec(_m2m_server())
    assert spec is not None
    assert isinstance(spec.config, ClientCredentialsConfig)
    assert spec.config.client_id == "cid"
    assert spec.config.token_url == "https://idp/token"
    assert spec.config.client_secret.get_secret_value() == "csecret"
    assert spec.config.scopes == ("a", "b")


async def test_client_credentials_graft_end_to_end(v2_on, monkeypatch):
    # M2M flows through the real fetcher; mock the IdP token endpoint and assert the Bearer.
    from litellm.proxy._experimental.mcp_server import v2_port_bodies

    class _Resp:
        status_code = 200

        def json(self):
            return {"access_token": "m2m-tok", "expires_in": 3600}

    class _Client:
        async def post(self, url, data=None):
            return _Resp()

    monkeypatch.setattr(
        v2_port_bodies, "get_async_httpx_client", lambda **kw: _Client()
    )
    assert await resolve_v2_auth_value(_m2m_server()) == {
        "Authorization": "Bearer m2m-tok"
    }


def _aws_server(**aws):
    return MCPServer(
        server_id="aws",
        name="aws",
        transport=MCPTransport.http,
        url="https://svc.us-east-1.amazonaws.com/mcp",
        auth_type=MCPAuth.aws_sigv4,
        **aws,
    )


async def test_aws_sigv4_flag_off_defers_to_v1(v2_off):
    server = _aws_server(aws_access_key_id="AKIA", aws_secret_access_key="s")
    assert await resolve_v2_aws_auth(server) is None


async def test_aws_sigv4_non_aws_server_returns_none(v2_on):
    assert await resolve_v2_aws_auth(_server(MCPAuth.api_key, "k")) is None


async def test_aws_sigv4_static_keys_returns_signing_auth(v2_on):
    server = _aws_server(
        aws_access_key_id="AKIATEST",
        aws_secret_access_key="secret",
        aws_region_name="us-east-1",
    )
    auth = await resolve_v2_aws_auth(server)
    assert auth is not None
    req = httpx.Request(
        "POST", "https://svc.us-east-1.amazonaws.com/mcp", content=b"{}"
    )
    signed = next(auth.auth_flow(req))
    assert signed.headers["Authorization"].startswith(
        "AWS4-HMAC-SHA256 Credential=AKIATEST/"
    )
    assert "X-Amz-Date" in signed.headers


async def test_aws_sigv4_config_maps_static_keys(v2_on):
    from litellm.proxy._experimental.mcp_server.v2_resolver_bridge import (
        _to_aws_sigv4_config,
    )
    from litellm.proxy.gateway.mcp.outbound_credentials.types import StaticKeys

    cfg = _to_aws_sigv4_config(
        _aws_server(
            aws_access_key_id="AKIA",
            aws_secret_access_key="s",
            aws_region_name="eu-west-1",
        )
    )
    assert cfg is not None
    assert cfg.region == "eu-west-1"
    assert isinstance(cfg.credentials, StaticKeys)
    assert cfg.credentials.access_key_id == "AKIA"


async def test_aws_sigv4_config_maps_assume_role(v2_on):
    from litellm.proxy._experimental.mcp_server.v2_resolver_bridge import (
        _to_aws_sigv4_config,
    )
    from litellm.proxy.gateway.mcp.outbound_credentials.types import AssumeRole

    cfg = _to_aws_sigv4_config(
        _aws_server(aws_role_name="arn:aws:iam::1:role/r", aws_session_name="sess")
    )
    assert cfg is not None
    assert isinstance(cfg.credentials, AssumeRole)
    assert cfg.credentials.role_arn == "arn:aws:iam::1:role/r"


async def test_aws_sigv4_config_role_with_base_keys_defers_to_v1(v2_on):
    from litellm.proxy._experimental.mcp_server.v2_resolver_bridge import (
        _to_aws_sigv4_config,
    )

    cfg = _to_aws_sigv4_config(
        _aws_server(
            aws_role_name="arn:aws:iam::1:role/r",
            aws_access_key_id="AKIA",
            aws_secret_access_key="s",
        )
    )
    assert cfg is None


async def test_aws_sigv4_config_defaults_to_ambient(v2_on):
    from litellm.proxy._experimental.mcp_server.v2_resolver_bridge import (
        _to_aws_sigv4_config,
    )
    from litellm.proxy.gateway.mcp.outbound_credentials.types import Ambient

    cfg = _to_aws_sigv4_config(_aws_server())
    assert cfg is not None
    assert isinstance(cfg.credentials, Ambient)


async def test_to_subject_maps_v1_identity():
    auth = UserAPIKeyAuth(token="sk-test", user_id="u1", org_id="org1", team_id="t1")
    subj = _to_subject(auth, "inbound-jwt")
    assert subj.subject_id == "u1"
    assert subj.tenant_id == "org1"  # org preferred over team
    assert subj.inbound_token is not None
    assert subj.inbound_token.get_secret_value() == "inbound-jwt"


async def test_to_subject_anonymous_when_no_auth():
    subj = _to_subject(None, None)
    assert subj.subject_id == ""
    assert subj.tenant_id == ""
    assert subj.inbound_token is None


async def test_to_subject_falls_back_to_team_and_blanks_missing_user():
    auth = UserAPIKeyAuth(token="sk-test", team_id="team-x")
    subj = _to_subject(auth, None)
    assert subj.tenant_id == "team-x"  # no org -> team
    assert (
        subj.subject_id == ""
    )  # missing user -> empty; per-user arms fail closed on this


async def test_resolve_v2_auth_value_threads_identity_without_breaking_static(v2_on):
    auth = UserAPIKeyAuth(token="sk-test", user_id="u1", org_id="org1")
    server = _server(MCPAuth.api_key, "up-secret")
    result = await resolve_v2_auth_value(
        server, user_api_key_auth=auth, subject_token="jwt"
    )
    assert result == {"X-API-Key": "up-secret"}


async def test_byok_server_maps_to_byok_key_source():
    from litellm.proxy._experimental.mcp_server.v2_resolver_bridge import (
        _to_server_spec,
    )
    from litellm.proxy.gateway.mcp.outbound_credentials.types import ApiKeyConfig, Byok

    server = MCPServer(
        server_id="byok1",
        name="byok1",
        transport=MCPTransport.http,
        url="https://up.example/mcp",
        auth_type=MCPAuth.api_key,
        is_byok=True,
    )
    spec = _to_server_spec(server)
    assert spec is not None
    assert isinstance(spec.config, ApiKeyConfig)
    assert isinstance(spec.config.key_source, Byok)
    assert spec.config.header_name == "X-API-Key"


async def test_token_exchange_server_maps_to_config():
    from litellm.proxy._experimental.mcp_server.v2_resolver_bridge import (
        _to_server_spec,
    )
    from litellm.proxy.gateway.mcp.outbound_credentials.types import TokenExchangeConfig

    server = MCPServer(
        server_id="obo",
        name="obo",
        transport=MCPTransport.http,
        url="https://up.example/mcp",
        auth_type=MCPAuth.oauth2_token_exchange,
        client_id="cid",
        client_secret="csecret",
        token_exchange_endpoint="https://idp/exchange",
        audience="https://aud.example",
        scopes=["a", "b"],
    )
    spec = _to_server_spec(server)
    assert spec is not None
    assert isinstance(spec.config, TokenExchangeConfig)
    assert spec.config.token_exchange_endpoint == "https://idp/exchange"
    assert spec.config.client_id == "cid"
    assert spec.config.client_secret is not None
    assert spec.config.client_secret.get_secret_value() == "csecret"
    assert spec.config.scopes == ("a", "b")
    assert spec.resource == "https://aud.example"  # audience preferred for the binding
