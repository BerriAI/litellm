"""Tests for the resolver dispatch: live arms produce auth, stubbed arms fail closed.

`none`, `api_key` (shared-key source), and `authorization_code` are implemented; every other arm,
plus the `api_key` BYOK source, returns a typed `not_implemented` error until its mode lands.
Parametrizing the stubs over one config each also guards reachability: a dropped `case` would hit
`assert_never` and raise instead of returning the stub.
"""

import httpx
import pytest
from pydantic import SecretStr

from litellm.proxy._experimental.mcp_server.outbound_credentials import (
    ApiKeyConfig,
    AuthorizationCodeConfig,
    AwsSigV4Config,
    Byok,
    ClientCredentialsConfig,
    ClientSecretAuth,
    CredError,
    Error,
    IdJagConfig,
    NoneConfig,
    NoOpAuth,
    Ok,
    PassthroughConfig,
    Result,
    ServerSpec,
    SharedKey,
    StaticHeaderAuth,
    Subject,
    TokenExchangeConfig,
    UpstreamCredentialProvider,
)
from litellm.proxy._experimental.mcp_server.outbound_credentials.oauth_token_store import (
    OAuthToken,
    TokenStoreUnavailable,
)
from litellm.proxy._experimental.mcp_server.outbound_credentials.token_endpoint import (
    ExchangedToken,
)

_SUBJECT = Subject(tenant_id="", subject_id="")


def _id_jag_config() -> IdJagConfig:
    return IdJagConfig(
        org_token_endpoint="https://idp.example.com/token",
        resource_token_endpoint="https://mcp-as.example.com/token",
        client_id="litellm",
        client_auth=ClientSecretAuth(client_secret=SecretStr("s")),
        audience="api://mcp",
        scopes=("mcp.read",),
    )


class _FakeTokenEndpoint:
    """Records each fetch and returns the next canned Result, leg by leg."""

    def __init__(self, results: list[Result[ExchangedToken, CredError]]) -> None:
        self._results = list(results)
        self.calls: list[tuple[str, str, dict[str, str]]] = []

    async def fetch(self, endpoint, client_id, grant_params, client_auth):
        self.calls.append((endpoint, client_id, dict(grant_params)))
        return self._results.pop(0)


def _with_inbound(token: str) -> Subject:
    return Subject(tenant_id="", subject_id="alice", inbound_token=SecretStr(token))


def _spec(config):
    return ServerSpec(
        server_id="s", resource="https://upstream.example.com", config=config
    )


def _emitted(auth: httpx.Auth) -> httpx.Headers:
    request = httpx.Request("GET", "https://upstream.example.com/mcp")
    flow = auth.auth_flow(request)
    next(flow)
    flow.close()
    return request.headers


@pytest.mark.asyncio
async def test_none_mode_yields_a_no_op_auth():
    result = await UpstreamCredentialProvider().resolve_credentials(
        _SUBJECT, _spec(NoneConfig())
    )
    assert isinstance(result, Ok)
    assert isinstance(result.ok, NoOpAuth)


@pytest.mark.asyncio
async def test_api_key_shared_emits_the_configured_header():
    config = ApiKeyConfig(
        header_name="X-API-Key",
        value_prefix="",
        key_source=SharedKey(value=SecretStr("secret-key")),
    )
    result = await UpstreamCredentialProvider().resolve_credentials(
        _SUBJECT, _spec(config)
    )
    assert isinstance(result, Ok)
    assert isinstance(result.ok, StaticHeaderAuth)
    assert _emitted(result.ok)["X-API-Key"] == "secret-key"


@pytest.mark.asyncio
async def test_api_key_shared_honors_authorization_scheme():
    config = ApiKeyConfig(
        header_name="Authorization",
        value_prefix="Bearer",
        key_source=SharedKey(value=SecretStr("tok")),
    )
    result = await UpstreamCredentialProvider().resolve_credentials(
        _SUBJECT, _spec(config)
    )
    assert isinstance(result, Ok)
    assert _emitted(result.ok)["Authorization"] == "Bearer tok"


class _FakeTokenStore:
    """An OAuthTokenStore returning a canned per-user token (None == not authorized)."""

    def __init__(self, by_user: dict) -> None:
        self._by_user = by_user

    async def fetch(self, user_id: str, server_id: str):
        return self._by_user.get((user_id, server_id))


@pytest.mark.asyncio
async def test_authorization_code_emits_bearer_for_a_stored_token():
    store = _FakeTokenStore({("alice", "s"): OAuthToken(access_token="at-alice")})
    result = await UpstreamCredentialProvider(
        oauth_token_store=store
    ).resolve_credentials(
        Subject(tenant_id="", subject_id="alice"), _spec(AuthorizationCodeConfig())
    )
    assert isinstance(result, Ok)
    assert _emitted(result.ok)["Authorization"] == "Bearer at-alice"


@pytest.mark.asyncio
async def test_authorization_code_without_token_is_semantically_unauthorized():
    result = await UpstreamCredentialProvider(
        oauth_token_store=_FakeTokenStore({})
    ).resolve_credentials(
        Subject(tenant_id="", subject_id="alice"), _spec(AuthorizationCodeConfig())
    )
    assert isinstance(result, Error)
    assert result.error.tag == "unauthorized"
    # Semantic only: the per-server challenge is built at the edge, not in the request-free arm.
    assert "Authorization required" in result.error.unauthorized.detail
    assert result.error.unauthorized.www_authenticate is None
    assert result.error.unauthorized.body is None


@pytest.mark.asyncio
async def test_authorization_code_store_unavailable_is_unauthorized():
    class _Unavailable:
        async def fetch(self, user_id: str, server_id: str):
            raise TokenStoreUnavailable("down")

    result = await UpstreamCredentialProvider(
        oauth_token_store=_Unavailable()
    ).resolve_credentials(
        Subject(tenant_id="", subject_id="alice"), _spec(AuthorizationCodeConfig())
    )
    assert isinstance(result, Error)
    assert result.error.tag == "unauthorized"


@pytest.mark.asyncio
async def test_authorization_code_with_no_store_wired_is_unauthorized():
    result = await UpstreamCredentialProvider().resolve_credentials(
        Subject(tenant_id="", subject_id="alice"), _spec(AuthorizationCodeConfig())
    )
    assert isinstance(result, Error)
    assert result.error.tag == "unauthorized"


@pytest.mark.asyncio
async def test_authorization_code_isolates_by_subject():
    store = _FakeTokenStore({("alice", "s"): OAuthToken(access_token="at-alice")})
    provider = UpstreamCredentialProvider(oauth_token_store=store)
    alice = await provider.resolve_credentials(
        Subject(tenant_id="", subject_id="alice"), _spec(AuthorizationCodeConfig())
    )
    bob = await provider.resolve_credentials(
        Subject(tenant_id="", subject_id="bob"), _spec(AuthorizationCodeConfig())
    )
    assert (
        isinstance(alice, Ok)
        and _emitted(alice.ok)["Authorization"] == "Bearer at-alice"
    )
    assert isinstance(bob, Error) and bob.error.tag == "unauthorized"


@pytest.mark.asyncio
async def test_has_user_token_reflects_the_stored_token():
    present = UpstreamCredentialProvider(
        oauth_token_store=_FakeTokenStore(
            {("alice", "s"): OAuthToken(access_token="at")}
        )
    )
    absent = UpstreamCredentialProvider(oauth_token_store=_FakeTokenStore({}))
    spec = _spec(AuthorizationCodeConfig())
    subject = Subject(tenant_id="", subject_id="alice")
    assert await present.has_user_token(subject, spec) is True
    assert await absent.has_user_token(subject, spec) is False


@pytest.mark.asyncio
async def test_has_user_token_false_for_a_non_per_user_mode():
    # A none-mode server has no per-user token to check.
    provider = UpstreamCredentialProvider()
    spec = _spec(NoneConfig())
    assert (
        await provider.has_user_token(Subject(tenant_id="", subject_id="a"), spec)
        is False
    )


_STUBBED = [
    ("api_key_byok", ApiKeyConfig(key_source=Byok())),
    ("passthrough", PassthroughConfig()),
    ("client_credentials", ClientCredentialsConfig()),
    ("token_exchange", TokenExchangeConfig()),
    ("aws_sigv4", AwsSigV4Config(region="us-east-1")),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("label, config", _STUBBED)
async def test_unbuilt_arms_fail_closed_with_not_implemented(label, config):
    result = await UpstreamCredentialProvider().resolve_credentials(
        _SUBJECT, _spec(config)
    )
    assert isinstance(result, Error)
    assert result.error.tag == "not_implemented"


@pytest.mark.asyncio
async def test_id_jag_runs_both_legs_and_returns_the_leg2_bearer():
    endpoint = _FakeTokenEndpoint(
        [
            Ok(ExchangedToken(access_token="the-id-jag", expires_in=300)),
            Ok(ExchangedToken(access_token="final-access", expires_in=3600)),
        ]
    )
    provider = UpstreamCredentialProvider(token_endpoint=endpoint)
    result = await provider.resolve_credentials(
        _with_inbound("user-id-token"), _spec(_id_jag_config())
    )

    assert isinstance(result, Ok)
    assert _emitted(result.ok)["Authorization"] == "Bearer final-access"

    leg1_endpoint, _, leg1_params = endpoint.calls[0]
    assert leg1_endpoint == "https://idp.example.com/token"
    assert (
        leg1_params["grant_type"] == "urn:ietf:params:oauth:grant-type:token-exchange"
    )
    assert (
        leg1_params["requested_token_type"] == "urn:ietf:params:oauth:token-type:id-jag"
    )
    assert leg1_params["subject_token"] == "user-id-token"

    leg2_endpoint, _, leg2_params = endpoint.calls[1]
    assert leg2_endpoint == "https://mcp-as.example.com/token"
    assert leg2_params["grant_type"] == "urn:ietf:params:oauth:grant-type:jwt-bearer"
    # The leg-1 token is forwarded verbatim as the leg-2 assertion.
    assert leg2_params["assertion"] == "the-id-jag"


@pytest.mark.asyncio
async def test_id_jag_without_inbound_token_is_precondition_required_no_http():
    endpoint = _FakeTokenEndpoint([])
    provider = UpstreamCredentialProvider(token_endpoint=endpoint)
    result = await provider.resolve_credentials(
        Subject(tenant_id="", subject_id="alice"), _spec(_id_jag_config())
    )

    assert isinstance(result, Error)
    assert result.error.tag == "precondition_required"
    assert endpoint.calls == []


@pytest.mark.asyncio
async def test_id_jag_propagates_a_leg1_error_without_calling_leg2():
    endpoint = _FakeTokenEndpoint(
        [Error(CredError.of_upstream_unavailable("leg1 down"))]
    )
    provider = UpstreamCredentialProvider(token_endpoint=endpoint)
    result = await provider.resolve_credentials(
        _with_inbound("user-id-token"), _spec(_id_jag_config())
    )

    assert isinstance(result, Error)
    assert result.error.tag == "upstream_unavailable"
    assert "leg1 down" in result.error.summary
    assert len(endpoint.calls) == 1


@pytest.mark.asyncio
async def test_id_jag_propagates_a_leg2_error():
    endpoint = _FakeTokenEndpoint(
        [
            Ok(ExchangedToken(access_token="the-id-jag", expires_in=300)),
            Error(CredError.of_upstream_unavailable("leg2 forbidden")),
        ]
    )
    provider = UpstreamCredentialProvider(token_endpoint=endpoint)
    result = await provider.resolve_credentials(
        _with_inbound("user-id-token"), _spec(_id_jag_config())
    )

    assert isinstance(result, Error)
    assert result.error.tag == "upstream_unavailable"
    assert "leg2 forbidden" in result.error.summary
    assert len(endpoint.calls) == 2
