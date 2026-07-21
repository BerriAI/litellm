"""Tests for the resolver dispatch: live arms produce auth, stubbed arms fail closed.

`none`, `api_key` (shared-key source), `passthrough`, `authorization_code`, `token_exchange`, and
`client_credentials` are implemented; every other arm, plus the `api_key` BYOK source, returns a
typed `not_implemented` error until its mode lands. Parametrizing the stubs over one config each
also guards reachability: a dropped `case` would hit `assert_never` and raise instead of
returning the stub.
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
    """Records each fetch (and its rejection classifier) and returns the next canned Result."""

    def __init__(self, results: list[Result[ExchangedToken, CredError]]) -> None:
        self._results = list(results)
        self.calls: list[tuple[str, str, dict[str, str]]] = []
        self.classifiers: list[object] = []

    async def fetch(self, endpoint, client_id, grant_params, client_auth, classify_rejection=None):
        self.calls.append((endpoint, client_id, dict(grant_params)))
        self.classifiers.append(classify_rejection)
        return self._results.pop(0)


def _with_inbound(token: str) -> Subject:
    return Subject(tenant_id="", subject_id="alice", inbound_token=SecretStr(token))


def _spec(config):
    return ServerSpec(server_id="s", resource="https://upstream.example.com", config=config)


def _emitted(auth: httpx.Auth) -> httpx.Headers:
    request = httpx.Request("GET", "https://upstream.example.com/mcp")
    flow = auth.auth_flow(request)
    next(flow)
    flow.close()
    return request.headers


@pytest.mark.asyncio
async def test_none_mode_yields_a_no_op_auth():
    result = await UpstreamCredentialProvider().resolve_credentials(_SUBJECT, _spec(NoneConfig()))
    assert isinstance(result, Ok)
    assert isinstance(result.ok, NoOpAuth)


@pytest.mark.asyncio
async def test_api_key_shared_emits_the_configured_header():
    config = ApiKeyConfig(
        header_name="X-API-Key",
        value_prefix="",
        key_source=SharedKey(value=SecretStr("secret-key")),
    )
    result = await UpstreamCredentialProvider().resolve_credentials(_SUBJECT, _spec(config))
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
    result = await UpstreamCredentialProvider().resolve_credentials(_SUBJECT, _spec(config))
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
    result = await UpstreamCredentialProvider(oauth_token_store=store).resolve_credentials(
        Subject(tenant_id="", subject_id="alice"), _spec(AuthorizationCodeConfig())
    )
    assert isinstance(result, Ok)
    assert _emitted(result.ok)["Authorization"] == "Bearer at-alice"


@pytest.mark.asyncio
async def test_authorization_code_without_token_is_semantically_unauthorized():
    result = await UpstreamCredentialProvider(oauth_token_store=_FakeTokenStore({})).resolve_credentials(
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

    result = await UpstreamCredentialProvider(oauth_token_store=_Unavailable()).resolve_credentials(
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
    bob = await provider.resolve_credentials(Subject(tenant_id="", subject_id="bob"), _spec(AuthorizationCodeConfig()))
    assert isinstance(alice, Ok) and _emitted(alice.ok)["Authorization"] == "Bearer at-alice"
    assert isinstance(bob, Error) and bob.error.tag == "unauthorized"


@pytest.mark.asyncio
async def test_authorization_code_isolates_by_server_id_even_when_servers_share_a_url():
    """A token stored for one server must be invisible to a different server_id pointing at the
    same upstream URL: credentials bind to the server entry they were authorized for, so a
    recreated or duplicated server starts unauthorized instead of inheriting the old grant. Guards
    against any future token lookup keyed on the resource URL instead of (user_id, server_id) --
    both the egress resolve and the has_user_token discovery check must agree."""
    shared_url = "https://upstream.example.com"
    store = _FakeTokenStore({("alice", "server-a"): OAuthToken(access_token="at-alice")})
    provider = UpstreamCredentialProvider(oauth_token_store=store)
    subject = Subject(tenant_id="", subject_id="alice")
    spec_a = ServerSpec(server_id="server-a", resource=shared_url, config=AuthorizationCodeConfig())
    spec_b = ServerSpec(server_id="server-b", resource=shared_url, config=AuthorizationCodeConfig())

    granted = await provider.resolve_credentials(subject, spec_a)
    fresh = await provider.resolve_credentials(subject, spec_b)

    assert isinstance(granted, Ok) and _emitted(granted.ok)["Authorization"] == "Bearer at-alice"
    assert isinstance(fresh, Error) and fresh.error.tag == "unauthorized"
    assert await provider.has_user_token(subject, spec_a) is True
    assert await provider.has_user_token(subject, spec_b) is False


@pytest.mark.asyncio
async def test_has_user_token_reflects_the_stored_token():
    present = UpstreamCredentialProvider(
        oauth_token_store=_FakeTokenStore({("alice", "s"): OAuthToken(access_token="at")})
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
    assert await provider.has_user_token(Subject(tenant_id="", subject_id="a"), spec) is False


class _FakeExchanger:
    def __init__(self, result: Result[OAuthToken, CredError]) -> None:
        self._result = result
        self.calls: list[tuple[str, str, str]] = []
        self.invalidations: list[tuple[str, str, str]] = []

    async def exchange(self, subject_token, server, config, *, tenant_id=""):
        self.calls.append((subject_token, tenant_id, server.server_id))
        return self._result

    async def invalidate(self, subject_token, server, config, *, tenant_id=""):
        self.invalidations.append((subject_token, tenant_id, server.server_id))


_OBO = TokenExchangeConfig(
    token_exchange_endpoint="https://idp.example.com/token",
    client_id="cid",
    client_secret=SecretStr("csec"),
)


@pytest.mark.asyncio
async def test_token_exchange_emits_the_exchanged_bearer():
    exchanger = _FakeExchanger(Ok(OAuthToken(access_token="exchanged-at")))
    subject = Subject(tenant_id="acme", subject_id="alice", inbound_token=SecretStr("caller-jwt"))
    result = await UpstreamCredentialProvider(token_exchanger=exchanger).resolve_credentials(subject, _spec(_OBO))
    assert isinstance(result, Ok)
    assert _emitted(result.ok)["Authorization"] == "Bearer exchanged-at"
    # The arm hands the unwrapped caller token AND the tenant to the exchanger, never the upstream.
    assert exchanger.calls == [("caller-jwt", "acme", "s")]


@pytest.mark.asyncio
async def test_invalidate_credentials_drops_the_exchanged_token_for_the_subject_and_tenant():
    exchanger = _FakeExchanger(Ok(OAuthToken(access_token="exchanged-at")))
    provider = UpstreamCredentialProvider(token_exchanger=exchanger)
    subject = Subject(tenant_id="acme", subject_id="alice", inbound_token=SecretStr("caller-jwt"))
    await provider.invalidate_credentials(subject, _spec(_OBO))
    assert exchanger.invalidations == [("caller-jwt", "acme", "s")]


@pytest.mark.asyncio
async def test_invalidate_credentials_is_a_noop_without_a_caller_token():
    exchanger = _FakeExchanger(Ok(OAuthToken(access_token="never")))
    provider = UpstreamCredentialProvider(token_exchanger=exchanger)
    await provider.invalidate_credentials(Subject(tenant_id="acme", subject_id="alice"), _spec(_OBO))
    assert exchanger.invalidations == []


@pytest.mark.asyncio
async def test_token_exchange_without_caller_token_is_unauthorized():
    exchanger = _FakeExchanger(Ok(OAuthToken(access_token="never")))
    result = await UpstreamCredentialProvider(token_exchanger=exchanger).resolve_credentials(
        Subject(tenant_id="", subject_id="alice"), _spec(_OBO)
    )
    assert isinstance(result, Error)
    assert result.error.tag == "unauthorized"
    assert result.error.unauthorized.www_authenticate == 'Bearer error="invalid_request"'
    # No caller token means nothing to exchange: the IdP is never hit.
    assert exchanger.calls == []


@pytest.mark.asyncio
async def test_token_exchange_propagates_the_exchanger_error():
    err = CredError.of_upstream_unavailable("idp down")
    result = await UpstreamCredentialProvider(token_exchanger=_FakeExchanger(Error(err))).resolve_credentials(
        Subject(tenant_id="", subject_id="alice", inbound_token=SecretStr("jwt")),
        _spec(_OBO),
    )
    assert isinstance(result, Error)
    assert result.error.tag == "upstream_unavailable"


@pytest.mark.asyncio
async def test_token_exchange_without_an_exchanger_fails_closed():
    # The fail-closed default (no exchanger wired) must not produce a credential.
    result = await UpstreamCredentialProvider().resolve_credentials(
        Subject(tenant_id="", subject_id="alice", inbound_token=SecretStr("jwt")),
        _spec(_OBO),
    )
    assert isinstance(result, Error)
    assert result.error.tag == "misconfigured"


@pytest.mark.asyncio
async def test_passthrough_forwards_the_inbound_token_verbatim():
    subject = Subject(tenant_id="", subject_id="", inbound_token=SecretStr("Bearer upstream-xyz"))
    result = await UpstreamCredentialProvider().resolve_credentials(subject, _spec(PassthroughConfig()))
    assert isinstance(result, Ok)
    assert isinstance(result.ok, StaticHeaderAuth)
    assert _emitted(result.ok)["Authorization"] == "Bearer upstream-xyz"


@pytest.mark.asyncio
async def test_passthrough_without_inbound_token_is_a_no_op():
    result = await UpstreamCredentialProvider().resolve_credentials(_SUBJECT, _spec(PassthroughConfig()))
    assert isinstance(result, Ok)
    assert isinstance(result.ok, NoOpAuth)


class _FakeM2MSource:
    """A ClientCredentialsTokenSource returning a canned result and recording refetches."""

    def __init__(self, result) -> None:
        self._result = result
        self.gets: list[str] = []
        self.refetches: list[tuple[str, str]] = []

    async def get(self, server_id: str, config):
        self.gets.append(server_id)
        return self._result

    async def refetch(self, server_id: str, config, failed_access_token: str):
        self.refetches.append((server_id, failed_access_token))
        return "fresh-m2m"


_M2M = ClientCredentialsConfig(
    client_id="cid",
    client_secret=SecretStr("csec"),
    token_url="https://idp.example.com/token",
)


async def _emitted_async(auth: httpx.Auth, respond=None) -> tuple[httpx.Headers, list[httpx.Request]]:
    """Drive the async auth flow one request at a time, replying via ``respond`` when given."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return respond(request) if respond else httpx.Response(200)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler), auth=auth) as client:
        await client.get("https://upstream.example.com/mcp")
    return seen[-1].headers, seen


@pytest.mark.asyncio
async def test_client_credentials_emits_the_minted_bearer():
    source = _FakeM2MSource(Ok(OAuthToken(access_token="m2m-at")))
    result = await UpstreamCredentialProvider(client_credentials_source=source).resolve_credentials(
        _SUBJECT, _spec(_M2M)
    )
    assert isinstance(result, Ok)
    headers, _ = await _emitted_async(result.ok)
    assert headers["Authorization"] == "Bearer m2m-at"
    assert source.gets == ["s"]


@pytest.mark.asyncio
async def test_client_credentials_ignores_the_subject():
    # The contract's no-user-context clause: every caller shares the one client identity.
    source = _FakeM2MSource(Ok(OAuthToken(access_token="m2m-at")))
    provider = UpstreamCredentialProvider(client_credentials_source=source)
    alice = await provider.resolve_credentials(Subject(tenant_id="t1", subject_id="alice"), _spec(_M2M))
    bob = await provider.resolve_credentials(Subject(tenant_id="t2", subject_id="bob"), _spec(_M2M))
    assert isinstance(alice, Ok) and isinstance(bob, Ok)
    alice_headers, _ = await _emitted_async(alice.ok)
    bob_headers, _ = await _emitted_async(bob.ok)
    assert alice_headers["Authorization"] == bob_headers["Authorization"] == "Bearer m2m-at"


@pytest.mark.asyncio
async def test_client_credentials_auth_retries_a_401_through_the_source():
    source = _FakeM2MSource(Ok(OAuthToken(access_token="stale-at")))
    result = await UpstreamCredentialProvider(client_credentials_source=source).resolve_credentials(
        _SUBJECT, _spec(_M2M)
    )
    assert isinstance(result, Ok)

    def respond(request: httpx.Request) -> httpx.Response:
        is_stale = request.headers["Authorization"] == "Bearer stale-at"
        return httpx.Response(401) if is_stale else httpx.Response(200)

    headers, seen = await _emitted_async(result.ok, respond)
    assert headers["Authorization"] == "Bearer fresh-m2m"
    assert len(seen) == 2
    assert source.refetches == [("s", "stale-at")]


@pytest.mark.asyncio
async def test_client_credentials_propagates_the_source_error():
    source = _FakeM2MSource(Error(CredError.of_upstream_unavailable("idp down")))
    result = await UpstreamCredentialProvider(client_credentials_source=source).resolve_credentials(
        _SUBJECT, _spec(_M2M)
    )
    assert isinstance(result, Error)
    assert result.error.tag == "upstream_unavailable"


@pytest.mark.asyncio
async def test_client_credentials_with_no_source_wired_fails_closed_on_missing_config():
    # The default source validates the grant fields before any network is touched.
    result = await UpstreamCredentialProvider().resolve_credentials(_SUBJECT, _spec(ClientCredentialsConfig()))
    assert isinstance(result, Error)
    assert result.error.tag == "misconfigured"


_STUBBED = [
    ("api_key_byok", ApiKeyConfig(key_source=Byok())),
    ("aws_sigv4", AwsSigV4Config(region="us-east-1")),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("label, config", _STUBBED)
async def test_unbuilt_arms_fail_closed_with_not_implemented(label, config):
    result = await UpstreamCredentialProvider().resolve_credentials(_SUBJECT, _spec(config))
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


def _two_leg_ok(bearer: str) -> list:
    return [
        Ok(ExchangedToken(access_token="the-id-jag", expires_in=300)),
        Ok(ExchangedToken(access_token=bearer, expires_in=3600)),
    ]


@pytest.mark.asyncio
async def test_id_jag_reuses_the_cached_bearer_for_an_unchanged_config():
    endpoint = _FakeTokenEndpoint(_two_leg_ok("first-bearer"))
    provider = UpstreamCredentialProvider(token_endpoint=endpoint)

    first = await provider.resolve_credentials(_with_inbound("user-id-token"), _spec(_id_jag_config()))
    second = await provider.resolve_credentials(_with_inbound("user-id-token"), _spec(_id_jag_config()))

    assert isinstance(first, Ok) and isinstance(second, Ok)
    assert _emitted(second.ok)["Authorization"] == "Bearer first-bearer"
    assert len(endpoint.calls) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "changed",
    [
        _id_jag_config().model_copy(update={"audience": "api://other"}),
        _id_jag_config().model_copy(update={"resource": "https://other.example.com/mcp"}),
        _id_jag_config().model_copy(update={"scopes": ("mcp.read", "mcp.write")}),
        _id_jag_config().model_copy(update={"org_token_endpoint": "https://idp.example.com/v2/token"}),
        _id_jag_config().model_copy(update={"resource_token_endpoint": "https://mcp-as.example.com/v2/token"}),
        _id_jag_config().model_copy(update={"client_id": "litellm-rotated"}),
        _id_jag_config().model_copy(update={"client_auth": ClientSecretAuth(client_secret=SecretStr("rotated"))}),
        _id_jag_config().model_copy(update={"subject_token_type": "urn:ietf:params:oauth:token-type:saml2"}),
    ],
    ids=[
        "audience",
        "resource",
        "scopes",
        "org_token_endpoint",
        "resource_token_endpoint",
        "client_id",
        "client_auth",
        "subject_token_type",
    ],
)
async def test_id_jag_config_change_forces_a_fresh_exchange(changed):
    endpoint = _FakeTokenEndpoint(_two_leg_ok("old-policy-bearer") + _two_leg_ok("new-policy-bearer"))
    provider = UpstreamCredentialProvider(token_endpoint=endpoint)

    before = await provider.resolve_credentials(_with_inbound("user-id-token"), _spec(_id_jag_config()))
    after = await provider.resolve_credentials(_with_inbound("user-id-token"), _spec(changed))

    assert isinstance(before, Ok) and isinstance(after, Ok)
    assert _emitted(after.ok)["Authorization"] == "Bearer new-policy-bearer"
    assert len(endpoint.calls) == 4


@pytest.mark.asyncio
async def test_id_jag_does_not_share_the_cached_bearer_across_caller_tokens():
    endpoint = _FakeTokenEndpoint(_two_leg_ok("alice-bearer") + _two_leg_ok("bob-bearer"))
    provider = UpstreamCredentialProvider(token_endpoint=endpoint)

    alice = await provider.resolve_credentials(_with_inbound("alice-id-token"), _spec(_id_jag_config()))
    bob = await provider.resolve_credentials(_with_inbound("bob-id-token"), _spec(_id_jag_config()))

    assert isinstance(alice, Ok) and isinstance(bob, Ok)
    assert _emitted(bob.ok)["Authorization"] == "Bearer bob-bearer"
    assert len(endpoint.calls) == 4


@pytest.mark.asyncio
async def test_invalidate_credentials_evicts_the_id_jag_bearer_so_the_next_resolve_re_exchanges():
    endpoint = _FakeTokenEndpoint(_two_leg_ok("rejected-bearer") + _two_leg_ok("fresh-bearer"))
    provider = UpstreamCredentialProvider(token_endpoint=endpoint)
    subject = _with_inbound("user-id-token")

    first = await provider.resolve_credentials(subject, _spec(_id_jag_config()))
    await provider.invalidate_credentials(subject, _spec(_id_jag_config()))
    second = await provider.resolve_credentials(subject, _spec(_id_jag_config()))

    assert isinstance(first, Ok) and isinstance(second, Ok)
    assert _emitted(second.ok)["Authorization"] == "Bearer fresh-bearer"
    assert len(endpoint.calls) == 4


@pytest.mark.asyncio
async def test_invalidate_credentials_for_id_jag_is_a_noop_without_a_caller_token():
    endpoint = _FakeTokenEndpoint(_two_leg_ok("cached-bearer"))
    provider = UpstreamCredentialProvider(token_endpoint=endpoint)
    subject = _with_inbound("user-id-token")

    first = await provider.resolve_credentials(subject, _spec(_id_jag_config()))
    await provider.invalidate_credentials(Subject(tenant_id="", subject_id="alice"), _spec(_id_jag_config()))
    second = await provider.resolve_credentials(subject, _spec(_id_jag_config()))

    assert isinstance(first, Ok) and isinstance(second, Ok)
    assert _emitted(second.ok)["Authorization"] == "Bearer cached-bearer"
    assert len(endpoint.calls) == 2


@pytest.mark.asyncio
async def test_id_jag_leg2_carries_a_resource_as_rejection_classifier_and_leg1_does_not():
    """A section 5.2 rejection means different things per leg: at the resource AS redeeming a
    freshly minted ID-JAG it is a client-registration misconfiguration; at the IdP it keeps the
    default mapping. The classifier therefore rides only the leg-2 fetch."""
    from litellm.proxy._experimental.mcp_server.outbound_credentials.token_endpoint import (
        TokenEndpointRejection,
    )

    endpoint = _FakeTokenEndpoint(_two_leg_ok("final-access"))
    provider = UpstreamCredentialProvider(token_endpoint=endpoint)

    result = await provider.resolve_credentials(_with_inbound("user-id-token"), _spec(_id_jag_config()))

    assert isinstance(result, Ok)
    leg1_classifier, leg2_classifier = endpoint.classifiers
    assert leg1_classifier is None
    assert leg2_classifier is not None
    classified = leg2_classifier(
        TokenEndpointRejection(status_code=400, error="invalid_grant", error_description="client mismatch")
    )
    assert classified is not None
    assert classified.tag == "misconfigured"
    assert "litellm" in classified.summary
    assert "client mismatch" in classified.summary
    assert (
        leg2_classifier(TokenEndpointRejection(status_code=502, error="server_error", error_description=None)) is None
    )


@pytest.mark.parametrize("code", ["invalid_grant", "invalid_client", "unauthorized_client", "invalid_target"])
def test_resource_as_misconfig_codes_classify_as_misconfigured(code):
    from litellm.proxy._experimental.mcp_server.outbound_credentials.resolver import (
        _classify_resource_as_rejection,
    )
    from litellm.proxy._experimental.mcp_server.outbound_credentials.token_endpoint import (
        TokenEndpointRejection,
    )

    classified = _classify_resource_as_rejection(
        "gw-client", TokenEndpointRejection(status_code=400, error=code, error_description=None)
    )
    assert classified is not None
    assert classified.tag == "misconfigured"
    assert "gw-client" in classified.summary
