"""Tests for the resolver dispatch: live arms produce auth, stubbed arms fail closed.

`none`, `api_key` (shared-key source), `passthrough`, `authorization_code`, `token_exchange`, and
`client_credentials` are implemented; every other arm, plus the `api_key` BYOK source, returns a
typed `not_implemented` error until its mode lands. Parametrizing the stubs over one config each
also guards reachability: a dropped `case` would hit `assert_never` and raise instead of
returning the stub.
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone

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
    PrivateKeyJwtAuth,
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
from litellm.proxy._experimental.mcp_server.outbound_credentials.sso_assertion_store import (
    AssertionStoreUnavailable,
    SSOIdentityAssertion,
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


class _FakeAssertionStore:
    """The SSO assertion read seam, canned per user_id and recording every lookup."""

    def __init__(self, assertions: dict[str, SSOIdentityAssertion] | None = None) -> None:
        self._assertions = dict(assertions or {})
        self.lookups: list[str] = []

    async def fetch(self, user_id: str) -> SSOIdentityAssertion | None:
        self.lookups.append(user_id)
        return self._assertions.get(user_id)


def _assertion(id_token: str, expires_in: timedelta | None = timedelta(minutes=30)) -> SSOIdentityAssertion:
    expires_at = datetime.now(timezone.utc) + expires_in if expires_in is not None else None
    return SSOIdentityAssertion(id_token=SecretStr(id_token), expires_at=expires_at)


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
async def test_id_jag_without_inbound_token_or_stored_assertion_is_precondition_required_no_http():
    endpoint = _FakeTokenEndpoint([])
    store = _FakeAssertionStore()
    provider = UpstreamCredentialProvider(token_endpoint=endpoint, sso_assertion_store=store)
    result = await provider.resolve_credentials(
        Subject(tenant_id="", subject_id="alice"), _spec(_id_jag_config())
    )

    assert isinstance(result, Error)
    assert result.error.tag == "precondition_required"
    assert endpoint.calls == []
    assert store.lookups == ["alice"]


@pytest.mark.asyncio
async def test_id_jag_exchanges_the_stored_sso_assertion_when_the_caller_presents_no_token():
    """The agent-triggered flow: a brokered LiteLLM credential carries no IdP token, so leg 1's
    subject is the assertion captured for that user at SSO login."""
    endpoint = _FakeTokenEndpoint(_two_leg_ok("final-access"))
    store = _FakeAssertionStore({"alice": _assertion("alice-id-token")})
    provider = UpstreamCredentialProvider(token_endpoint=endpoint, sso_assertion_store=store)

    result = await provider.resolve_credentials(
        Subject(tenant_id="", subject_id="alice"), _spec(_id_jag_config())
    )

    assert isinstance(result, Ok)
    assert _emitted(result.ok)["Authorization"] == "Bearer final-access"
    assert store.lookups == ["alice"]
    _, _, leg1_params = endpoint.calls[0]
    assert leg1_params["subject_token"] == "alice-id-token"
    assert leg1_params["requested_token_type"] == "urn:ietf:params:oauth:token-type:id-jag"


@pytest.mark.asyncio
async def test_id_jag_prefers_the_callers_own_token_over_the_stored_assertion():
    endpoint = _FakeTokenEndpoint(_two_leg_ok("final-access"))
    store = _FakeAssertionStore({"alice": _assertion("stored-id-token")})
    provider = UpstreamCredentialProvider(token_endpoint=endpoint, sso_assertion_store=store)

    result = await provider.resolve_credentials(_with_inbound("inbound-id-token"), _spec(_id_jag_config()))

    assert isinstance(result, Ok)
    _, _, leg1_params = endpoint.calls[0]
    assert leg1_params["subject_token"] == "inbound-id-token"
    assert store.lookups == []


@pytest.mark.asyncio
async def test_id_jag_refuses_an_expired_stored_assertion_without_calling_the_idp():
    endpoint = _FakeTokenEndpoint([])
    store = _FakeAssertionStore({"alice": _assertion("stale-id-token", expires_in=-timedelta(seconds=1))})
    provider = UpstreamCredentialProvider(token_endpoint=endpoint, sso_assertion_store=store)

    result = await provider.resolve_credentials(
        Subject(tenant_id="", subject_id="alice"), _spec(_id_jag_config())
    )

    assert isinstance(result, Error)
    assert result.error.tag == "precondition_required"
    assert endpoint.calls == []


@pytest.mark.asyncio
async def test_id_jag_accepts_a_stored_assertion_that_declares_no_expiry():
    endpoint = _FakeTokenEndpoint(_two_leg_ok("final-access"))
    store = _FakeAssertionStore({"alice": _assertion("undated-id-token", expires_in=None)})
    provider = UpstreamCredentialProvider(token_endpoint=endpoint, sso_assertion_store=store)

    result = await provider.resolve_credentials(
        Subject(tenant_id="", subject_id="alice"), _spec(_id_jag_config())
    )

    assert isinstance(result, Ok)
    _, _, leg1_params = endpoint.calls[0]
    assert leg1_params["subject_token"] == "undated-id-token"


@pytest.mark.asyncio
async def test_id_jag_never_reads_the_store_for_an_unidentified_caller():
    """An empty subject_id must not select a credential; otherwise every anonymous caller would
    share one store slot."""
    endpoint = _FakeTokenEndpoint([])
    store = _FakeAssertionStore({"": _assertion("anonymous-slot")})
    provider = UpstreamCredentialProvider(token_endpoint=endpoint, sso_assertion_store=store)

    result = await provider.resolve_credentials(Subject(tenant_id="", subject_id=""), _spec(_id_jag_config()))

    assert isinstance(result, Error)
    assert result.error.tag == "precondition_required"
    assert store.lookups == []
    assert endpoint.calls == []


@pytest.mark.asyncio
async def test_id_jag_keeps_store_sourced_bearers_partitioned_per_user():
    endpoint = _FakeTokenEndpoint(
        [
            Ok(ExchangedToken(access_token="alice-id-jag", expires_in=300)),
            Ok(ExchangedToken(access_token="alice-bearer", expires_in=3600)),
            Ok(ExchangedToken(access_token="bob-id-jag", expires_in=300)),
            Ok(ExchangedToken(access_token="bob-bearer", expires_in=3600)),
        ]
    )
    store = _FakeAssertionStore(
        {"alice": _assertion("alice-id-token"), "bob": _assertion("bob-id-token")}
    )
    provider = UpstreamCredentialProvider(token_endpoint=endpoint, sso_assertion_store=store)

    alice = await provider.resolve_credentials(
        Subject(tenant_id="", subject_id="alice"), _spec(_id_jag_config())
    )
    bob = await provider.resolve_credentials(Subject(tenant_id="", subject_id="bob"), _spec(_id_jag_config()))

    assert isinstance(alice, Ok) and isinstance(bob, Ok)
    assert _emitted(alice.ok)["Authorization"] == "Bearer alice-bearer"
    assert _emitted(bob.ok)["Authorization"] == "Bearer bob-bearer"


_DRIVER_DETAIL = "could not connect to host=pg-primary.internal port=5432 user=litellm"


class _OutageAssertionStore:
    """A store whose backing DB is down, failing with a driver message full of internals."""

    def __init__(self) -> None:
        self.lookups: list[str] = []

    async def fetch(self, user_id: str) -> SSOIdentityAssertion | None:
        self.lookups.append(user_id)
        raise AssertionStoreUnavailable(_DRIVER_DETAIL)


@pytest.mark.asyncio
async def test_id_jag_maps_an_assertion_store_outage_to_upstream_unavailable():
    """A store outage must not escape as an unhandled error, and must not be reported as a missing
    assertion: telling the user to sign in again does not fix a database that is down."""
    endpoint = _FakeTokenEndpoint([])
    provider = UpstreamCredentialProvider(token_endpoint=endpoint, sso_assertion_store=_OutageAssertionStore())

    result = await provider.resolve_credentials(
        Subject(tenant_id="", subject_id="alice"), _spec(_id_jag_config())
    )

    assert isinstance(result, Error)
    assert result.error.tag == "upstream_unavailable"
    assert endpoint.calls == []


@pytest.mark.asyncio
async def test_id_jag_store_outage_does_not_leak_driver_detail_to_the_caller(caplog):
    """`upstream_unavailable` is rendered into the 503 body verbatim, so the driver's message, which
    can name hosts, ports and users, must stay out of the summary and go to the log instead."""
    provider = UpstreamCredentialProvider(
        token_endpoint=_FakeTokenEndpoint([]), sso_assertion_store=_OutageAssertionStore()
    )

    with caplog.at_level(logging.WARNING):
        result = await provider.resolve_credentials(
            Subject(tenant_id="", subject_id="alice"), _spec(_id_jag_config())
        )

    assert isinstance(result, Error)
    assert _DRIVER_DETAIL not in result.error.summary
    assert "pg-primary.internal" not in result.error.summary
    # The operator still needs it, so it must be in the log.
    assert _DRIVER_DETAIL in caplog.text


@pytest.mark.asyncio
async def test_id_jag_invalidation_survives_an_assertion_store_outage():
    """invalidate_credentials runs on the upstream-401 retry path, so a store outage there must be
    swallowed rather than turning a recoverable 401 into a 500."""
    provider = UpstreamCredentialProvider(
        token_endpoint=_FakeTokenEndpoint([]), sso_assertion_store=_OutageAssertionStore()
    )

    await provider.invalidate_credentials(
        Subject(tenant_id="", subject_id="alice"), _spec(_id_jag_config())
    )


class _FlakyAssertionStore:
    """Serves an assertion, but fails while ``down`` is set."""

    def __init__(self, assertion: SSOIdentityAssertion) -> None:
        self._assertion = assertion
        self.down = False

    async def fetch(self, user_id: str) -> SSOIdentityAssertion | None:
        if self.down:
            raise AssertionStoreUnavailable("connection refused")
        return self._assertion


@pytest.mark.asyncio
async def test_id_jag_evicts_the_rejected_bearer_even_if_the_store_is_down_during_invalidation():
    """The upstream-401 recovery sequence with a transient store blip.

    Invalidation runs while the store is unreachable and the store recovers before the retry
    resolves. Deriving the eviction key from a fresh lookup would evict nothing and then recompute
    the identical key, handing the retry the very bearer the upstream just rejected.
    """
    endpoint = _FakeTokenEndpoint(_two_leg_ok("rejected-bearer") + _two_leg_ok("reminted-bearer"))
    store = _FlakyAssertionStore(_assertion("alice-id-token"))
    provider = UpstreamCredentialProvider(token_endpoint=endpoint, sso_assertion_store=store)
    subject = Subject(tenant_id="", subject_id="alice")
    spec = _spec(_id_jag_config())

    first = await provider.resolve_credentials(subject, spec)
    assert isinstance(first, Ok)
    assert _emitted(first.ok)["Authorization"] == "Bearer rejected-bearer"

    store.down = True
    await provider.invalidate_credentials(subject, spec)
    store.down = False

    second = await provider.resolve_credentials(subject, spec)
    assert isinstance(second, Ok)
    assert _emitted(second.ok)["Authorization"] == "Bearer reminted-bearer"
    assert len(endpoint.calls) == 4


class _SwitchableAssertionStore:
    """Serves whichever assertion the test currently points it at, as a re-login would."""

    def __init__(self, id_token: str) -> None:
        self.id_token = id_token

    async def fetch(self, user_id: str) -> SSOIdentityAssertion | None:
        return _assertion(self.id_token)


@pytest.mark.asyncio
async def test_id_jag_invalidation_clears_every_live_bearer_for_the_principal():
    """Overlapping store-sourced requests for one principal can hold different keys (a re-login
    between them mints a different subject token). Invalidation must clear all of them: keeping
    only the newest would let one request's 401 recovery evict the other's entry and leave its own
    rejected bearer cached to be replayed on the retry."""
    endpoint = _FakeTokenEndpoint(
        _two_leg_ok("bearer-from-first") + _two_leg_ok("bearer-from-second") + _two_leg_ok("reminted")
    )
    store = _SwitchableAssertionStore("id-token-first")
    provider = UpstreamCredentialProvider(token_endpoint=endpoint, sso_assertion_store=store)
    subject = Subject(tenant_id="", subject_id="alice")
    spec = _spec(_id_jag_config())

    first = await provider.resolve_credentials(subject, spec)
    store.id_token = "id-token-second"
    second = await provider.resolve_credentials(subject, spec)
    assert isinstance(first, Ok) and isinstance(second, Ok)
    assert _emitted(first.ok)["Authorization"] == "Bearer bearer-from-first"
    assert _emitted(second.ok)["Authorization"] == "Bearer bearer-from-second"

    await provider.invalidate_credentials(subject, spec)

    # Point the store back at the first token. If that entry had survived the invalidation this
    # would replay "bearer-from-first", which is the bearer an upstream may already have rejected.
    store.id_token = "id-token-first"
    third = await provider.resolve_credentials(subject, spec)
    assert isinstance(third, Ok)
    assert _emitted(third.ok)["Authorization"] == "Bearer reminted"


class _SequentialAssertionStore:
    """Issues a distinct assertion per call unless pinned, so concurrent resolutions genuinely
    mint distinct credentials rather than collapsing onto one through single-flight."""

    def __init__(self) -> None:
        self.pinned: str | None = None
        self.issued: list[str] = []
        self._n = 0

    async def fetch(self, user_id: str) -> SSOIdentityAssertion | None:
        await asyncio.sleep(0)
        if self.pinned is not None:
            return _assertion(self.pinned)
        self._n += 1
        token = f"id-token-{self._n}"
        self.issued.append(token)
        return _assertion(token)


class _CountingTokenEndpoint:
    """Mints a unique bearer per exchange and yields, so exchanges interleave."""

    def __init__(self) -> None:
        self._n = 0

    async def fetch(self, endpoint, client_id, grant_params, client_auth):
        await asyncio.sleep(0)
        self._n += 1
        return Ok(ExchangedToken(access_token=f"tok-{self._n}", expires_in=3600))


@pytest.mark.asyncio
async def test_id_jag_invalidation_leaves_no_bearer_behind_under_concurrency():
    """After invalidation, no bearer minted before it may ever be served again.

    Drives many overlapping resolutions that each mint a distinct credential, invalidates once,
    then replays every subject token that was issued. Any credential the eviction could not reach
    would show up here as a replayed pre-invalidation bearer.
    """
    concurrency = 20
    endpoint = _CountingTokenEndpoint()
    store = _SequentialAssertionStore()
    provider = UpstreamCredentialProvider(token_endpoint=endpoint, sso_assertion_store=store)
    subject = Subject(tenant_id="t", subject_id="alice")
    spec = _spec(_id_jag_config())

    results = await asyncio.gather(*(provider.resolve_credentials(subject, spec) for _ in range(concurrency)))
    before = {_emitted(r.ok)["Authorization"] for r in results if isinstance(r, Ok)}
    issued = list(store.issued)
    # Guard the guard: if these collapsed onto one credential the test would prove nothing.
    assert len(before) > 1

    await provider.invalidate_credentials(subject, spec)

    for token in issued:
        store.pinned = token
        replayed = await provider.resolve_credentials(subject, spec)
        assert isinstance(replayed, Ok)
        assert _emitted(replayed.ok)["Authorization"] not in before


@pytest.mark.asyncio
async def test_id_jag_never_serves_a_bearer_minted_for_a_different_caller():
    """Two unidentified-principal callers share a slot, so the fingerprint, not the key, is what
    keeps them apart: a mismatch must read as a miss rather than hand over the other's bearer."""
    endpoint = _FakeTokenEndpoint(_two_leg_ok("first-callers-bearer") + _two_leg_ok("second-callers-bearer"))
    provider = UpstreamCredentialProvider(token_endpoint=endpoint)
    spec = _spec(_id_jag_config())

    first = await provider.resolve_credentials(_with_inbound("caller-one-token"), spec)
    second = await provider.resolve_credentials(_with_inbound("caller-two-token"), spec)

    assert isinstance(first, Ok) and isinstance(second, Ok)
    assert _emitted(first.ok)["Authorization"] == "Bearer first-callers-bearer"
    assert _emitted(second.ok)["Authorization"] == "Bearer second-callers-bearer"


@pytest.mark.asyncio
async def test_id_jag_rotating_the_signing_key_does_not_reuse_the_cached_bearer():
    """The cache key fingerprints the private-key-JWT client auth, so a rotated signing key
    re-mints instead of serving a bearer authorized under the retired key."""
    endpoint = _FakeTokenEndpoint(_two_leg_ok("old-key-bearer") + _two_leg_ok("new-key-bearer"))
    store = _FakeAssertionStore({"alice": _assertion("alice-id-token")})
    provider = UpstreamCredentialProvider(token_endpoint=endpoint, sso_assertion_store=store)
    subject = Subject(tenant_id="", subject_id="alice")

    def _with_key(pem: str) -> IdJagConfig:
        return _id_jag_config().model_copy(
            update={"client_auth": PrivateKeyJwtAuth(private_key=SecretStr(pem), key_id="kid-1")}
        )

    first = await provider.resolve_credentials(subject, _spec(_with_key("-----OLD KEY-----")))
    second = await provider.resolve_credentials(subject, _spec(_with_key("-----NEW KEY-----")))

    assert isinstance(first, Ok) and isinstance(second, Ok)
    assert _emitted(first.ok)["Authorization"] == "Bearer old-key-bearer"
    assert _emitted(second.ok)["Authorization"] == "Bearer new-key-bearer"
    assert len(endpoint.calls) == 4


@pytest.mark.asyncio
async def test_id_jag_reads_a_naive_stored_expiry_as_utc():
    """A stored expires_at that lost its offset must still compare rather than raise: an aware/naive
    comparison would be a TypeError on the egress path, turning a 412 into a 500."""
    endpoint = _FakeTokenEndpoint([])
    naive_past = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=1)
    store = _FakeAssertionStore(
        {"alice": SSOIdentityAssertion(id_token=SecretStr("stale"), expires_at=naive_past)}
    )
    provider = UpstreamCredentialProvider(token_endpoint=endpoint, sso_assertion_store=store)

    result = await provider.resolve_credentials(
        Subject(tenant_id="", subject_id="alice"), _spec(_id_jag_config())
    )

    assert isinstance(result, Error)
    assert result.error.tag == "precondition_required"
    assert endpoint.calls == []


@pytest.mark.asyncio
async def test_invalidate_evicts_a_store_sourced_id_jag_bearer():
    """The upstream-401 recovery path. Keyed off the request alone the eviction would miss, and the
    rejected bearer would be replayed until its TTL."""
    endpoint = _FakeTokenEndpoint(_two_leg_ok("first-bearer") + _two_leg_ok("second-bearer"))
    store = _FakeAssertionStore({"alice": _assertion("alice-id-token")})
    provider = UpstreamCredentialProvider(token_endpoint=endpoint, sso_assertion_store=store)
    subject = Subject(tenant_id="", subject_id="alice")
    spec = _spec(_id_jag_config())

    first = await provider.resolve_credentials(subject, spec)
    await provider.invalidate_credentials(subject, spec)
    second = await provider.resolve_credentials(subject, spec)

    assert isinstance(first, Ok) and isinstance(second, Ok)
    assert _emitted(first.ok)["Authorization"] == "Bearer first-bearer"
    assert _emitted(second.ok)["Authorization"] == "Bearer second-bearer"
    assert len(endpoint.calls) == 4


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
