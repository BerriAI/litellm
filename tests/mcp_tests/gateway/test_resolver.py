"""Spec tests for the v2 upstream-credential resolver scaffold.

Clean-room litmus: every case constructs `Subject` / `ServerSpec` directly, with zero v1
fixtures. If an arm could not be exercised without a v1 request object, the seam has leaked.
"""

from datetime import datetime, timedelta, timezone

import httpx
import pytest
from pydantic import SecretStr, ValidationError

from litellm.proxy.gateway.mcp._spike_exhaustiveness import http_status
from litellm.proxy.gateway.mcp.outbound_credentials.client_credentials_fetcher import (
    ClientCredentialsFetcher,
)
from litellm.proxy.gateway.mcp.outbound_credentials.clock import Clock
from litellm.proxy.gateway.mcp.outbound_credentials.credential_store import (
    CredentialKey,
    CredentialStore,
    InMemoryCredentialStore,
)
from litellm.proxy.gateway.mcp.outbound_credentials.httpx_auth import (
    NoOpAuth,
    StaticHeaderAuth,
)
from litellm.proxy.gateway.mcp.outbound_credentials.resolver import (
    UpstreamCredentialProvider,
)
from litellm.proxy.gateway.mcp.outbound_credentials.service_token_store import (
    InMemoryServiceTokenStore,
    ServiceTokenKey,
    ServiceTokenStore,
)
from litellm.proxy.gateway.mcp.outbound_credentials.signer_factory import (
    SignerFactory,
)
from litellm.proxy.gateway.mcp.outbound_credentials.token_exchanger import (
    TokenExchanger,
)
from litellm.proxy.gateway.mcp.outbound_credentials.token_refresher import (
    TokenRefresher,
)
from litellm.proxy.gateway.mcp.outbound_credentials.token_store import (
    InMemoryTokenStore,
    StoredToken,
    TokenKey,
    TokenStore,
)
from litellm.proxy.gateway.mcp.outbound_credentials.types import (
    Ambient,
    ApiKeyConfig,
    AssumeRole,
    AuthorizationCodeConfig,
    AuthSpecKind,
    AwsSigV4Config,
    Byok,
    ClientCredentialsConfig,
    CredError,
    NoneConfig,
    PassthroughConfig,
    PerUserEnvVar,
    ServerSpec,
    SharedKey,
    StaticKeys,
    Subject,
    TokenExchangeConfig,
)
from litellm.proxy.gateway.mcp.result import Error, Ok, Result

RESOURCE = "https://up.example/mcp"
NOW = datetime(2026, 6, 17, 12, 0, 0, tzinfo=timezone.utc)
SUBJECT = Subject(tenant_id="t1", subject_id="u1")


class FixedClock:
    def now(self) -> datetime:
        return NOW


class FakeRefresher:
    def __init__(self, result: Result[StoredToken, CredError]) -> None:
        self._result = result

    async def refresh(
        self, config: AuthorizationCodeConfig, refresh_token: SecretStr
    ) -> Result[StoredToken, CredError]:
        return self._result


class FailingCredentialStore:
    async def get(self, key: CredentialKey) -> Result[str | None, CredError]:
        return Error(CredError.of_upstream_unavailable("credential store down"))


class FailingTokenStore:
    async def get(self, key: TokenKey) -> Result[StoredToken | None, CredError]:
        return Error(CredError.of_upstream_unavailable("token store down"))

    async def put(self, key: TokenKey, token: StoredToken) -> Result[None, CredError]:
        return Error(CredError.of_upstream_unavailable("token store down"))


class FakeFetcher:
    def __init__(self, result: Result[StoredToken, CredError]) -> None:
        self._result = result
        self.calls = 0

    async def fetch(
        self, config: ClientCredentialsConfig
    ) -> Result[StoredToken, CredError]:
        self.calls += 1
        return self._result


class FlakyServiceTokenStore:
    """A ServiceTokenStore that can be told to fail reads and/or writes."""

    def __init__(self, *, fail_get: bool = False, fail_put: bool = False) -> None:
        self._fail_get = fail_get
        self._fail_put = fail_put
        self._tokens: dict[ServiceTokenKey, StoredToken] = {}  # mutable-ok: test double

    async def get(self, key: ServiceTokenKey) -> Result[StoredToken | None, CredError]:
        if self._fail_get:
            return Error(CredError.of_upstream_unavailable("cache read down"))
        return Ok(self._tokens.get(key))

    async def put(
        self, key: ServiceTokenKey, token: StoredToken
    ) -> Result[None, CredError]:
        if self._fail_put:
            return Error(CredError.of_upstream_unavailable("cache write down"))
        self._tokens[key] = token  # mutable-ok: test double
        return Ok(None)


class FakeExchanger:
    def __init__(self, result: Result[StoredToken, CredError]) -> None:
        self._result = result
        self.calls = 0
        self.received_subject_token: str | None = None
        self.received_resource: str | None = None

    async def exchange(
        self, config: TokenExchangeConfig, subject_token: SecretStr, resource: str
    ) -> Result[StoredToken, CredError]:
        self.calls += 1
        self.received_subject_token = subject_token.get_secret_value()
        self.received_resource = resource
        return self._result


class FakeSignerFactory:
    def __init__(self, result: Result[httpx.Auth, CredError]) -> None:
        self._result = result
        self.calls = 0

    async def build(self, config: AwsSigV4Config) -> Result[httpx.Auth, CredError]:
        self.calls += 1
        return self._result


def _provider(
    *,
    credential_store: CredentialStore | None = None,
    token_store: TokenStore | None = None,
    refresher: TokenRefresher | None = None,
    clock: Clock | None = None,
    service_token_store: ServiceTokenStore | None = None,
    fetcher: ClientCredentialsFetcher | None = None,
    token_exchanger: TokenExchanger | None = None,
    signer_factory: SignerFactory | None = None,
) -> UpstreamCredentialProvider:
    return UpstreamCredentialProvider(
        credential_store=credential_store or InMemoryCredentialStore(),
        token_store=token_store or InMemoryTokenStore(),
        token_refresher=refresher
        or FakeRefresher(Error(CredError.of_upstream_unavailable("unused"))),
        clock=clock or FixedClock(),
        service_token_store=service_token_store or InMemoryServiceTokenStore(),
        client_credentials_fetcher=fetcher
        or FakeFetcher(Error(CredError.of_upstream_unavailable("unused"))),
        token_exchanger=token_exchanger
        or FakeExchanger(Error(CredError.of_upstream_unavailable("unused"))),
        signer_factory=signer_factory
        or FakeSignerFactory(Error(CredError.of_upstream_unavailable("unused"))),
    )


PROVIDER = _provider()


def _spec(config: object) -> ServerSpec:
    return ServerSpec(server_id="s1", resource=RESOURCE, config=config)  # type: ignore[arg-type]


def _token_key() -> TokenKey:
    return TokenKey(tenant_id="t1", subject_id="u1", server_id="s1", resource=RESOURCE)


def _authz_cfg() -> AuthorizationCodeConfig:
    return AuthorizationCodeConfig(
        client_id="c",
        client_secret="s",
        authorization_url="https://idp/auth",
        token_url="https://idp/token",
    )


def _cc_cfg() -> ClientCredentialsConfig:
    return ClientCredentialsConfig(
        client_id="c", client_secret="s", token_url="https://idp/token"
    )


def _svc_key() -> ServiceTokenKey:
    return ServiceTokenKey(server_id="s1", resource=RESOURCE)


def _tx_cfg() -> TokenExchangeConfig:
    return TokenExchangeConfig()


def _aws_cfg() -> AwsSigV4Config:
    return AwsSigV4Config(region="us-east-1")


def _applied_headers(auth: httpx.Auth) -> httpx.Headers:
    request = httpx.Request("POST", "https://up.example/mcp")
    return next(auth.auth_flow(request)).headers


def test_auth_spec_kind_is_derived_from_config():
    spec = _spec(ApiKeyConfig(key_source=SharedKey(value="k")))
    assert spec.auth_spec_kind is AuthSpecKind.api_key


def test_discriminated_union_rejects_config_missing_required_fields():
    # client_credentials requires client_id/secret/token_url; an empty body must fail.
    with pytest.raises(ValidationError):
        _spec({"kind": "client_credentials"})


def test_authorization_code_config_allows_discovery_defaults():
    # Discovery + DCR: endpoints and client creds are obtained at runtime, so an empty
    # authorization_code config is valid - the fields are optional manual overrides.
    spec = _spec({"kind": "authorization_code"})
    assert isinstance(spec.config, AuthorizationCodeConfig)


def test_discriminated_union_picks_the_variant_by_kind():
    spec = _spec({"kind": "none"})
    assert isinstance(spec.config, NoneConfig)


async def test_none_attaches_no_credential():
    result = await PROVIDER.resolve(SUBJECT, _spec(NoneConfig()))
    assert isinstance(result, Ok)
    assert isinstance(result.ok, NoOpAuth)
    assert "Authorization" not in _applied_headers(result.ok)


@pytest.mark.parametrize(
    "header_name,value_prefix,expected_header,expected_value",
    [
        ("Authorization", "Bearer", "Authorization", "Bearer k"),  # v1 bearer_token
        ("Authorization", "Basic", "Authorization", "Basic k"),  # v1 basic
        ("Authorization", "token", "Authorization", "token k"),  # v1 token
        ("Authorization", "", "Authorization", "k"),  # v1 authorization (raw)
        ("X-API-Key", "", "X-API-Key", "k"),  # v1 api_key (its own header, raw)
        ("Ocp-Apim-Subscription-Key", "", "Ocp-Apim-Subscription-Key", "k"),  # custom
    ],
)
async def test_api_key_writes_value_to_the_configured_header(
    header_name: str, value_prefix: str, expected_header: str, expected_value: str
):
    config = ApiKeyConfig(
        header_name=header_name,
        value_prefix=value_prefix,
        key_source=SharedKey(value="k"),
    )
    result = await PROVIDER.resolve(SUBJECT, _spec(config))
    assert isinstance(result, Ok)
    assert isinstance(result.ok, StaticHeaderAuth)
    assert _applied_headers(result.ok)[expected_header] == expected_value


def test_secret_fields_are_masked_in_serialization():
    # SecretStr keeps the value out of model_dump / repr / logs but usable in the resolver.
    config = ApiKeyConfig(key_source=SharedKey(value="SUPER-SECRET"))
    dumped = config.model_dump_json()
    assert "SUPER-SECRET" not in dumped
    assert "**********" in dumped
    assert config.key_source.value.get_secret_value() == "SUPER-SECRET"


@pytest.mark.parametrize("source", [Byok(), PerUserEnvVar()])
async def test_api_key_per_user_pulls_the_subject_credential(source: object):
    store = InMemoryCredentialStore(
        {CredentialKey(tenant_id="t1", subject_id="u1", server_id="s1"): "user-secret"}
    )
    provider = _provider(credential_store=store)
    result = await provider.resolve(SUBJECT, _spec(ApiKeyConfig(key_source=source)))  # type: ignore[arg-type]
    assert isinstance(result, Ok)
    assert _applied_headers(result.ok)["Authorization"] == "Bearer user-secret"


async def test_api_key_byok_missing_returns_unauthorized():
    # Missing BYOK credential -> 401 + WWW-Authenticate (the user must provide it).
    result = await PROVIDER.resolve(SUBJECT, _spec(ApiKeyConfig(key_source=Byok())))
    assert isinstance(result, Error)
    assert result.error.tag == "unauthorized"


async def test_api_key_env_var_missing_returns_precondition_required():
    # Missing per-user env var -> 412 (a setup precondition), distinct from BYOK's 401.
    result = await PROVIDER.resolve(
        SUBJECT, _spec(ApiKeyConfig(key_source=PerUserEnvVar()))
    )
    assert isinstance(result, Error)
    assert result.error.tag == "precondition_required"


async def test_api_key_per_user_isolated_by_subject():
    # The stored key belongs to (t1,u1,s1); a different subject must not receive it.
    store = InMemoryCredentialStore(
        {CredentialKey(tenant_id="t1", subject_id="u1", server_id="s1"): "u1-secret"}
    )
    provider = _provider(credential_store=store)
    other = Subject(tenant_id="t1", subject_id="u2")
    result = await provider.resolve(other, _spec(ApiKeyConfig(key_source=Byok())))
    assert isinstance(result, Error)
    assert result.error.tag == "unauthorized"


def test_crederror_maps_to_distinct_http_statuses():
    # Each failure class surfaces its own HTTP status at the edge.
    assert http_status(CredError.of_unauthorized("byok missing")) == 401
    assert http_status(CredError.of_precondition_required("env var missing")) == 412
    assert http_status(CredError.of_not_implemented("stub arm")) == 501


async def test_passthrough_forwards_the_inbound_token():
    subject = Subject(tenant_id="t1", subject_id="u1", inbound_token="upstream-tok")
    result = await PROVIDER.resolve(subject, _spec(PassthroughConfig()))
    assert isinstance(result, Ok)
    assert _applied_headers(result.ok)["Authorization"] == "Bearer upstream-tok"


async def test_passthrough_without_a_token_fails_closed():
    result = await PROVIDER.resolve(SUBJECT, _spec(PassthroughConfig()))
    assert isinstance(result, Error)
    assert result.error.tag == "unauthorized"


async def test_self_contained_arms_never_read_the_inbound_token():
    # The #30559 guard: none/api_key must produce the identical credential whether or not a
    # caller bearer is present, proving they never forward the gateway-bound token.
    with_token = Subject(tenant_id="t1", subject_id="u1", inbound_token="leak-me")
    for config in (NoneConfig(), ApiKeyConfig(key_source=SharedKey(value="k"))):
        without = await PROVIDER.resolve(SUBJECT, _spec(config))
        present = await PROVIDER.resolve(with_token, _spec(config))
        assert isinstance(without, Ok) and isinstance(present, Ok)
        assert _applied_headers(without.ok).get("Authorization") == _applied_headers(
            present.ok
        ).get("Authorization")


async def test_authorization_code_returns_a_valid_stored_token():
    store = InMemoryTokenStore(
        {
            _token_key(): StoredToken(
                access_token="valid", expires_at=NOW + timedelta(hours=1)
            )
        }
    )
    result = await _provider(token_store=store).resolve(SUBJECT, _spec(_authz_cfg()))
    assert isinstance(result, Ok)
    assert _applied_headers(result.ok)["Authorization"] == "Bearer valid"


async def test_authorization_code_without_a_token_fails_closed():
    # No stored token -> unauthorized, which the edge turns into the 401 that starts the dance.
    result = await _provider().resolve(SUBJECT, _spec(_authz_cfg()))
    assert isinstance(result, Error)
    assert result.error.tag == "unauthorized"


async def test_authorization_code_expired_without_refresh_fails_closed():
    store = InMemoryTokenStore(
        {
            _token_key(): StoredToken(
                access_token="old", expires_at=NOW - timedelta(minutes=1)
            )
        }
    )
    result = await _provider(token_store=store).resolve(SUBJECT, _spec(_authz_cfg()))
    assert isinstance(result, Error)
    assert result.error.tag == "unauthorized"


async def test_authorization_code_refreshes_proactively_near_expiry():
    store = InMemoryTokenStore(
        {
            _token_key(): StoredToken(
                access_token="old",
                expires_at=NOW + timedelta(seconds=30),  # within the 60s refresh buffer
                refresh_token="r",
            )
        }
    )
    fresh = StoredToken(
        access_token="new", expires_at=NOW + timedelta(hours=1), refresh_token="r2"
    )
    provider = _provider(token_store=store, refresher=FakeRefresher(Ok(fresh)))
    result = await provider.resolve(SUBJECT, _spec(_authz_cfg()))
    assert isinstance(result, Ok)
    assert _applied_headers(result.ok)["Authorization"] == "Bearer new"
    persisted = await store.get(_token_key())
    assert isinstance(persisted, Ok)
    assert persisted.ok is not None
    assert persisted.ok.access_token.get_secret_value() == "new"


async def test_authorization_code_refresh_rejected_fails_closed():
    store = InMemoryTokenStore(
        {
            _token_key(): StoredToken(
                access_token="old",
                expires_at=NOW - timedelta(minutes=1),
                refresh_token="r",
            )
        }
    )
    provider = _provider(
        token_store=store,
        refresher=FakeRefresher(Error(CredError.of_unauthorized("refresh revoked"))),
    )
    result = await provider.resolve(SUBJECT, _spec(_authz_cfg()))
    assert isinstance(result, Error)
    assert result.error.tag == "unauthorized"


async def test_authorization_code_refresh_unreachable_is_upstream_unavailable():
    store = InMemoryTokenStore(
        {
            _token_key(): StoredToken(
                access_token="old",
                expires_at=NOW - timedelta(minutes=1),
                refresh_token="r",
            )
        }
    )
    provider = _provider(
        token_store=store,
        refresher=FakeRefresher(
            Error(CredError.of_upstream_unavailable("token endpoint timeout"))
        ),
    )
    result = await provider.resolve(SUBJECT, _spec(_authz_cfg()))
    assert isinstance(result, Error)
    assert result.error.tag == "upstream_unavailable"


def test_stored_token_secrets_are_masked():
    token = StoredToken(
        access_token="ACCESS-SECRET", expires_at=NOW, refresh_token="REFRESH-SECRET"
    )
    dumped = token.model_dump_json()
    assert "ACCESS-SECRET" not in dumped
    assert "REFRESH-SECRET" not in dumped
    assert "**********" in dumped


async def test_client_credentials_uses_cached_fresh_token():
    store = InMemoryServiceTokenStore(
        {
            _svc_key(): StoredToken(
                access_token="svc", expires_at=NOW + timedelta(hours=1)
            )
        }
    )
    fetcher = FakeFetcher(
        Error(CredError.of_upstream_unavailable("must not be called"))
    )
    result = await _provider(service_token_store=store, fetcher=fetcher).resolve(
        SUBJECT, _spec(_cc_cfg())
    )
    assert isinstance(result, Ok)
    assert _applied_headers(result.ok)["Authorization"] == "Bearer svc"
    assert fetcher.calls == 0


async def test_client_credentials_mints_and_caches_on_miss():
    store = InMemoryServiceTokenStore()
    minted = StoredToken(access_token="fresh", expires_at=NOW + timedelta(hours=1))
    result = await _provider(
        service_token_store=store, fetcher=FakeFetcher(Ok(minted))
    ).resolve(SUBJECT, _spec(_cc_cfg()))
    assert isinstance(result, Ok)
    assert _applied_headers(result.ok)["Authorization"] == "Bearer fresh"
    cached = await store.get(_svc_key())
    assert isinstance(cached, Ok) and cached.ok is not None
    assert cached.ok.access_token.get_secret_value() == "fresh"


async def test_client_credentials_remints_near_expiry():
    store = InMemoryServiceTokenStore(
        {
            _svc_key(): StoredToken(
                access_token="old", expires_at=NOW + timedelta(seconds=30)
            )
        }
    )
    fetcher = FakeFetcher(
        Ok(StoredToken(access_token="new", expires_at=NOW + timedelta(hours=1)))
    )
    result = await _provider(service_token_store=store, fetcher=fetcher).resolve(
        SUBJECT, _spec(_cc_cfg())
    )
    assert isinstance(result, Ok)
    assert _applied_headers(result.ok)["Authorization"] == "Bearer new"
    assert fetcher.calls == 1


async def test_client_credentials_rejected_grant_is_misconfigured():
    fetcher = FakeFetcher(Error(CredError.of_misconfigured("invalid_client")))
    result = await _provider(fetcher=fetcher).resolve(SUBJECT, _spec(_cc_cfg()))
    assert isinstance(result, Error)
    assert result.error.tag == "misconfigured"


async def test_client_credentials_endpoint_down_is_upstream_unavailable():
    fetcher = FakeFetcher(
        Error(CredError.of_upstream_unavailable("token endpoint timeout"))
    )
    result = await _provider(fetcher=fetcher).resolve(SUBJECT, _spec(_cc_cfg()))
    assert isinstance(result, Error)
    assert result.error.tag == "upstream_unavailable"


async def test_client_credentials_never_reads_inbound_token():
    # M2M must never forward the caller bearer: the result is identical with or without one.
    minted = StoredToken(access_token="svc", expires_at=NOW + timedelta(hours=1))
    provider = _provider(fetcher=FakeFetcher(Ok(minted)))
    without = await provider.resolve(SUBJECT, _spec(_cc_cfg()))
    with_token = Subject(tenant_id="t1", subject_id="u1", inbound_token="leak-me")
    present = await provider.resolve(with_token, _spec(_cc_cfg()))
    assert isinstance(without, Ok) and isinstance(present, Ok)
    assert (
        _applied_headers(without.ok)["Authorization"]
        == _applied_headers(present.ok)["Authorization"]
        == "Bearer svc"
    )


async def test_client_credentials_shared_across_subjects():
    # Keyed by (server, resource) with no subject: every subject gets the same token.
    store = InMemoryServiceTokenStore(
        {
            _svc_key(): StoredToken(
                access_token="shared", expires_at=NOW + timedelta(hours=1)
            )
        }
    )
    provider = _provider(service_token_store=store)
    u1 = await provider.resolve(SUBJECT, _spec(_cc_cfg()))
    u2 = await provider.resolve(
        Subject(tenant_id="t1", subject_id="u2"), _spec(_cc_cfg())
    )
    assert isinstance(u1, Ok) and isinstance(u2, Ok)
    assert (
        _applied_headers(u1.ok)["Authorization"]
        == _applied_headers(u2.ok)["Authorization"]
    )


async def test_client_credentials_cache_read_failure_degrades_to_mint():
    # A cache read outage must not fail the request; we just mint fresh.
    minted = StoredToken(access_token="fresh", expires_at=NOW + timedelta(hours=1))
    provider = _provider(
        service_token_store=FlakyServiceTokenStore(fail_get=True),
        fetcher=FakeFetcher(Ok(minted)),
    )
    result = await provider.resolve(SUBJECT, _spec(_cc_cfg()))
    assert isinstance(result, Ok)
    assert _applied_headers(result.ok)["Authorization"] == "Bearer fresh"


async def test_client_credentials_cache_write_failure_is_best_effort():
    # A cache write outage must not fail a valid mint; we return it and skip caching.
    minted = StoredToken(access_token="fresh", expires_at=NOW + timedelta(hours=1))
    provider = _provider(
        service_token_store=FlakyServiceTokenStore(fail_put=True),
        fetcher=FakeFetcher(Ok(minted)),
    )
    result = await provider.resolve(SUBJECT, _spec(_cc_cfg()))
    assert isinstance(result, Ok)
    assert _applied_headers(result.ok)["Authorization"] == "Bearer fresh"


async def test_token_exchange_swaps_inbound_for_an_upstream_token():
    # Core invariant: the inbound token goes to the exchanger; the upstream gets the EXCHANGED
    # token bound to server.resource, never the inbound one.
    subject = Subject(tenant_id="t1", subject_id="u1", inbound_token="user-idp-jwt")
    exchanger = FakeExchanger(
        Ok(StoredToken(access_token="exchanged", expires_at=NOW + timedelta(hours=1)))
    )
    result = await _provider(token_exchanger=exchanger).resolve(
        subject, _spec(_tx_cfg())
    )
    assert isinstance(result, Ok)
    header = _applied_headers(result.ok)["Authorization"]
    assert header == "Bearer exchanged"
    assert "user-idp-jwt" not in header  # inbound is never forwarded to the upstream
    assert (
        exchanger.received_subject_token == "user-idp-jwt"
    )  # it went to the exchanger
    assert exchanger.received_resource == RESOURCE  # bound to server.resource


async def test_token_exchange_without_inbound_token_fails_closed():
    result = await _provider().resolve(SUBJECT, _spec(_tx_cfg()))
    assert isinstance(result, Error)
    assert result.error.tag == "unauthorized"


async def test_token_exchange_uses_cached_fresh_token():
    store = InMemoryTokenStore(
        {
            _token_key(): StoredToken(
                access_token="cached", expires_at=NOW + timedelta(hours=1)
            )
        }
    )
    exchanger = FakeExchanger(
        Error(CredError.of_upstream_unavailable("must not be called"))
    )
    subject = Subject(tenant_id="t1", subject_id="u1", inbound_token="jwt")
    result = await _provider(token_store=store, token_exchanger=exchanger).resolve(
        subject, _spec(_tx_cfg())
    )
    assert isinstance(result, Ok)
    assert _applied_headers(result.ok)["Authorization"] == "Bearer cached"
    assert exchanger.calls == 0


async def test_token_exchange_remints_near_expiry():
    store = InMemoryTokenStore(
        {
            _token_key(): StoredToken(
                access_token="old", expires_at=NOW + timedelta(seconds=30)
            )
        }
    )
    exchanger = FakeExchanger(
        Ok(StoredToken(access_token="new", expires_at=NOW + timedelta(hours=1)))
    )
    subject = Subject(tenant_id="t1", subject_id="u1", inbound_token="jwt")
    result = await _provider(token_store=store, token_exchanger=exchanger).resolve(
        subject, _spec(_tx_cfg())
    )
    assert isinstance(result, Ok)
    assert _applied_headers(result.ok)["Authorization"] == "Bearer new"
    assert exchanger.calls == 1


async def test_token_exchange_subject_token_invalid_is_unauthorized():
    exchanger = FakeExchanger(Error(CredError.of_unauthorized("invalid subject_token")))
    subject = Subject(tenant_id="t1", subject_id="u1", inbound_token="stale")
    result = await _provider(token_exchanger=exchanger).resolve(
        subject, _spec(_tx_cfg())
    )
    assert isinstance(result, Error)
    assert result.error.tag == "unauthorized"


async def test_token_exchange_config_error_is_misconfigured():
    exchanger = FakeExchanger(
        Error(CredError.of_misconfigured("IdP lacks token exchange"))
    )
    subject = Subject(tenant_id="t1", subject_id="u1", inbound_token="jwt")
    result = await _provider(token_exchanger=exchanger).resolve(
        subject, _spec(_tx_cfg())
    )
    assert isinstance(result, Error)
    assert result.error.tag == "misconfigured"


async def test_token_exchange_endpoint_down_is_upstream_unavailable():
    exchanger = FakeExchanger(
        Error(CredError.of_upstream_unavailable("exchange endpoint timeout"))
    )
    subject = Subject(tenant_id="t1", subject_id="u1", inbound_token="jwt")
    result = await _provider(token_exchanger=exchanger).resolve(
        subject, _spec(_tx_cfg())
    )
    assert isinstance(result, Error)
    assert result.error.tag == "upstream_unavailable"


async def test_token_exchange_cache_failure_degrades_to_exchange():
    # Same TokenStore as authorization_code, but token_exchange treats it as an optimization:
    # a read/write outage degrades to a fresh exchange rather than failing (token re-exchangeable).
    exchanger = FakeExchanger(
        Ok(StoredToken(access_token="fresh", expires_at=NOW + timedelta(hours=1)))
    )
    subject = Subject(tenant_id="t1", subject_id="u1", inbound_token="jwt")
    result = await _provider(
        token_store=FailingTokenStore(), token_exchanger=exchanger
    ).resolve(subject, _spec(_tx_cfg()))
    assert isinstance(result, Ok)
    assert _applied_headers(result.ok)["Authorization"] == "Bearer fresh"


def test_token_exchange_config_allows_discovery_defaults():
    spec = _spec({"kind": "token_exchange"})
    assert isinstance(spec.config, TokenExchangeConfig)


async def test_aws_sigv4_returns_the_signer():
    signer = StaticHeaderAuth("AWS4-HMAC-SHA256 Credential=AKIA/...")
    factory = FakeSignerFactory(Ok(signer))
    result = await _provider(signer_factory=factory).resolve(SUBJECT, _spec(_aws_cfg()))
    assert isinstance(result, Ok)
    assert _applied_headers(result.ok)["Authorization"].startswith("AWS4-HMAC-SHA256")
    assert factory.calls == 1


async def test_aws_sigv4_config_error_is_misconfigured():
    factory = FakeSignerFactory(Error(CredError.of_misconfigured("role not assumable")))
    result = await _provider(signer_factory=factory).resolve(SUBJECT, _spec(_aws_cfg()))
    assert isinstance(result, Error)
    assert result.error.tag == "misconfigured"


async def test_aws_sigv4_sts_unreachable_is_upstream_unavailable():
    factory = FakeSignerFactory(Error(CredError.of_upstream_unavailable("STS timeout")))
    result = await _provider(signer_factory=factory).resolve(SUBJECT, _spec(_aws_cfg()))
    assert isinstance(result, Error)
    assert result.error.tag == "upstream_unavailable"


async def test_aws_sigv4_never_reads_inbound_token():
    # Signs with the gateway's AWS identity; the caller bearer never changes the result.
    factory = FakeSignerFactory(Ok(StaticHeaderAuth("AWS4-HMAC-SHA256 sig")))
    provider = _provider(signer_factory=factory)
    without = await provider.resolve(SUBJECT, _spec(_aws_cfg()))
    with_token = Subject(tenant_id="t1", subject_id="u1", inbound_token="leak-me")
    present = await provider.resolve(with_token, _spec(_aws_cfg()))
    assert isinstance(without, Ok) and isinstance(present, Ok)
    header = _applied_headers(present.ok)["Authorization"]
    assert _applied_headers(without.ok)["Authorization"] == header
    assert "leak-me" not in header


def test_aws_sigv4_credentials_default_to_ambient():
    assert isinstance(AwsSigV4Config(region="us-east-1").credentials, Ambient)


def test_aws_sigv4_credential_sources_select_by_discriminator():
    static = _spec(
        {
            "kind": "aws_sigv4",
            "region": "us-east-1",
            "credentials": {
                "source": "static_keys",
                "access_key_id": "AKIA",
                "secret_access_key": "shhh",
            },
        }
    )
    assert isinstance(static.config, AwsSigV4Config)
    assert isinstance(static.config.credentials, StaticKeys)
    role = _spec(
        {
            "kind": "aws_sigv4",
            "region": "us-east-1",
            "credentials": {
                "source": "assume_role",
                "role_arn": "arn:aws:iam::1:role/r",
            },
        }
    )
    assert isinstance(role.config, AwsSigV4Config)
    assert isinstance(role.config.credentials, AssumeRole)


def test_aws_sigv4_rejects_static_keys_missing_secret():
    # The discriminated source makes illegal combos unrepresentable: static_keys needs a secret.
    with pytest.raises(ValidationError):
        _spec(
            {
                "kind": "aws_sigv4",
                "region": "us-east-1",
                "credentials": {"source": "static_keys", "access_key_id": "AKIA"},
            }
        )


def test_aws_sigv4_static_keys_secret_is_masked():
    config = AwsSigV4Config(
        region="us-east-1",
        credentials=StaticKeys(access_key_id="AKIA", secret_access_key="TOP-SECRET"),
    )
    assert "TOP-SECRET" not in config.model_dump_json()


async def test_api_key_per_user_store_error_is_upstream_unavailable():
    # A store/DB outage on read is distinct from a miss: 503, not the 401/412 of "not found".
    provider = _provider(credential_store=FailingCredentialStore())
    result = await provider.resolve(SUBJECT, _spec(ApiKeyConfig(key_source=Byok())))
    assert isinstance(result, Error)
    assert result.error.tag == "upstream_unavailable"


async def test_authorization_code_store_error_is_upstream_unavailable():
    provider = _provider(token_store=FailingTokenStore())
    result = await provider.resolve(SUBJECT, _spec(_authz_cfg()))
    assert isinstance(result, Error)
    assert result.error.tag == "upstream_unavailable"
