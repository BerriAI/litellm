"""Spec tests for the v2 upstream-credential resolver scaffold.

Clean-room litmus: every case constructs `Subject` / `ServerSpec` directly, with zero v1
fixtures. If an arm could not be exercised without a v1 request object, the seam has leaked.
"""

from datetime import datetime, timedelta, timezone

import httpx
import pytest
from pydantic import SecretStr, ValidationError

from litellm.proxy.gateway.mcp._spike_exhaustiveness import http_status
from litellm.proxy.gateway.mcp.outbound_credentials.credential_store import (
    CredentialKey,
    InMemoryCredentialStore,
)
from litellm.proxy.gateway.mcp.outbound_credentials.httpx_auth import (
    NoOpAuth,
    StaticHeaderAuth,
)
from litellm.proxy.gateway.mcp.outbound_credentials.resolver import (
    UpstreamCredentialProvider,
)
from litellm.proxy.gateway.mcp.outbound_credentials.token_store import (
    InMemoryTokenStore,
    StoredToken,
    TokenKey,
)
from litellm.proxy.gateway.mcp.outbound_credentials.types import (
    ApiKeyConfig,
    AuthorizationCodeConfig,
    AuthSpecKind,
    Byok,
    CredError,
    NoneConfig,
    PassthroughConfig,
    PerUserEnvVar,
    ServerSpec,
    SharedKey,
    Subject,
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


def _provider(
    *,
    credential_store: InMemoryCredentialStore | None = None,
    token_store: InMemoryTokenStore | None = None,
    refresher: FakeRefresher | None = None,
    clock: FixedClock | None = None,
) -> UpstreamCredentialProvider:
    return UpstreamCredentialProvider(
        credential_store=credential_store or InMemoryCredentialStore(),
        token_store=token_store or InMemoryTokenStore(),
        token_refresher=refresher
        or FakeRefresher(Error(CredError.of_upstream_unavailable("unused"))),
        clock=clock or FixedClock(),
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


def _applied_headers(auth: httpx.Auth) -> httpx.Headers:
    request = httpx.Request("POST", "https://up.example/mcp")
    return next(auth.auth_flow(request)).headers


def test_auth_spec_kind_is_derived_from_config():
    spec = _spec(ApiKeyConfig(key_source=SharedKey(value="k")))
    assert spec.auth_spec_kind is AuthSpecKind.api_key


def test_discriminated_union_rejects_config_missing_required_fields():
    # authorization_code requires client_id/secret/urls; an empty body must fail at construction.
    with pytest.raises(ValidationError):
        _spec({"kind": "authorization_code"})


def test_discriminated_union_picks_the_variant_by_kind():
    spec = _spec({"kind": "none"})
    assert isinstance(spec.config, NoneConfig)


async def test_none_attaches_no_credential():
    result = await PROVIDER.resolve(SUBJECT, _spec(NoneConfig()))
    assert isinstance(result, Ok)
    assert isinstance(result.ok, NoOpAuth)
    assert "Authorization" not in _applied_headers(result.ok)


@pytest.mark.parametrize(
    "scheme,expected",
    [
        ("bearer", "Bearer k"),
        ("apikey", "ApiKey k"),
        ("basic", "Basic k"),
        ("token", "token k"),
        ("raw", "k"),
    ],
)
async def test_api_key_emits_the_right_scheme(scheme: str, expected: str):
    config = ApiKeyConfig(scheme=scheme, key_source=SharedKey(value="k"))  # type: ignore[arg-type]
    result = await PROVIDER.resolve(SUBJECT, _spec(config))
    assert isinstance(result, Ok)
    assert isinstance(result.ok, StaticHeaderAuth)
    assert _applied_headers(result.ok)["Authorization"] == expected


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


@pytest.mark.parametrize(
    "config",
    [
        {
            "kind": "client_credentials",
            "client_id": "c",
            "client_secret": "s",
            "token_url": "https://idp/token",
        },
        {
            "kind": "token_exchange",
            "token_exchange_endpoint": "https://idp/token",
            "audience": "https://up.example",
        },
        {"kind": "aws_sigv4", "region": "us-east-1"},
    ],
)
async def test_unimplemented_arms_fail_closed(config: dict):
    # Stub arms signal not_implemented (-> 501), not misconfigured (-> 500 operator error).
    result = await PROVIDER.resolve(SUBJECT, _spec(config))
    assert isinstance(result, Error)
    assert result.error.tag == "not_implemented"


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
    assert persisted is not None
    assert persisted.access_token.get_secret_value() == "new"


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
