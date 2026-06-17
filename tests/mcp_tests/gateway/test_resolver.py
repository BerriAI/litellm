"""Spec tests for the v2 upstream-credential resolver scaffold.

Clean-room litmus: every case constructs `Subject` / `ServerSpec` directly, with zero v1
fixtures. If an arm could not be exercised without a v1 request object, the seam has leaked.
"""

import httpx
import pytest
from pydantic import ValidationError

from litellm.proxy.gateway.mcp.outbound_credentials.credential_store import (
    CredentialKey,
    InMemoryCredentialStore,
)
from litellm.proxy.gateway.mcp.outbound_credentials.httpx_auth import (
    NoOpAuth,
    StaticHeaderAuth,
)
from litellm.proxy.gateway.mcp.outbound_credentials.types import (
    ApiKeyConfig,
    AuthSpecKind,
    NoneConfig,
    PassthroughConfig,
    PerUserKey,
    ServerSpec,
    SharedKey,
    Subject,
)
from litellm.proxy.gateway.mcp.outbound_credentials.resolver import (
    UpstreamCredentialProvider,
)
from litellm.proxy.gateway.mcp.result import Error, Ok

PROVIDER = UpstreamCredentialProvider(InMemoryCredentialStore())
SUBJECT = Subject(tenant_id="t1", subject_id="u1")


def _spec(config: object) -> ServerSpec:
    return ServerSpec(server_id="s1", resource="https://up.example/mcp", config=config)  # type: ignore[arg-type]


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


def test_none_attaches_no_credential():
    result = PROVIDER.resolve(SUBJECT, _spec(NoneConfig()))
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
def test_api_key_emits_the_right_scheme(scheme: str, expected: str):
    config = ApiKeyConfig(scheme=scheme, key_source=SharedKey(value="k"))  # type: ignore[arg-type]
    result = PROVIDER.resolve(SUBJECT, _spec(config))
    assert isinstance(result, Ok)
    assert isinstance(result.ok, StaticHeaderAuth)
    assert _applied_headers(result.ok)["Authorization"] == expected


def test_api_key_per_user_pulls_the_subject_credential():
    store = InMemoryCredentialStore(
        {CredentialKey(tenant_id="t1", subject_id="u1", server_id="s1"): "user-secret"}
    )
    provider = UpstreamCredentialProvider(store)
    result = provider.resolve(SUBJECT, _spec(ApiKeyConfig(key_source=PerUserKey())))
    assert isinstance(result, Ok)
    assert _applied_headers(result.ok)["Authorization"] == "Bearer user-secret"


def test_api_key_per_user_missing_credential_fails_closed():
    # Empty store -> the per-user arm fails closed rather than sending no/garbage auth.
    result = PROVIDER.resolve(SUBJECT, _spec(ApiKeyConfig(key_source=PerUserKey())))
    assert isinstance(result, Error)
    assert result.error.tag == "unauthorized"


def test_api_key_per_user_isolated_by_subject():
    # The stored key belongs to (t1,u1,s1); a different subject must not receive it.
    store = InMemoryCredentialStore(
        {CredentialKey(tenant_id="t1", subject_id="u1", server_id="s1"): "u1-secret"}
    )
    provider = UpstreamCredentialProvider(store)
    other = Subject(tenant_id="t1", subject_id="u2")
    result = provider.resolve(other, _spec(ApiKeyConfig(key_source=PerUserKey())))
    assert isinstance(result, Error)
    assert result.error.tag == "unauthorized"


def test_passthrough_forwards_the_inbound_token():
    subject = Subject(tenant_id="t1", subject_id="u1", inbound_token="upstream-tok")
    result = PROVIDER.resolve(subject, _spec(PassthroughConfig()))
    assert isinstance(result, Ok)
    assert _applied_headers(result.ok)["Authorization"] == "Bearer upstream-tok"


def test_passthrough_without_a_token_fails_closed():
    result = PROVIDER.resolve(SUBJECT, _spec(PassthroughConfig()))
    assert isinstance(result, Error)
    assert result.error.tag == "unauthorized"


def test_self_contained_arms_never_read_the_inbound_token():
    # The #30559 guard: none/api_key must produce the identical credential whether or not a
    # caller bearer is present, proving they never forward the gateway-bound token.
    with_token = Subject(tenant_id="t1", subject_id="u1", inbound_token="leak-me")
    for config in (NoneConfig(), ApiKeyConfig(key_source=SharedKey(value="k"))):
        without = PROVIDER.resolve(SUBJECT, _spec(config))
        present = PROVIDER.resolve(with_token, _spec(config))
        assert isinstance(without, Ok) and isinstance(present, Ok)
        assert _applied_headers(without.ok).get("Authorization") == _applied_headers(
            present.ok
        ).get("Authorization")


@pytest.mark.parametrize(
    "config",
    [
        {
            "kind": "authorization_code",
            "client_id": "c",
            "client_secret": "s",
            "authorization_url": "https://idp/auth",
            "token_url": "https://idp/token",
        },
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
def test_unimplemented_arms_fail_closed(config: dict):
    result = PROVIDER.resolve(SUBJECT, _spec(config))
    assert isinstance(result, Error)
    assert result.error.tag == "misconfigured"
