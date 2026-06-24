"""Tests for the resolver dispatch: live arms produce auth, stubbed arms fail closed.

`none` and `api_key` (shared-key source) are implemented; every other arm, plus the `api_key`
BYOK source, returns a typed `not_implemented` error until its mode lands. Parametrizing the
stubs over one config each also guards reachability: a dropped `case` would hit `assert_never`
and raise instead of returning the stub.
"""

import httpx
import pytest
from pydantic import SecretStr

from litellm.proxy._experimental.mcp_server.outbound_credentials import (
    ApiKeyConfig,
    AuthSpecKind,
    AuthorizationCodeConfig,
    AwsSigV4Config,
    Byok,
    ClientCredentialsConfig,
    Error,
    NoneConfig,
    NoOpAuth,
    Ok,
    PassthroughConfig,
    ServerSpec,
    SharedKey,
    StaticHeaderAuth,
    Subject,
    TokenExchangeConfig,
    UpstreamCredentialProvider,
)

_SUBJECT = Subject(tenant_id="", subject_id="")


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


_STUBBED = [
    ("api_key_byok", ApiKeyConfig(key_source=Byok())),
    ("passthrough", PassthroughConfig()),
    ("client_credentials", ClientCredentialsConfig()),
    ("token_exchange", TokenExchangeConfig()),
    ("authorization_code", AuthorizationCodeConfig()),
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


def test_every_auth_spec_kind_is_exercised():
    """A new AuthSpecKind that ships without a resolver test fails here.

    The resolver's match over the config union is exhaustive at the type level (assert_never), so
    a missing arm fails basedpyright; this guards the parallel gap the type checker cannot see, a
    new mode that ships without a test exercising its arm. The two live arms cover `none` and the
    shared-key `api_key`; `_STUBBED` covers the BYOK `api_key` source and every remaining mode.
    """
    live_kinds = {
        NoneConfig().kind,
        ApiKeyConfig(key_source=SharedKey(value=SecretStr("k"))).kind,
    }
    stubbed_kinds = {config.kind for _, config in _STUBBED}
    assert live_kinds | stubbed_kinds == set(AuthSpecKind)
