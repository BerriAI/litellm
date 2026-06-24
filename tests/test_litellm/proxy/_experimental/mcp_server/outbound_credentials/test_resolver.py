"""Tests for the resolver dispatch: live arms produce auth, stubbed arms fail closed.

`none` and `api_key` (shared-key source) are implemented; every other arm, plus the `api_key`
BYOK source, returns a typed `not_implemented` error until its mode lands. Parametrizing the
stubs over one config each also guards reachability: a dropped `case` would hit `assert_never`
and raise instead of returning the stub.
"""

import base64

import httpx
import pytest
from pydantic import SecretStr

from litellm.proxy._experimental.mcp_server.outbound_credentials import (
    ApiKeyConfig,
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


class _FakeByokStore:
    """A ByokCredentialStore returning a canned per-user key (None == unprovisioned)."""

    def __init__(self, by_user: dict) -> None:
        self._by_user = by_user

    async def fetch(self, user_id: str, server_id: str):
        return self._by_user.get(user_id)


def _byok_spec(header_name="X-API-Key", value_prefix="", encode_base64=False):
    return _spec(
        ApiKeyConfig(
            header_name=header_name,
            value_prefix=value_prefix,
            encode_base64=encode_base64,
            key_source=Byok(),
        )
    )


@pytest.mark.asyncio
async def test_api_key_byok_emits_the_per_user_key():
    provider = UpstreamCredentialProvider(
        byok_store=_FakeByokStore({"alice": "k-alice"})
    )
    result = await provider.resolve_credentials(
        Subject(tenant_id="", subject_id="alice"), _byok_spec()
    )
    assert isinstance(result, Ok)
    assert _emitted(result.ok)["X-API-Key"] == "k-alice"


@pytest.mark.asyncio
async def test_api_key_byok_honors_the_configured_scheme():
    provider = UpstreamCredentialProvider(
        byok_store=_FakeByokStore({"alice": "user:pass"})
    )
    result = await provider.resolve_credentials(
        Subject(tenant_id="", subject_id="alice"),
        _byok_spec(
            header_name="Authorization", value_prefix="Basic", encode_base64=True
        ),
    )
    assert isinstance(result, Ok)
    expected = base64.b64encode(b"user:pass").decode()
    assert _emitted(result.ok)["Authorization"] == f"Basic {expected}"


@pytest.mark.asyncio
async def test_api_key_byok_missing_credential_is_unauthorized():
    provider = UpstreamCredentialProvider(byok_store=_FakeByokStore({}))
    result = await provider.resolve_credentials(
        Subject(tenant_id="", subject_id="alice"), _byok_spec()
    )
    assert isinstance(result, Error)
    assert result.error.tag == "unauthorized"
    # the 401 carries the provisioning challenge, not a bare message
    challenge = result.error.unauthorized
    assert challenge.www_authenticate is not None
    assert challenge.body is not None
    assert challenge.body["error"] == "byok_auth_required"


@pytest.mark.asyncio
async def test_api_key_byok_isolates_by_subject():
    provider = UpstreamCredentialProvider(
        byok_store=_FakeByokStore({"alice": "k-alice"})
    )
    alice = await provider.resolve_credentials(
        Subject(tenant_id="", subject_id="alice"), _byok_spec()
    )
    bob = await provider.resolve_credentials(
        Subject(tenant_id="", subject_id="bob"), _byok_spec()
    )
    assert isinstance(alice, Ok) and _emitted(alice.ok)["X-API-Key"] == "k-alice"
    assert isinstance(bob, Error) and bob.error.tag == "unauthorized"


@pytest.mark.asyncio
async def test_api_key_byok_with_no_store_wired_is_unauthorized():
    result = await UpstreamCredentialProvider().resolve_credentials(
        Subject(tenant_id="", subject_id="alice"), _byok_spec()
    )
    assert isinstance(result, Error)
    assert result.error.tag == "unauthorized"


_STUBBED = [
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
