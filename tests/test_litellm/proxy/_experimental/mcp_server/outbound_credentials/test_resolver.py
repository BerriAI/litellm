"""Tests for the resolver dispatch skeleton.

Every mode must reach its own arm and, until that arm is built, return a typed
`not_implemented` CredError rather than silently producing no credential. Parametrizing over
one config per mode also guards reachability: if a `case` were dropped, that mode would fall to
the `assert_never` tail and raise here instead of returning the stub.
"""

import pytest
from pydantic import SecretStr

from litellm.proxy._experimental.mcp_server.outbound_credentials import (
    ApiKeyConfig,
    AuthorizationCodeConfig,
    AuthSpecKind,
    AwsSigV4Config,
    ClientCredentialsConfig,
    Error,
    NoneConfig,
    PassthroughConfig,
    ServerSpec,
    SharedKey,
    Subject,
    TokenExchangeConfig,
    UpstreamCredentialProvider,
)

_ONE_CONFIG_PER_MODE = [
    (AuthSpecKind.none, NoneConfig()),
    (AuthSpecKind.api_key, ApiKeyConfig(key_source=SharedKey(value=SecretStr("k")))),
    (AuthSpecKind.passthrough, PassthroughConfig()),
    (AuthSpecKind.client_credentials, ClientCredentialsConfig()),
    (AuthSpecKind.token_exchange, TokenExchangeConfig()),
    (AuthSpecKind.authorization_code, AuthorizationCodeConfig()),
    (AuthSpecKind.aws_sigv4, AwsSigV4Config(region="us-east-1")),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("kind, config", _ONE_CONFIG_PER_MODE)
async def test_every_mode_reaches_its_arm_and_returns_not_implemented(kind, config):
    spec = ServerSpec(
        server_id="s", resource="https://upstream.example.com", config=config
    )
    subject = Subject(tenant_id="", subject_id="")

    result = await UpstreamCredentialProvider().resolve_credentials(subject, spec)

    assert isinstance(result, Error)
    assert result.error.tag == "not_implemented"
    assert kind.value in result.error.summary


def test_all_seven_modes_are_covered():
    # Guards that the parametrization (and therefore the dispatch) spans every AuthSpecKind, so a
    # newly added mode without a test row is caught here rather than slipping through.
    assert {kind for kind, _ in _ONE_CONFIG_PER_MODE} == set(AuthSpecKind)
