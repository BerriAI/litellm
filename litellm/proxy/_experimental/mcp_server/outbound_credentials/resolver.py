"""The one credential resolver: dispatch on the declared mode, fail closed.

`resolve_credentials` selects exactly one arm off the server's typed `config` and either
produces an `httpx.Auth` or returns a typed `CredError`. The `match` is over the `AuthConfig`
variant, so each arm receives its own fully-typed config with no field-presence inference and
no precedence cascade. It is wildcard-free with an `assert_never` tail, so adding a mode without
an arm fails the type gate (basedpyright `reportMatchNotExhaustive`); a bypassed gate fails loudly
at runtime instead of returning `None`.

`none` and `api_key` (shared-key source) are live; the remaining arms are `not_implemented`
stubs that each land in a follow-up PR with their injected seam. The self-contained arms read
straight from the config and need no collaborator. Pure v2: no imports from v1.
"""

from __future__ import annotations

import httpx
from typing_extensions import assert_never

from litellm.proxy._experimental.mcp_server.outbound_credentials.httpx_auth import (
    NoOpAuth,
    StaticHeaderAuth,
)
from litellm.proxy._experimental.mcp_server.outbound_credentials.result import (
    Error,
    Ok,
    Result,
)
from litellm.proxy._experimental.mcp_server.outbound_credentials.types import (
    ApiKeyConfig,
    AuthorizationCodeConfig,
    AuthSpecKind,
    AwsSigV4Config,
    Byok,
    ClientCredentialsConfig,
    CredError,
    NoneConfig,
    PassthroughConfig,
    ServerSpec,
    SharedKey,
    Subject,
    TokenExchangeConfig,
)


class UpstreamCredentialProvider:
    """Produces the one `httpx.Auth` for a `(subject, upstream)` pair, per declared mode.

    Collaborators (the per-mode credential stores and token fetchers) are injected as each arm
    is built; the live `none` and `api_key`-shared arms read from the config and need none.
    """

    async def resolve_credentials(
        self, subject: Subject, server: ServerSpec
    ) -> Result[httpx.Auth, CredError]:
        match server.config:
            case NoneConfig():
                return Ok(NoOpAuth())
            case ApiKeyConfig() as config:
                return self._api_key(config)
            case PassthroughConfig():
                return _not_implemented(AuthSpecKind.passthrough)
            case ClientCredentialsConfig():
                return _not_implemented(AuthSpecKind.client_credentials)
            case TokenExchangeConfig():
                return _not_implemented(AuthSpecKind.token_exchange)
            case AuthorizationCodeConfig():
                return _not_implemented(AuthSpecKind.authorization_code)
            case AwsSigV4Config():
                return _not_implemented(AuthSpecKind.aws_sigv4)
        assert_never(server.config)

    def _api_key(self, config: ApiKeyConfig) -> Result[httpx.Auth, CredError]:
        match config.key_source:
            case SharedKey() as source:
                header_name, header_value = config.header(
                    source.value.get_secret_value()
                )
                return Ok(StaticHeaderAuth(header_value, header_name=header_name))
            case Byok():
                # Per-user key pulled from the credential store; lands with that seam.
                return Error(
                    CredError.of_not_implemented(
                        "api_key BYOK source not implemented yet"
                    )
                )
        assert_never(config.key_source)


def _not_implemented(kind: AuthSpecKind) -> Result[httpx.Auth, CredError]:
    return Error(
        CredError.of_not_implemented(f"{kind.value}: resolver arm not implemented yet")
    )
