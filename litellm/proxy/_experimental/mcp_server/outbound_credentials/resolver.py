"""The one credential resolver: dispatch on the declared mode, fail closed.

`resolve_credentials` selects exactly one arm off the server's typed `config` and either
produces an `httpx.Auth` or returns a typed `CredError`. The `match` is over the `AuthConfig`
variant, so each arm receives its own fully-typed config with no field-presence inference and
no precedence cascade. It is wildcard-free with an `assert_never` tail, so adding a mode without
an arm fails the type gate (basedpyright `reportMatchNotExhaustive`); a bypassed gate fails loudly
at runtime instead of returning `None`.

This skeleton ships every arm as a `not_implemented` stub. Each mode's real body, with its
injected seam, lands in its own follow-up PR; until then the arm returns a typed error rather
than silently producing no credential. Pure v2: no imports from v1.
"""

from __future__ import annotations

import httpx
from typing_extensions import assert_never

from litellm.proxy._experimental.mcp_server.outbound_credentials.result import (
    Error,
    Result,
)
from litellm.proxy._experimental.mcp_server.outbound_credentials.types import (
    ApiKeyConfig,
    AuthorizationCodeConfig,
    AuthSpecKind,
    AwsSigV4Config,
    ClientCredentialsConfig,
    CredError,
    NoneConfig,
    PassthroughConfig,
    ServerSpec,
    Subject,
    TokenExchangeConfig,
)


class UpstreamCredentialProvider:
    """Produces the one `httpx.Auth` for a `(subject, upstream)` pair, per declared mode.

    Collaborators (the per-mode credential stores and token fetchers) are injected as each arm
    is built; the skeleton needs none, since every arm is a stub.
    """

    async def resolve_credentials(
        self, subject: Subject, server: ServerSpec
    ) -> Result[httpx.Auth, CredError]:
        match server.config:
            case NoneConfig():
                return _not_implemented(AuthSpecKind.none)
            case ApiKeyConfig():
                return _not_implemented(AuthSpecKind.api_key)
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


def _not_implemented(kind: AuthSpecKind) -> Result[httpx.Auth, CredError]:
    return Error(
        CredError.of_not_implemented(f"{kind.value}: resolver arm not implemented yet")
    )
