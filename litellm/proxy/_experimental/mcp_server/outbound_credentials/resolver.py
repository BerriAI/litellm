"""The one credential resolver: dispatch on the declared mode, fail closed.

`resolve_credentials` selects exactly one arm off the server's typed `config` and either
produces an `httpx.Auth` or returns a typed `CredError`. The `match` is over the `AuthConfig`
variant, so each arm receives its own fully-typed config with no field-presence inference and
no precedence cascade. It is wildcard-free with an `assert_never` tail, so adding a mode without
an arm fails the type gate (basedpyright `reportMatchNotExhaustive`); a bypassed gate fails loudly
at runtime instead of returning `None`.

`none` and `api_key` are live: the shared-key source reads from the config, the BYOK source
pulls the per-user key from the injected `ByokCredentialStore`. The remaining arms are
`not_implemented` stubs that each land in a follow-up PR with their injected seam. Pure v2:
no imports from v1.
"""

from __future__ import annotations

from typing import Optional

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
from litellm.proxy._experimental.mcp_server.outbound_credentials.seams import (
    ByokCredentialStore,
    ByokStoreUnavailable,
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


class _NullByokStore:
    """Fail-closed default: with no BYOK store wired, every user reads as unprovisioned."""

    async def fetch(self, user_id: str, server_id: str) -> Optional[str]:
        return None


class UpstreamCredentialProvider:
    """Produces the one `httpx.Auth` for a `(subject, upstream)` pair, per declared mode.

    Collaborators (the per-mode credential stores and token fetchers) are injected as each arm
    is built; the live `none` and shared-key `api_key` arms read from the config and need none,
    while the `api_key` BYOK source pulls the per-user key from the injected `ByokCredentialStore`.
    """

    def __init__(self, byok_store: Optional[ByokCredentialStore] = None) -> None:
        self._byok_store: ByokCredentialStore = byok_store or _NullByokStore()

    async def resolve_credentials(
        self, subject: Subject, server: ServerSpec
    ) -> Result[httpx.Auth, CredError]:
        match server.config:
            case NoneConfig():
                return Ok(NoOpAuth())
            case ApiKeyConfig() as config:
                return await self._api_key(config, subject, server)
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

    async def _api_key(
        self, config: ApiKeyConfig, subject: Subject, server: ServerSpec
    ) -> Result[httpx.Auth, CredError]:
        match config.key_source:
            case SharedKey() as source:
                header_name, header_value = config.header(
                    source.value.get_secret_value()
                )
                return Ok(StaticHeaderAuth(header_value, header_name=header_name))
            case Byok():
                try:
                    key = await self._byok_store.fetch(
                        subject.subject_id, server.server_id
                    )
                except ByokStoreUnavailable:
                    # Store unreachable: not cached, and surfaced as the same 401 challenge as a
                    # missing key (v1 parity) so a transient outage does not 500.
                    key = None
                if not key:
                    return Error(_byok_challenge(server.server_id))
                header_name, header_value = config.header(key)
                return Ok(StaticHeaderAuth(header_value, header_name=header_name))
        assert_never(config.key_source)


def _not_implemented(kind: AuthSpecKind) -> Result[httpx.Auth, CredError]:
    return Error(
        CredError.of_not_implemented(f"{kind.value}: resolver arm not implemented yet")
    )


_BYOK_AUTH_MESSAGE = (
    "No stored credential found for this BYOK server. Complete the OAuth authorization flow "
    "to provide your API key."
)
_BYOK_WWW_AUTHENTICATE = (
    'Bearer resource_metadata="/.well-known/oauth-protected-resource"'
)


def _byok_challenge(server_id: str) -> CredError:
    """The 401 a BYOK server returns when the user has not provisioned a key.

    Carries the RFC 9728 ``WWW-Authenticate`` challenge that drives the provisioning flow, plus
    a ``byok_auth_required`` body, reproducing v1's BYOK 401 through the resolver.
    """
    return CredError.of_unauthorized(
        _BYOK_AUTH_MESSAGE,
        www_authenticate=_BYOK_WWW_AUTHENTICATE,
        body={
            "error": "byok_auth_required",
            "server_id": server_id,
            "message": _BYOK_AUTH_MESSAGE,
        },
    )
