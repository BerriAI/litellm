"""The ONE credential resolver: `(subject, server) -> Result[httpx.Auth, CredError]`.

`resolve()` selects exactly one arm off the server's declared per-mode `config` and either
produces an `httpx.Auth` or fails closed with a typed `CredError`. There is no precedence
cascade and no silent downgrade; a header an attacker strips cannot change the mode.

The match is over the `AuthConfig` variant (not a separate enum), so each arm receives its
own fully-typed config with every field guaranteed present — no `None`-checks. The match is
wildcard-free with an `assert_never` tail, so adding a mode without an arm fails the type
gate, and a bypassed gate fails loudly at runtime instead of returning `None`.

Implemented: `none`, `passthrough`, and `api_key` (shared from config, or per-user / BYOK
pulled from the injected `CredentialStore`). The OAuth-flow and signing modes are typed
stubs that fail closed until their collaborators (token store, OAuth providers, RFC 8693
exchanger, SigV4 signer) are injected.
"""

from __future__ import annotations

import httpx
from typing_extensions import assert_never

from ..result import Error, Ok, Result
from .credential_store import CredentialKey, CredentialStore
from .httpx_auth import NoOpAuth, StaticHeaderAuth
from .types import (
    ApiKeyConfig,
    AuthorizationCodeConfig,
    AuthSpecKind,
    AwsSigV4Config,
    ClientCredentialsConfig,
    CredError,
    NoneConfig,
    PassthroughConfig,
    PerUserKey,
    ServerSpec,
    SharedKey,
    Subject,
    TokenExchangeConfig,
)


class UpstreamCredentialProvider:
    """Produces the one `httpx.Auth` for a `(subject, upstream)` pair, per declared mode."""

    def __init__(self, credential_store: CredentialStore) -> None:
        self._credential_store = credential_store

    def resolve(
        self, subject: Subject, server: ServerSpec
    ) -> Result[httpx.Auth, CredError]:
        match server.config:
            case AuthorizationCodeConfig() as config:
                return self._authorization_code(subject, server, config)
            case ClientCredentialsConfig() as config:
                return self._client_credentials(subject, server, config)
            case TokenExchangeConfig() as config:
                return self._token_exchange(subject, server, config)
            case ApiKeyConfig() as config:
                return self._api_key(subject, server, config)
            case PassthroughConfig() as config:
                return self._passthrough(subject, server, config)
            case NoneConfig() as config:
                return self._none(subject, server, config)
            case AwsSigV4Config() as config:
                return self._aws_sigv4(subject, server, config)
        assert_never(server.config)

    # --- implemented arms -----------------------------------------------------------------
    def _none(
        self, subject: Subject, server: ServerSpec, config: NoneConfig
    ) -> Result[httpx.Auth, CredError]:
        return Ok(NoOpAuth())

    def _api_key(
        self, subject: Subject, server: ServerSpec, config: ApiKeyConfig
    ) -> Result[httpx.Auth, CredError]:
        match config.key_source:
            case SharedKey() as source:
                return Ok(StaticHeaderAuth(config.header_for(source.value)))
            case PerUserKey():
                value = self._credential_store.get(
                    CredentialKey(
                        tenant_id=subject.tenant_id,
                        subject_id=subject.subject_id,
                        server_id=server.server_id,
                    )
                )
                if value is None:
                    return Error(
                        CredError.of_unauthorized(
                            "api_key: no per-user credential for this subject"
                        )
                    )
                return Ok(StaticHeaderAuth(config.header_for(value)))
        assert_never(config.key_source)

    def _passthrough(
        self, subject: Subject, server: ServerSpec, config: PassthroughConfig
    ) -> Result[httpx.Auth, CredError]:
        # The one arm that forwards a caller-supplied token, and only one the client obtained
        # for the upstream's audience — never a gateway-bound credential.
        if subject.inbound_token is None:
            return Error(
                CredError.of_unauthorized("passthrough: no inbound token to forward")
            )
        return Ok(StaticHeaderAuth(f"Bearer {subject.inbound_token}"))

    # --- arms awaiting their collaborators (typed stubs, fail closed) ----------------------
    def _authorization_code(
        self, subject: Subject, server: ServerSpec, config: AuthorizationCodeConfig
    ) -> Result[httpx.Auth, CredError]:
        return _todo(AuthSpecKind.authorization_code)

    def _client_credentials(
        self, subject: Subject, server: ServerSpec, config: ClientCredentialsConfig
    ) -> Result[httpx.Auth, CredError]:
        return _todo(AuthSpecKind.client_credentials)

    def _token_exchange(
        self, subject: Subject, server: ServerSpec, config: TokenExchangeConfig
    ) -> Result[httpx.Auth, CredError]:
        return _todo(AuthSpecKind.token_exchange)

    def _aws_sigv4(
        self, subject: Subject, server: ServerSpec, config: AwsSigV4Config
    ) -> Result[httpx.Auth, CredError]:
        return _todo(AuthSpecKind.aws_sigv4)


def _todo(kind: AuthSpecKind) -> Result[httpx.Auth, CredError]:
    return Error(
        CredError.of_misconfigured(f"{kind.value}: resolver arm not implemented yet")
    )
