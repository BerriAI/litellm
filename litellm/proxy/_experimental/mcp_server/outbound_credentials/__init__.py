"""Typed upstream-credential resolution for MCP servers.

This subpackage houses the typed credential vocabulary and the ``resolve_credentials``
dispatch. A server declares one per-mode config from the ``AuthConfig`` discriminated union;
``UpstreamCredentialProvider.resolve_credentials`` selects one arm and returns an ``httpx.Auth``
or a typed ``CredError``. Failures are modeled as values via :mod:`.result` (``Result[T,
CredError]``) rather than raised, so every seam is total. Nothing here is wired onto a live
request path yet.
"""

from litellm.proxy._experimental.mcp_server.outbound_credentials.httpx_auth import (
    NoOpAuth,
    StaticHeaderAuth,
)
from litellm.proxy._experimental.mcp_server.outbound_credentials.resolver import (
    UpstreamCredentialProvider,
)
from litellm.proxy._experimental.mcp_server.outbound_credentials.result import (
    Error,
    Ok,
    Result,
)
from litellm.proxy._experimental.mcp_server.outbound_credentials.types import (
    Ambient,
    ApiKeyConfig,
    ApiKeySource,
    AssumeRole,
    AuthConfig,
    AuthorizationCodeConfig,
    AuthSpecKind,
    AwsCredentialSource,
    AwsSigV4Config,
    Byok,
    ClientAuth,
    ClientCredentialsConfig,
    ClientSecretAuth,
    CredError,
    IdJagConfig,
    NoneConfig,
    PassthroughConfig,
    PrivateKeyJwtAuth,
    ServerSpec,
    SharedKey,
    StaticKeys,
    Subject,
    TokenExchangeConfig,
    parse_auth_spec_kind,
)

__all__ = [
    "Ok",
    "Error",
    "Result",
    "NoOpAuth",
    "StaticHeaderAuth",
    "UpstreamCredentialProvider",
    "AuthSpecKind",
    "CredError",
    "Subject",
    "ServerSpec",
    "AuthConfig",
    "parse_auth_spec_kind",
    "AuthorizationCodeConfig",
    "ClientCredentialsConfig",
    "TokenExchangeConfig",
    "IdJagConfig",
    "ClientAuth",
    "PrivateKeyJwtAuth",
    "ClientSecretAuth",
    "ApiKeyConfig",
    "ApiKeySource",
    "SharedKey",
    "Byok",
    "PassthroughConfig",
    "NoneConfig",
    "AwsSigV4Config",
    "AwsCredentialSource",
    "StaticKeys",
    "AssumeRole",
    "Ambient",
]
