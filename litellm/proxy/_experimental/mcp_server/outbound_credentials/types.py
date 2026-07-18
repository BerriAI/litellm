"""The upstream-credential vocabulary — the typed seam the resolver dispatches on.

This module ships the data types only; the resolver lands in a later PR. It is the contract
the credential build implements and the spec tests assert against.

Design invariants encoded here:

- **Mode is the single source of truth.** A server declares exactly one per-mode `config`
  (the `AuthConfig` discriminated union); `auth_spec_kind` is *derived* from it, never a
  second field that can drift. The resolver dispatches on the config variant, one arm per
  mode. No field-presence inference, no precedence cascade.
- **Illegal states unrepresentable.** Each mode's config is its own frozen model holding
  only that mode's fields — an `aws_sigv4` server cannot hold OAuth fields, and a config
  missing a required field is rejected at construction, not at call time.
- **Fail-closed at the boundary.** A raw mode string can only enter through
  `parse_auth_spec_kind()`, which returns a typed `CredError`.
- **Errors as values.** Every seam returns `Result[_, CredError]`; only edge adapters raise.
- **No v1 imports.** This vocabulary stays free of `MCPServer` and the rest of v1; the
  v1 -> v2 adapter maps onto these types in a later PR.

Sum types are Expression `@tagged_union`s discriminated on a `Literal` `tag`, matched via
`self.tag` with an `assert_never` tail; `Result` is this package's vendored `Ok | Error`
union (see `result.py`), not `expression.Result`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Annotated, Literal

from expression import case, tag, tagged_union
from pydantic import BaseModel, ConfigDict, Field, SecretStr
from typing_extensions import assert_never

from litellm.proxy._experimental.mcp_server.outbound_credentials.result import (
    Error,
    Ok,
    Result,
)
from litellm.types.mcp import DEFAULT_SUBJECT_TOKEN_TYPE


class AuthSpecKind(str, Enum):
    """The server's statically-declared upstream-auth mode — derived from its `config`.

    Covers v1's full `MCPAuth` surface, not only OAuth grants: the three grant modes, the
    collapsed static-header family, client passthrough, no-auth, and AWS request signing.
    BYOK is *not* a member: it is the `api_key` mode seeded per-user, a source selector
    inside that arm. The static-header schemes v1 splits into separate `MCPAuth` values
    (`bearer_token`/`api_key`/`basic`/`token`/`authorization`) collapse into `api_key`; the
    scheme is a parameter the arm carries, not its own mode.
    """

    authorization_code = "authorization_code"  # per-user 3LO; gateway-stored token
    client_credentials = "client_credentials"  # gateway service account (M2M)
    token_exchange = "token_exchange"  # RFC 8693: token endpoint + subject_token (OBO)
    api_key = "api_key"  # static header, any scheme (BYOK = per-user-seeded source)
    passthrough = "passthrough"  # client forwards an upstream-audience token
    none = "none"  # no upstream credential; resolve yields a no-op auth, never an error
    aws_sigv4 = "aws_sigv4"  # AWS SigV4 per-request signing (e.g. Bedrock AgentCore)


@dataclass(frozen=True, slots=True)
class Unauthorized:
    """A 401 plus the optional challenge a client needs to recover.

    ``detail`` is the human message; ``www_authenticate`` and ``body`` carry a scheme-specific
    challenge (e.g. BYOK's provisioning prompt) so the edge can reproduce it verbatim.
    ``claims`` carries an IdP step-up challenge (e.g. Entra Conditional Access) so the edge can
    fold it into the ``WWW-Authenticate`` it builds; the client replays the claims to the IdP to
    satisfy the step-up, then retries with the fresh token.
    """

    detail: str
    www_authenticate: str | None = None
    body: Mapping[str, str] | None = None
    claims: str | None = None


@tagged_union(frozen=True)
class CredError:
    """Why a credential could not be produced. Fail-closed: an arm yields this or an `httpx.Auth`.

    Discriminated on the `Literal` `tag`; consumers `match self.tag` (see `summary`) so the
    type checker can prove exhaustiveness. Construct via the `of_*` factories.
    """

    tag: Literal[
        "unauthorized",
        "misconfigured",
        "upstream_unavailable",
        "unsupported_mode",
        "precondition_required",
        "not_implemented",
    ] = tag()

    unauthorized: Unauthorized = case()  # no usable credential for this (subject, server) -> 401 challenge
    misconfigured: str = case()  # the declared mode is missing required config -> 5xx (operator)
    upstream_unavailable: str = case()  # the IdP / token endpoint could not be reached -> 503
    unsupported_mode: str = case()  # a raw mode string did not parse into AuthSpecKind (boundary)
    precondition_required: str = case()  # a required per-user value (e.g. an env var) has not been provided -> 412
    not_implemented: str = case()  # the declared mode's resolver arm is not built yet -> 501 (not operator error)

    @staticmethod
    def of_unauthorized(
        detail: str,
        *,
        www_authenticate: str | None = None,
        body: Mapping[str, str] | None = None,
        claims: str | None = None,
    ) -> CredError:
        return CredError(
            unauthorized=Unauthorized(
                detail=detail,
                www_authenticate=www_authenticate,
                body=body,
                claims=claims,
            )
        )

    @staticmethod
    def of_misconfigured(detail: str) -> CredError:
        return CredError(misconfigured=detail)

    @staticmethod
    def of_upstream_unavailable(detail: str) -> CredError:
        return CredError(upstream_unavailable=detail)

    @staticmethod
    def of_unsupported_mode(detail: str) -> CredError:
        return CredError(unsupported_mode=detail)

    @staticmethod
    def of_precondition_required(detail: str) -> CredError:
        return CredError(precondition_required=detail)

    @staticmethod
    def of_not_implemented(detail: str) -> CredError:
        return CredError(not_implemented=detail)

    @property
    def summary(self) -> str:
        # Exhaustiveness: every Literal tag has an arm; the trailing assert_never typechecks
        # only while that stays true (a `case _` would defeat reportMatchNotExhaustive).
        match self.tag:
            case "unauthorized":
                return f"unauthorized: {self.unauthorized.detail}"
            case "misconfigured":
                return f"misconfigured: {self.misconfigured}"
            case "upstream_unavailable":
                return f"upstream unavailable: {self.upstream_unavailable}"
            case "unsupported_mode":
                return self.unsupported_mode
            case "precondition_required":
                return f"precondition required: {self.precondition_required}"
            case "not_implemented":
                return f"not implemented: {self.not_implemented}"
        assert_never(self.tag)


class AuthorizationCodeConfig(BaseModel):
    """Per-user 3LO; the gateway is the OAuth client and stores the user's token.

    Endpoints are discovered (RFC 9728 -> RFC 8414) and the client is registered via DCR
    (RFC 7591), so the common case carries none of the fields below; they are optional manual
    overrides for IdPs without discovery / DCR. The per-user token is read from the token store
    at resolve time, not held here.
    """

    model_config = ConfigDict(frozen=True)
    kind: Literal[AuthSpecKind.authorization_code] = AuthSpecKind.authorization_code
    scopes: tuple[str, ...] = ()
    client_id: str | None = None
    client_secret: SecretStr | None = None
    authorization_url: str | None = None
    token_url: str | None = None


class ClientCredentialsConfig(BaseModel):
    """M2M service account; one upstream identity for every user.

    Fields are optional so the config can be built incomplete: a value may be supplied at
    runtime (`token_url` via RFC 8414 discovery, `client_id`/`secret` via DCR), and the
    resolver arm raises `CredError.misconfigured` when a needed field is still absent.
    """

    model_config = ConfigDict(frozen=True)
    kind: Literal[AuthSpecKind.client_credentials] = AuthSpecKind.client_credentials
    client_id: str | None = None
    client_secret: SecretStr | None = None
    token_url: str | None = None
    scopes: tuple[str, ...] = ()


class TokenExchangeConfig(BaseModel):
    """OBO: swap the caller's live inbound token for a token bound to the upstream's audience. The
    gateway authenticates to the exchange endpoint as an OAuth client (`client_id`/`client_secret`);
    the inbound token is sent only to that endpoint, never to the upstream.

    `profile` selects the wire dialect, since not every IdP speaks RFC 8693:
      - `rfc8693` (default) is the standard token-exchange grant: the inbound token is the
        `subject_token` (typed by `subject_token_type`), the target is the optional `audience`.
      - `entra_obo` is Microsoft Entra On-Behalf-Of, which is the RFC 7523 `jwt-bearer` grant rather
        than 8693: the inbound token rides as `assertion`, the target resource is carried in `scopes`
        (`api://<app-id>/.default`, since Entra has no audience parameter), and the Microsoft-only
        `requested_token_use=on_behalf_of` extension makes the jwt-bearer grant a delegation.
        `subject_token_type` and `audience` are unused in this profile.

    `audience` (rfc8693 only) is optional and sent only when the operator configured one, since both
    `audience` and `resource` are optional in RFC 8693 and the authorization server applies its own
    default when neither is sent (fabricating one risks `invalid_target`).
    """

    model_config = ConfigDict(frozen=True)
    kind: Literal[AuthSpecKind.token_exchange] = AuthSpecKind.token_exchange
    profile: Literal["rfc8693", "entra_obo"] = "rfc8693"
    subject_token_type: str = DEFAULT_SUBJECT_TOKEN_TYPE
    token_exchange_endpoint: str | None = None
    audience: str | None = None
    client_id: str | None = None
    client_secret: SecretStr | None = None
    token_endpoint_auth_method: Literal["client_secret_basic", "client_secret_post"] | None = None
    scopes: tuple[str, ...] = ()


class SharedKey(BaseModel):
    """A fixed key configured on the server, identical for every caller."""

    model_config = ConfigDict(frozen=True)
    source: Literal["shared"] = "shared"
    value: SecretStr


class Byok(BaseModel):
    """A key the user brings via the entry flow, stored per-user and pulled from the credential
    store at resolve time. Missing means the user must provide it, a 401 + WWW-Authenticate
    challenge."""

    model_config = ConfigDict(frozen=True)
    source: Literal["byok"] = "byok"


ApiKeySource = Annotated[SharedKey | Byok, Field(discriminator="source")]


class ApiKeyConfig(BaseModel):
    """A fixed credential injected as a header. The value is shared (in config) or seeded
    per-user (pulled from the store); `header_name` and `value_prefix` say where and how it is
    written, modeled like OpenAPI's apiKey scheme so any upstream convention is expressible
    (Authorization + Bearer, a raw value on X-API-Key, Ocp-Apim-Subscription-Key, etc.).
    """

    model_config = ConfigDict(frozen=True)
    kind: Literal[AuthSpecKind.api_key] = AuthSpecKind.api_key
    header_name: str = "Authorization"
    value_prefix: str = "Bearer"
    key_source: ApiKeySource

    def header(self, value: str) -> tuple[str, str]:
        formatted = f"{self.value_prefix} {value}" if self.value_prefix else value
        return self.header_name, formatted


class PassthroughConfig(BaseModel):
    """Client-driven upstream OAuth; the gateway forwards the client's upstream token."""

    model_config = ConfigDict(frozen=True)
    kind: Literal[AuthSpecKind.passthrough] = AuthSpecKind.passthrough


class NoneConfig(BaseModel):
    """No upstream credential; the request is sent unauthenticated."""

    model_config = ConfigDict(frozen=True)
    kind: Literal[AuthSpecKind.none] = AuthSpecKind.none


class StaticKeys(BaseModel):
    """Long-lived AWS access keys configured on the server."""

    model_config = ConfigDict(frozen=True)
    source: Literal["static_keys"] = "static_keys"
    access_key_id: str
    secret_access_key: SecretStr
    session_token: SecretStr | None = None


class AssumeRole(BaseModel):
    """An IAM role the gateway assumes via STS for short-lived, auto-refreshed credentials."""

    model_config = ConfigDict(frozen=True)
    source: Literal["assume_role"] = "assume_role"
    role_arn: str
    session_name: str | None = None
    external_id: str | None = None


class Ambient(BaseModel):
    """The environment's default AWS credential chain (instance profile, IRSA, env vars)."""

    model_config = ConfigDict(frozen=True)
    source: Literal["ambient"] = "ambient"


AwsCredentialSource = Annotated[StaticKeys | AssumeRole | Ambient, Field(discriminator="source")]


class AwsSigV4Config(BaseModel):
    """AWS SigV4 per-request signing for an AWS-hosted upstream (e.g. Bedrock AgentCore). The
    gateway signs with its own AWS identity, never the caller's; `credentials` selects how that
    identity is obtained, defaulting to the ambient credential chain."""

    model_config = ConfigDict(frozen=True)
    kind: Literal[AuthSpecKind.aws_sigv4] = AuthSpecKind.aws_sigv4
    region: str
    service: str = "bedrock-agentcore"
    credentials: AwsCredentialSource = Ambient()


AuthConfig = Annotated[
    AuthorizationCodeConfig
    | ClientCredentialsConfig
    | TokenExchangeConfig
    | ApiKeyConfig
    | PassthroughConfig
    | NoneConfig
    | AwsSigV4Config,
    Field(discriminator="kind"),
]


class Subject(BaseModel):
    """The validated inbound principal. NOT the v1 request object and NOT the LiteLLM key."""

    model_config = ConfigDict(frozen=True)

    tenant_id: str
    subject_id: str
    # Opaque, already-validated inbound identity. Only `token_exchange` / `passthrough` read it.
    inbound_token: SecretStr | None = None
    # Set only for an agent-delegated (on-behalf-of) request; equals subject_id when present. Its
    # presence means inbound_token is the AGENT's admission credential, NOT this subject's own token,
    # so an arm that would otherwise consume inbound_token as the subject's proof (token_exchange)
    # must instead source the subject's own material. subject_id stays the principal whose upstream
    # credential is resolved either way, so arms keyed purely on subject_id (authorization_code) are
    # already correct and read nothing here.
    delegated_user_id: str | None = None


class ServerSpec(BaseModel):
    """The declared upstream. A v2-native type; the v1 -> v2 adapter maps onto this."""

    model_config = ConfigDict(frozen=True)

    server_id: str
    resource: str  # RFC 8707 audience URI this upstream's tokens are bound to
    config: AuthConfig

    @property
    def auth_spec_kind(self) -> AuthSpecKind:
        return self.config.kind


def parse_auth_spec_kind(raw: str) -> Result[AuthSpecKind, CredError]:
    """Boundary parser — the *only* place an unknown mode is handled, and it fails closed.

    Inside the core the mode is always a valid `AuthSpecKind`, so the resolver never needs a
    wildcard arm and basedpyright can prove its `match` exhaustive.
    """
    try:
        return Ok(AuthSpecKind(raw))
    except ValueError:
        return Error(CredError.of_unsupported_mode(f"unknown auth_spec_kind: {raw!r}"))
