"""The OAuth / upstream-credential vocabulary — the v2 typed seam.

This module is the *contract* the credential build implements and the spec tests assert
against. It ships the data types only; the resolver lives in `resolver.py`.

Design invariants encoded here:

- **Mode is the single source of truth.** A server declares exactly one per-mode `config`
  (the `AuthConfig` discriminated union); `auth_spec_kind` is *derived* from it, never a
  second field that can drift. The resolver dispatches on the config variant, one arm per
  mode. No field-presence inference, no precedence cascade.
- **Illegal states unrepresentable.** Each mode's config is its own frozen model holding
  only that mode's fields, all required — an `aws_sigv4` server cannot hold OAuth fields,
  and bad config is rejected at construction, not at call time.
- **Fail-closed at the boundary.** A raw mode string can only enter through
  `parse_auth_spec_kind()`, which returns a typed `CredError`.
- **Errors as values.** Every seam returns `Result[_, CredError]`; only edge adapters raise.
- **Clean-room.** No imports from v1.

House style follows `litellm/translation/` (sum types = Expression `@tagged_union`
discriminated on a `Literal` `tag`, matched via `self.tag` with an `assert_never` tail;
`Result` is the vendored `Ok | Error` union, not `expression.Result`).
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Literal

from expression import case, tag, tagged_union
from pydantic import BaseModel, ConfigDict, Field, SecretStr
from typing_extensions import assert_never

from ..result import Error, Ok, Result


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

    unauthorized: str = (
        case()
    )  # no usable credential for this (subject, server) -> 401 challenge
    misconfigured: str = (
        case()
    )  # the declared mode is missing required config -> 5xx (operator)
    upstream_unavailable: str = (
        case()
    )  # the IdP / token endpoint could not be reached -> 503
    unsupported_mode: str = (
        case()
    )  # a raw mode string did not parse into AuthSpecKind (boundary)
    precondition_required: str = (
        case()
    )  # a required per-user value (e.g. an env var) has not been provided -> 412
    not_implemented: str = (
        case()
    )  # the declared mode's resolver arm is not built yet -> 501 (not operator error)

    @staticmethod
    def of_unauthorized(detail: str) -> CredError:
        return CredError(unauthorized=detail)

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
                return f"unauthorized: {self.unauthorized}"
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


ApiKeyScheme = Literal["bearer", "apikey", "basic", "token", "raw"]


class AuthorizationCodeConfig(BaseModel):
    """Per-user 3LO; the gateway is the OAuth client and stores the user's token.

    Endpoints are discovered (RFC 9728 -> RFC 8414) and the client is registered via DCR
    (RFC 7591), so the common case carries none of the fields below; they are optional manual
    overrides for IdPs without discovery / DCR. The discovered endpoints and the DCR-registered
    client are persisted by the AS surface, not here; `resolve()` reads the per-user token from
    the `TokenStore`.
    """

    model_config = ConfigDict(frozen=True)
    kind: Literal[AuthSpecKind.authorization_code] = AuthSpecKind.authorization_code
    scopes: tuple[str, ...] = ()
    client_id: str | None = None
    client_secret: SecretStr | None = None
    authorization_url: str | None = None
    token_url: str | None = None


class ClientCredentialsConfig(BaseModel):
    """M2M service account; one upstream identity for every user."""

    model_config = ConfigDict(frozen=True)
    kind: Literal[AuthSpecKind.client_credentials] = AuthSpecKind.client_credentials
    client_id: str
    client_secret: SecretStr
    token_url: str
    scopes: tuple[str, ...] = ()


class TokenExchangeConfig(BaseModel):
    """RFC 8693 OBO; swap the caller's live subject_token for an upstream-audience token."""

    model_config = ConfigDict(frozen=True)
    kind: Literal[AuthSpecKind.token_exchange] = AuthSpecKind.token_exchange
    token_exchange_endpoint: str
    audience: str
    subject_token_type: str = "urn:ietf:params:oauth:token-type:access_token"


class SharedKey(BaseModel):
    """A fixed key configured on the server, identical for every caller."""

    model_config = ConfigDict(frozen=True)
    source: Literal["shared"] = "shared"
    value: SecretStr


class PerUserEnvVar(BaseModel):
    """A per-user value the admin templated as an env var; the user fills it in. Pulled from
    the credential store at resolve time. Missing means the user has not completed setup, a
    precondition (412) rather than an auth failure."""

    model_config = ConfigDict(frozen=True)
    source: Literal["per_user_env_var"] = "per_user_env_var"


class Byok(BaseModel):
    """A key the user brings via the entry flow, stored per-user. Pulled from the credential
    store at resolve time. Missing means the user must provide it, a 401 + WWW-Authenticate
    challenge."""

    model_config = ConfigDict(frozen=True)
    source: Literal["byok"] = "byok"


ApiKeySource = Annotated[
    SharedKey | PerUserEnvVar | Byok, Field(discriminator="source")
]


class ApiKeyConfig(BaseModel):
    """A fixed credential injected as a header. The value is shared (in config) or seeded
    per-user (pulled from the store); `scheme` is how it is written into the header."""

    model_config = ConfigDict(frozen=True)
    kind: Literal[AuthSpecKind.api_key] = AuthSpecKind.api_key
    scheme: ApiKeyScheme = "bearer"
    key_source: ApiKeySource

    def header_for(self, value: str) -> str:
        # Reconstructs v1's per-scheme Authorization prefixes (mcp_server_manager.py:877-884).
        match self.scheme:
            case "bearer":
                return f"Bearer {value}"
            case "apikey":
                return f"ApiKey {value}"
            case "basic":
                return f"Basic {value}"
            case "token":
                return f"token {value}"
            case "raw":
                return value
        assert_never(self.scheme)


class PassthroughConfig(BaseModel):
    """Client-driven upstream OAuth; the gateway forwards the client's upstream token."""

    model_config = ConfigDict(frozen=True)
    kind: Literal[AuthSpecKind.passthrough] = AuthSpecKind.passthrough


class NoneConfig(BaseModel):
    """No upstream credential; the request is sent unauthenticated."""

    model_config = ConfigDict(frozen=True)
    kind: Literal[AuthSpecKind.none] = AuthSpecKind.none


class AwsSigV4Config(BaseModel):
    """AWS SigV4 per-request signing. Creds come from static keys, an assumed role, or
    the ambient environment; that source is left loose here and tightened when the arm lands.
    """

    model_config = ConfigDict(frozen=True)
    kind: Literal[AuthSpecKind.aws_sigv4] = AuthSpecKind.aws_sigv4
    region: str
    service: str = "bedrock-agentcore"
    access_key_id: str | None = None
    secret_access_key: SecretStr | None = None
    session_token: SecretStr | None = None
    role_arn: str | None = None
    session_name: str | None = None


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


class ServerSpec(BaseModel):
    """The declared upstream. A v2-native type; the v1->v2 adapter maps onto this."""

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
