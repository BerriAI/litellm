"""The OAuth / upstream-credential vocabulary — the v2 typed seam (Phase 0).

This module is the *contract* the Phase 1 build implements and the spec tests assert
against. It ships types and one stub `resolve()`; no credential logic lands in Phase 0.

Design invariants encoded here:

- **Mode is the single source of truth.** `resolve()` dispatches on the server's declared
  `AuthSpecKind`, one arm per mode. No field-presence inference, no precedence cascade.
- **Exhaustive dispatch, no wildcard.** `auth_spec_kind` is a closed enum, so the `match` in
  `resolve()` has no `_` arm — basedpyright (`reportMatchNotExhaustive`) guarantees every mode
  is handled. Adding a sixth mode without an arm fails the type gate, not at runtime.
- **Fail-closed at the boundary.** An unknown mode string can only enter through
  `parse_auth_spec_kind()`, which returns a typed `CredError`. Inside the core the mode is
  always valid, so illegal states are unrepresentable.
- **Errors as values.** Every seam returns `Result[_, CredError]`; only edge adapters raise.
- **Clean-room.** No imports from v1.

House style follows `litellm/translation/` (sum types = Expression `@tagged_union`
discriminated on a `Literal` `tag`, matched via `self.tag` with an `assert_never` tail;
`Result` is the vendored `Ok | Error` union, not `expression.Result`).
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

import httpx
from expression import case, tag, tagged_union
from pydantic import BaseModel, ConfigDict
from typing_extensions import assert_never

from ..result import Error, Ok, Result


class AuthSpecKind(str, Enum):
    """The server's statically-declared upstream-auth mode — the single source of truth.

    Covers v1's full `MCPAuth` surface, not only OAuth grants: the three grant modes, the
    collapsed static-header family, client passthrough, no-auth, and AWS request signing.
    BYOK is *not* a member: it is the `api_key` mode seeded per-user, a source selector
    inside that arm. The static-header schemes v1 splits into separate `MCPAuth` values
    (`bearer_token`/`api_key`/`basic`/`token`/`authorization`) collapse into `api_key`; the
    scheme is a parameter the arm carries, not its own mode.
    """

    authorization_code = (
        "authorization_code"  # per-user 3LO; gateway is the OAuth client
    )
    client_credentials = "client_credentials"  # gateway service account (M2M)
    token_exchange = "token_exchange"  # RFC 8693 on-behalf-of
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
        "unauthorized", "misconfigured", "upstream_unavailable", "unsupported_mode"
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
        assert_never(self.tag)


class Subject(BaseModel):
    """The validated inbound principal. NOT the v1 request object and NOT the LiteLLM key."""

    model_config = ConfigDict(frozen=True)

    tenant_id: str
    subject_id: str
    # Opaque, already-validated inbound identity. Only `token_exchange` / `passthrough` read it.
    inbound_token: str | None = None


class ServerSpec(BaseModel):
    """The declared upstream. A v2-native type; the v1->v2 adapter maps onto this."""

    model_config = ConfigDict(frozen=True)

    server_id: str
    auth_spec_kind: AuthSpecKind
    resource: str  # RFC 8707 audience URI this upstream's tokens are bound to


def parse_auth_spec_kind(raw: str) -> Result[AuthSpecKind, CredError]:
    """Boundary parser — the *only* place an unknown mode is handled, and it fails closed.

    Inside the core the mode is always a valid `AuthSpecKind`, so `resolve()` never needs a
    wildcard arm and basedpyright can prove its `match` exhaustive.
    """
    try:
        return Ok(AuthSpecKind(raw))
    except ValueError:
        return Error(CredError.of_unsupported_mode(f"unknown auth_spec_kind: {raw!r}"))


class UpstreamCredentialProvider:
    """The ONE credential resolver. Phase 0 ships the seam; arms are stubs filled in Phase 1."""

    def resolve(
        self, subject: Subject, server: ServerSpec
    ) -> Result[httpx.Auth, CredError]:
        """Select exactly one provider off the declared mode, or fail closed.

        The `match` is intentionally wildcard-free: every `AuthSpecKind` member must have an
        arm or the type gate fails. This is the property the Phase 0 spike verifies.
        """
        match server.auth_spec_kind:
            case AuthSpecKind.authorization_code:
                return self._authorization_code(subject, server)
            case AuthSpecKind.client_credentials:
                return self._client_credentials(subject, server)
            case AuthSpecKind.token_exchange:
                return self._token_exchange(subject, server)
            case AuthSpecKind.api_key:
                return self._api_key(subject, server)
            case AuthSpecKind.passthrough:
                return self._passthrough(subject, server)
            case AuthSpecKind.none:
                return self._none(subject, server)
            case AuthSpecKind.aws_sigv4:
                return self._aws_sigv4(subject, server)

    # --- arms: Phase 0 stubs (errors-as-values, no raise). Filled in Phase 1. -------------
    def _authorization_code(
        self, subject: Subject, server: ServerSpec
    ) -> Result[httpx.Auth, CredError]:
        return _todo(AuthSpecKind.authorization_code)

    def _client_credentials(
        self, subject: Subject, server: ServerSpec
    ) -> Result[httpx.Auth, CredError]:
        return _todo(AuthSpecKind.client_credentials)

    def _token_exchange(
        self, subject: Subject, server: ServerSpec
    ) -> Result[httpx.Auth, CredError]:
        return _todo(AuthSpecKind.token_exchange)

    def _api_key(
        self, subject: Subject, server: ServerSpec
    ) -> Result[httpx.Auth, CredError]:
        return _todo(AuthSpecKind.api_key)

    def _passthrough(
        self, subject: Subject, server: ServerSpec
    ) -> Result[httpx.Auth, CredError]:
        return _todo(AuthSpecKind.passthrough)

    def _none(
        self, subject: Subject, server: ServerSpec
    ) -> Result[httpx.Auth, CredError]:
        return _todo(AuthSpecKind.none)

    def _aws_sigv4(
        self, subject: Subject, server: ServerSpec
    ) -> Result[httpx.Auth, CredError]:
        return _todo(AuthSpecKind.aws_sigv4)


def _todo(kind: AuthSpecKind) -> Result[httpx.Auth, CredError]:
    return Error(
        CredError.of_misconfigured(
            f"{kind.value}: resolver arm not implemented (Phase 0 skeleton)"
        )
    )
