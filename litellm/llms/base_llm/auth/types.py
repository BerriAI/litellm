"""Provider-agnostic types for the RFC 7523 JWT-bearer token exchange engine."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Literal, Protocol, TypeAlias

import httpx
from pydantic import SecretStr

BodyEncoding: TypeAlias = Literal["json", "form"]
AssertionReader: TypeAlias = Callable[[str], str | None]  # mutable-ok: Callable param-list syntax, not a list
AssertionSource: TypeAlias = Callable[[], str | None]  # mutable-ok: Callable param-list syntax, not a list


@dataclass(frozen=True, slots=True)
class TokenExchangeSpec:
    """One grant profile as pure data: one instance per (provider, deployment, identity).

    ``token_url`` must be derived from deployment config/env only, never per-request caller
    input. ``assertion_ref`` is a ``oidc/...`` get_secret ref resolved fresh on every exchange.

    ``assertion_source``, when set, is a zero-arg per-config fetch/mint closure that the engine
    prefers over its own engine-level ``AssertionReader`` -- the dispatch mechanism identity
    sources beyond token_file/env (e.g. ``internal_issuer``, ``keycloak``) use to plug into the
    shared engine without a global registry. ``assertion_ref`` still names the cache-key
    discriminator and the ref echoed into operator-facing errors either way.
    """

    token_url: str
    assertion_ref: str
    assertion_field: str
    static_body: Mapping[str, str]
    body_encoding: BodyEncoding
    request_headers: Mapping[str, str]
    cache_key_identity: tuple[str, ...]
    timeout_seconds: float = 30.0
    assertion_source: AssertionSource | None = None


@dataclass(frozen=True, slots=True)
class MintedToken:
    access_token: SecretStr
    expires_at: float | None


@dataclass(frozen=True, slots=True)
class AssertionSourceError:
    kind: Literal["missing", "empty", "oversized", "unreadable", "disallowed_path"]
    source_ref: str
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class InsecureTokenUrl:
    host: str


@dataclass(frozen=True, slots=True)
class TokenEndpointError:
    status_code: int
    redacted_body: str


@dataclass(frozen=True, slots=True)
class TokenTransportError:
    detail: str


@dataclass(frozen=True, slots=True)
class MalformedTokenResponse:
    detail: str


ExchangeError: TypeAlias = (
    AssertionSourceError | InsecureTokenUrl | TokenEndpointError | TokenTransportError | MalformedTokenResponse
)
ExchangeResult: TypeAlias = MintedToken | ExchangeError


class SyncTokenPoster(Protocol):
    """Returns the response for ANY status; never raises for status."""

    def post(self, url: str, *, content: bytes, headers: Mapping[str, str], timeout: float) -> httpx.Response: ...
