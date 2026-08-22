"""Provider-agnostic types for the RFC 7523 JWT-bearer token exchange engine."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Literal, Protocol, TypeAlias

import httpx
from pydantic import SecretStr

BodyEncoding: TypeAlias = Literal["json", "form"]


@dataclass(frozen=True, slots=True)
class TokenExchangeSpec:
    """One grant profile as pure data: one instance per (provider, deployment, identity).

    ``token_url`` must be derived from deployment config/env only, never per-request caller
    input. ``assertion_ref`` is a ``oidc/...`` get_secret ref resolved fresh on every exchange.
    """

    token_url: str
    assertion_ref: str
    assertion_field: str
    static_body: Mapping[str, str]
    body_encoding: BodyEncoding
    request_headers: Mapping[str, str]
    cache_key_identity: tuple[str, ...]
    timeout_seconds: float = 30.0


@dataclass(frozen=True, slots=True)
class MintedToken:
    access_token: SecretStr
    expires_at: float | None


@dataclass(frozen=True, slots=True)
class AssertionSourceError:
    kind: Literal["missing", "empty", "oversized", "unreadable", "disallowed_path"]
    source_ref: str


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


AssertionReader: TypeAlias = Callable[[str], str | None]  # mutable-ok: Callable param-list syntax, not a list
