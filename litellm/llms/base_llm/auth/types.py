"""Provider-agnostic value types for OAuth token exchanges: where an assertion comes from, the typed
failure union every grant maps onto its public exception contract, and the observability seam."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Literal, Protocol, TypeAlias

import httpx

AssertionReader: TypeAlias = Callable[[str], str | None]  # mutable-ok: Callable param-list syntax, not a list
AssertionSource: TypeAlias = Callable[[], str | None]  # mutable-ok: Callable param-list syntax, not a list


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
ExchangeResult: TypeAlias = str | ExchangeError

ExchangeCallType: TypeAlias = Literal["cold_mint", "refresh"]


class TokenExchangeMetricsSink(Protocol):
    """Observability seam for a token exchange. Implementations must be best-effort: never raise
    into the mint path, never block the calling thread, and never receive credential material --
    ``ExchangeError`` values are redacted by construction."""

    def exchange_success(self, *, call_type: ExchangeCallType, duration_seconds: float) -> None: ...

    def exchange_failure(
        self, *, call_type: ExchangeCallType, duration_seconds: float, error: ExchangeError
    ) -> None: ...

    def cache_hit(self) -> None: ...


class SyncTokenPoster(Protocol):
    """Returns the response for ANY status; never raises for status."""

    def post(self, url: str, *, content: bytes, headers: Mapping[str, str], timeout: float) -> httpx.Response: ...
