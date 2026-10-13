"""How the control plane authenticates to a dataplane without losing who the caller is."""

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal, Protocol, TypeAlias

from litellm.exceptions import APIConnectionError, Timeout
from litellm.litellm_core_utils.oauth_token_exchange import (
    OAuthTokenExchangeConfig,
    OAuthTokenExchangeError,
    aexchange_token,
)
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.proxy.auth.impersonation import IMPERSONATE_EMAIL_HEADER, IMPERSONATE_USER_HEADER
from litellm.proxy.control_plane.auth.rbac import Principal

CredentialKind: TypeAlias = Literal["oidc_exchange", "service_token"]
TokenExchanger: TypeAlias = Callable[[OAuthTokenExchangeConfig, str], Awaitable[str]]


@dataclass(frozen=True, slots=True)
class Caller:
    """The authenticated control-plane principal plus the raw bearer it presented, needed for token exchange."""

    principal: Principal
    bearer: str


@dataclass(frozen=True, slots=True)
class CredentialUnavailable:
    kind: CredentialKind
    reason: str


@dataclass(frozen=True, slots=True)
class IssuedCredential:
    kind: CredentialKind
    headers: Mapping[str, str]


CredentialResult: TypeAlias = IssuedCredential | CredentialUnavailable


class DataplaneCredential(Protocol):
    async def headers(self, caller: Caller) -> CredentialResult: ...


def _bearer(token: str) -> Mapping[str, str]:
    return MappingProxyType({"Authorization": f"Bearer {token}"})


def _issued(kind: CredentialKind, headers: Mapping[str, str]) -> IssuedCredential:
    return IssuedCredential(kind, MappingProxyType(dict(headers)))


@dataclass(frozen=True, slots=True)
class OidcTokenExchange:
    """RFC 8693 (or jwt-bearer on-behalf-of) exchange of the caller's IdP token for a dataplane-audience token."""

    config: OAuthTokenExchangeConfig
    exchange: TokenExchanger

    async def headers(self, caller: Caller) -> CredentialResult:
        kind: Final = caller.principal.token_kind
        if kind != "oidc":
            return CredentialUnavailable(
                "oidc_exchange", f"token exchange needs an IdP token, caller presented a {kind} token"
            )
        try:
            return _issued("oidc_exchange", _bearer(await self.exchange(self.config, caller.bearer)))
        except OAuthTokenExchangeError as error:
            return CredentialUnavailable("oidc_exchange", f"token exchange failed with status {error.status_code}")


@dataclass(frozen=True, slots=True)
class ServiceTokenImpersonation:
    """A dataplane virtual key with the `impersonate` permission, asserting the caller via X-LiteLLM-Impersonate-*."""

    token: str

    async def headers(self, caller: Caller) -> CredentialResult:
        principal: Final = caller.principal
        if principal.token_kind == "master_key":
            return _issued("service_token", _bearer(self.token))
        email: Final = {} if principal.email is None else {IMPERSONATE_EMAIL_HEADER: principal.email}
        return _issued("service_token", {**_bearer(self.token), IMPERSONATE_USER_HEADER: principal.subject, **email})


@dataclass(frozen=True, slots=True)
class CredentialChain:
    """Tries each strategy in order and returns the first one that can authenticate this caller."""

    strategies: tuple[DataplaneCredential, ...]

    async def headers(self, caller: Caller) -> CredentialResult:
        return await _first_available(self.strategies, caller, ())

    @property
    def kinds(self) -> tuple[CredentialKind, ...]:
        return tuple(kind_of(strategy) for strategy in self.strategies)


async def _first_available(
    strategies: tuple[DataplaneCredential, ...],
    caller: Caller,
    failures: tuple[CredentialUnavailable, ...],
) -> CredentialResult:
    if not strategies:
        reasons: Final = "; ".join(f"{failure.kind}: {failure.reason}" for failure in failures)
        return CredentialUnavailable(
            failures[-1].kind if failures else "service_token", reasons or "no credential configured"
        )
    result: Final = await strategies[0].headers(caller)
    if not isinstance(result, CredentialUnavailable):
        return result
    return await _first_available(strategies[1:], caller, (*failures, result))


def kind_of(strategy: DataplaneCredential) -> CredentialKind:
    return "oidc_exchange" if isinstance(strategy, OidcTokenExchange) else "service_token"


def http_token_exchanger(client: AsyncHTTPHandler, timeout: float) -> TokenExchanger:
    """``AsyncHTTPHandler.post`` turns transport failures into litellm exceptions, map them back to the exchange
    error so the credential chain can fall through instead of surfacing a 500."""

    async def exchange(config: OAuthTokenExchangeConfig, subject_token: str) -> str:
        try:
            return await aexchange_token(client, config, subject_token, timeout=timeout)
        except (Timeout, APIConnectionError) as error:
            raise OAuthTokenExchangeError(
                status_code=502, message=f"token exchange transport failed: {type(error).__name__}"
            ) from None

    return exchange
