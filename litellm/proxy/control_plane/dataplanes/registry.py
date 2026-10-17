"""Which dataplanes exist and how to reach each one. Phase 1 is config-backed and read-only."""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Protocol

from litellm.litellm_core_utils.oauth_token_exchange import OAuthTokenExchangeConfig, OAuthTokenExchangeError
from litellm.proxy.control_plane.auth.rbac import Principal
from litellm.proxy.control_plane.dataplanes.credentials import (
    CredentialChain,
    CredentialKind,
    DataplaneCredential,
    OidcTokenExchange,
    ServiceTokenImpersonation,
    TokenExchanger,
)
from litellm.proxy.control_plane.settings import DataplaneSpec


@dataclass(frozen=True, slots=True)
class Dataplane:
    id: str
    name: str
    url: str
    credential: DataplaneCredential
    credential_kinds: tuple[CredentialKind, ...]


class DataplaneRegistry(Protocol):
    def get(self, dataplane_id: str) -> Dataplane | None: ...

    def all(self) -> tuple[Dataplane, ...]: ...


@dataclass(frozen=True, slots=True)
class StaticRegistry:
    entries: Mapping[str, Dataplane]

    def get(self, dataplane_id: str) -> Dataplane | None:
        return self.entries.get(dataplane_id)

    def all(self) -> tuple[Dataplane, ...]:
        return tuple(self.entries.values())


class InvalidDataplaneConfig(ValueError):
    def __init__(self, dataplane_id: str, reason: str) -> None:
        super().__init__(f"dataplane {dataplane_id!r}: {reason}")


def visible_to(registry: DataplaneRegistry, principal: Principal) -> tuple[Dataplane, ...]:
    return tuple(dataplane for dataplane in registry.all() if principal.dataplanes.allows(dataplane.id))


def _strategies(dataplane_id: str, spec: DataplaneSpec, exchanger: TokenExchanger) -> tuple[DataplaneCredential, ...]:
    oidc: Final = spec.oidc_exchange
    service: Final = spec.service_token
    if oidc is None and service is None:
        raise InvalidDataplaneConfig(dataplane_id, "set oidc_exchange, service_token or both")
    try:
        exchange: Final = (
            ()
            if oidc is None
            else (
                OidcTokenExchange(
                    OAuthTokenExchangeConfig(
                        token_endpoint=oidc.token_endpoint,
                        client_id=oidc.client_id,
                        client_secret=oidc.client_secret.get_secret_value(),
                        profile=oidc.profile,
                        scope=oidc.scope,
                        audience=oidc.audience,
                    ),
                    exchanger,
                ),
            )
        )
    except OAuthTokenExchangeError as error:
        raise InvalidDataplaneConfig(dataplane_id, error.message) from error
    impersonation: Final = () if service is None else (ServiceTokenImpersonation(service.token.get_secret_value()),)
    return (*exchange, *impersonation)


def build_dataplane(dataplane_id: str, spec: DataplaneSpec, exchanger: TokenExchanger) -> Dataplane:
    chain: Final = CredentialChain(_strategies(dataplane_id, spec, exchanger))
    return Dataplane(
        id=dataplane_id,
        name=spec.name or dataplane_id,
        url=spec.url.rstrip("/"),
        credential=chain,
        credential_kinds=chain.kinds,
    )


def build_registry(specs: Mapping[str, DataplaneSpec], exchanger: TokenExchanger) -> StaticRegistry:
    return StaticRegistry(
        MappingProxyType(
            {dataplane_id: build_dataplane(dataplane_id, spec, exchanger) for dataplane_id, spec in specs.items()}
        )
    )
