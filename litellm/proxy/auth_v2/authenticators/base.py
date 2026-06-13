from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Protocol, Sequence, Tuple, runtime_checkable

from fastapi import Request

from litellm.proxy.auth_v2.models import Credential
from litellm.proxy.auth_v2.network import ip_in_cidrs


def verified_client_cert_name(request: Request) -> Optional[str]:
    name = request.scope.get("extensions", {}).get("tls", {}).get("client_cert_name")
    return name or None


class CredentialLocation(str, Enum):
    AUTHORIZATION_SCHEME = "authorization_scheme"
    HEADER = "header"
    COOKIE = "cookie"
    CLIENT_CERTIFICATE = "client_certificate"


@dataclass(frozen=True)
class Carrier:
    """Where an authenticator reads its credential from.

    Each authenticator advertises the single carrier it reads, so the security
    layer can route a request straight to the one authenticator whose credential
    is present instead of trying each in turn. ``present`` mirrors that
    authenticator's accept condition exactly, so the chosen authenticator never
    declines and shadows a lower-priority credential.
    """

    location: CredentialLocation
    name: str = ""
    trusted_proxy_cidrs: Tuple[str, ...] = field(default=())

    def present(self, request: Request) -> bool:
        if self.location is CredentialLocation.AUTHORIZATION_SCHEME:
            scheme, _, value = request.headers.get("authorization", "").partition(" ")
            return scheme.lower() == self.name and bool(value)
        if self.location is CredentialLocation.HEADER:
            return bool(request.headers.get(self.name))
        if self.location is CredentialLocation.COOKIE:
            return self.name in request.cookies
        if verified_client_cert_name(request) is not None:
            return True
        peer = request.client.host if request.client else None
        return (
            bool(self.name)
            and ip_in_cidrs(peer, self.trusted_proxy_cidrs)
            and bool(request.headers.get(self.name))
        )


@runtime_checkable
class Authenticator(Protocol):
    async def authenticate(self, request: Request) -> Optional[Credential]: ...

    def carriers(self) -> Sequence[Carrier]: ...

    def challenge(self) -> str: ...
