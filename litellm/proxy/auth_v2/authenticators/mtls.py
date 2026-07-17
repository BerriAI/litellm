from __future__ import annotations

from typing import Optional, Sequence

from fastapi import Request

from litellm.proxy.auth_v2.config import MutualTLSConfig, TrustedProxyConfig
from litellm.proxy.auth_v2.models import (
    AuthMethod,
    ClientCertificate,
    Credential,
    SecuritySchemeType,
)
from litellm.proxy.auth_v2.network import ip_in_cidrs
from litellm.proxy.auth_v2.authenticators.base import (
    Authenticator,
    Carrier,
    CredentialLocation,
    verified_client_cert_name,
)


class MutualTLSAuthenticator(Authenticator):
    def __init__(self, config: MutualTLSConfig, network: TrustedProxyConfig) -> None:
        self._config = config
        self._network = network

    async def authenticate(self, request: Request) -> Optional[Credential]:
        cert = self._read_client_cert(request)
        if cert is None:
            return None
        return Credential(
            scheme=SecuritySchemeType.MUTUAL_TLS,
            method=AuthMethod.MUTUAL_TLS,
            subject=cert.subject_dn,
            client_certificate=cert,
        )

    def _read_client_cert(self, request: Request) -> Optional[ClientCertificate]:
        verified_dn = verified_client_cert_name(request)
        if verified_dn:
            return ClientCertificate(subject_dn=verified_dn)
        if self._config.forwarded_subject_header:
            peer = request.client.host if request.client else None
            if not ip_in_cidrs(peer, self._network.trusted_proxy_cidrs):
                return None
            dn = request.headers.get(self._config.forwarded_subject_header)
            return ClientCertificate(subject_dn=dn) if dn else None
        return None

    def carriers(self) -> Sequence[Carrier]:
        return (
            Carrier(
                CredentialLocation.CLIENT_CERTIFICATE,
                self._config.forwarded_subject_header or "",
                tuple(self._network.trusted_proxy_cidrs),
            ),
        )

    def challenge(self) -> str:
        return ""
