from __future__ import annotations

from typing import Optional

from fastapi import Request

from litellm.proxy.auth_v2.config import MutualTLSConfig, TrustedProxyConfig
from litellm.proxy.auth_v2.models import AuthMethod, ClientCertificate, Credential, SecuritySchemeType
from litellm.proxy.auth_v2.network import ip_in_trusted_proxies
from litellm.proxy.auth_v2.authenticators.base import Authenticator


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
        tls = request.scope.get("extensions", {}).get("tls", {})
        verified_dn = tls.get("client_cert_name")
        if verified_dn:
            return ClientCertificate(subject_dn=verified_dn)
        if self._config.forwarded_subject_header:
            peer = request.client.host if request.client else None
            if not ip_in_trusted_proxies(peer, self._network):
                return None
            dn = request.headers.get(self._config.forwarded_subject_header)
            return ClientCertificate(subject_dn=dn) if dn else None
        return None

    def challenge(self) -> str:
        return ""
