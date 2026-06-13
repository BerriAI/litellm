from __future__ import annotations

from typing import Optional, Sequence

from fastapi import Request

from litellm.proxy.auth_v2.config import ApiKeySchemeConfig
from litellm.proxy.auth_v2.models import (
    AuthMethod,
    Credential,
    CredentialRef,
    SecuritySchemeType,
)
from litellm.proxy.auth_v2.authenticators.base import (
    Authenticator,
    Carrier,
    CredentialLocation,
)


class APIKeyAuthenticator(Authenticator):
    def __init__(self, config: ApiKeySchemeConfig) -> None:
        self._header_name = config.header_name

    async def authenticate(self, request: Request) -> Optional[Credential]:
        raw = request.headers.get(self._header_name)
        if not raw:
            return None
        return Credential(
            scheme=SecuritySchemeType.API_KEY,
            method=AuthMethod.API_KEY,
            subject=raw,
            credential_ref=CredentialRef(key_id=raw[:10]),
            claims={"_raw_api_key": raw},
        )

    def carriers(self) -> Sequence[Carrier]:
        return (Carrier(CredentialLocation.HEADER, self._header_name),)

    def challenge(self) -> str:
        return ""
