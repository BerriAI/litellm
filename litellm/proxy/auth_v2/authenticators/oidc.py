from __future__ import annotations

from typing import List, Optional, Sequence

from fastapi import Request

from litellm.proxy.auth_v2 import errors
from litellm.proxy.auth_v2.models import AuthMethod, Credential, SecuritySchemeType
from litellm.proxy.auth_v2.authenticators.base import (
    Authenticator,
    Carrier,
    CredentialLocation,
)
from litellm.proxy.auth_v2.authenticators.utils import (
    JWTVerifier,
    authenticate_bearer_jwt,
    extract_bearer,
)


class OIDCAuthenticator(Authenticator):
    def __init__(self, jwt_verifiers: List[JWTVerifier]) -> None:
        self._verifiers = jwt_verifiers

    async def authenticate(self, request: Request) -> Optional[Credential]:
        token = extract_bearer(request)
        if token is None:
            return None
        return await authenticate_bearer_jwt(
            token, self._verifiers, SecuritySchemeType.OPENID_CONNECT, AuthMethod.OIDC
        )

    def carriers(self) -> Sequence[Carrier]:
        return (Carrier(CredentialLocation.AUTHORIZATION_SCHEME, "bearer"),)

    def challenge(self) -> str:
        return errors.bearer_challenge()
