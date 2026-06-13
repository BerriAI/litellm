from __future__ import annotations

import base64
import binascii
import hashlib
import secrets
from typing import List, Optional, Sequence

from fastapi import Request

from litellm.proxy.auth_v2 import errors
from litellm.proxy.auth_v2.config import HttpBasicConfig
from litellm.proxy.auth_v2.models import AuthMethod, Credential, SecuritySchemeType
from litellm.proxy.auth_v2.authenticators.base import (
    Authenticator,
    Carrier,
    CredentialLocation,
)
from litellm.proxy.auth_v2.authenticators.types import BasicAuthVerifier
from litellm.proxy.auth_v2.authenticators.utils import (
    JWTVerifier,
    authenticate_bearer_jwt,
)

_PBKDF2_ITERATIONS = 600_000


def hash_basic_password(password: str, salt: Optional[str] = None) -> str:
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode(), bytes.fromhex(salt), _PBKDF2_ITERATIONS
    ).hex()
    return f"pbkdf2_sha256${_PBKDF2_ITERATIONS}${salt}${digest}"


class HttpAuthenticator(Authenticator):
    def __init__(
        self,
        basic: HttpBasicConfig,
        jwt_verifiers: List[JWTVerifier],
        basic_verifier: Optional[BasicAuthVerifier] = None,
    ) -> None:
        self._basic = basic
        self._verifiers = jwt_verifiers
        self._basic_verifier = basic_verifier

    async def authenticate(self, request: Request) -> Optional[Credential]:
        header = request.headers.get("authorization")
        if not header:
            return None
        scheme, _, value = header.partition(" ")
        scheme_lower = scheme.lower()
        if scheme_lower == "bearer" and value:
            return await authenticate_bearer_jwt(
                value, self._verifiers, SecuritySchemeType.HTTP, AuthMethod.BEARER_JWT
            )
        if scheme_lower == "basic" and self._basic.enabled and value:
            return self._verify_basic(value)
        return None

    def _verify_basic(self, value: str) -> Credential:
        challenge = errors.basic_challenge(self._basic.realm)
        try:
            decoded = base64.b64decode(value).decode("utf-8")
        except (binascii.Error, UnicodeDecodeError) as exc:
            raise errors.unauthenticated(challenge) from exc
        username, separator, password = decoded.partition(":")
        if (
            not username
            or separator != ":"
            or self._basic_verifier is None
            or not self._basic_verifier.verify(username, password)
        ):
            raise errors.unauthenticated(challenge)
        return Credential(
            scheme=SecuritySchemeType.HTTP,
            method=AuthMethod.HTTP_BASIC,
            subject=username,
        )

    def carriers(self) -> Sequence[Carrier]:
        schemes = [Carrier(CredentialLocation.AUTHORIZATION_SCHEME, "bearer")]
        if self._basic.enabled:
            schemes.append(Carrier(CredentialLocation.AUTHORIZATION_SCHEME, "basic"))
        return tuple(schemes)

    def challenge(self) -> str:
        bearer = errors.bearer_challenge()
        if self._basic.enabled:
            return f"{bearer}, {errors.basic_challenge(self._basic.realm)}"
        return bearer
