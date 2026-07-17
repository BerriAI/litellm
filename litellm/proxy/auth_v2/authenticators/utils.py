from __future__ import annotations

import functools
from typing import List, Optional

import httpx
import jwt
from fastapi import Request
from jwt import PyJWKClient
from jwt import decode as jwt_decode
from starlette.concurrency import run_in_threadpool

from litellm.proxy.auth_v2 import errors
from litellm.proxy.auth_v2.models import (
    AuthMethod,
    Credential,
    CredentialRef,
    SecuritySchemeType,
)
from litellm.proxy.auth_v2.config import OIDCProviderConfig
from litellm.proxy.auth_v2.authorization import filter_claim_roles
from litellm.proxy.auth_v2.authenticators.types import Claims

AT_JWT_TYPES = {"at+jwt", "application/at+jwt"}


def apply_role_policy(claims: Claims, provider: OIDCProviderConfig) -> None:
    claims["roles"] = filter_claim_roles(
        claims.get("roles"), provider.allowed_roles, provider.allow_platform_roles
    )


def extract_bearer(request: Request) -> Optional[str]:
    header = request.headers.get("authorization")
    if not header:
        return None
    scheme, _, value = header.partition(" ")
    if scheme.lower() != "bearer" or not value:
        return None
    return value


def looks_like_jwt(token: str) -> bool:
    return token.count(".") == 2


def normalize_audience(value: object) -> List[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value]
    return []


def split_scope(value: object) -> List[str]:
    return value.split() if isinstance(value, str) else []


def credential_from_claims(
    scheme: SecuritySchemeType,
    method: AuthMethod,
    token: str,
    claims: Claims,
) -> Credential:
    header = jwt.get_unverified_header(token)
    issuer = claims.get("iss")
    jti = claims.get("jti")
    return Credential(
        scheme=scheme,
        method=method,
        subject=str(claims.get("sub", "")),
        issuer=issuer if isinstance(issuer, str) else None,
        audience=normalize_audience(claims.get("aud")),
        scopes=split_scope(claims.get("scope")),
        claims=claims,
        credential_ref=CredentialRef(
            key_id=header.get("kid"), token_id=jti if isinstance(jti, str) else None
        ),
        subject_token=token,
    )


class JWTVerifier:
    def __init__(
        self,
        provider: OIDCProviderConfig,
        jwks_client: Optional[PyJWKClient] = None,
    ) -> None:
        self.provider = provider
        if jwks_client is not None:
            self._jwks_client = jwks_client
            return
        jwks_uri = (
            str(provider.jwks_uri) if provider.jwks_uri else self._discover_jwks()
        )
        self._jwks_client = PyJWKClient(
            jwks_uri,
            cache_keys=True,
            cache_jwk_set=True,
            lifespan=300,
            timeout=10,
        )

    def _discover_jwks(self) -> str:
        url = f"{self.provider.issuer.rstrip('/')}/.well-known/openid-configuration"
        response = httpx.get(url, timeout=10.0)
        response.raise_for_status()
        jwks_uri = response.json().get("jwks_uri")
        if not jwks_uri:
            raise ValueError(f"discovery document missing jwks_uri: {url}")
        return str(jwks_uri)

    def verify(self, token: str, *, require_at_jwt: Optional[bool] = None) -> Claims:
        enforce = (
            self.provider.require_at_jwt if require_at_jwt is None else require_at_jwt
        )
        if enforce:
            header = jwt.get_unverified_header(token)
            if str(header.get("typ", "")).lower() not in AT_JWT_TYPES:
                raise errors.invalid_token("token typ must be at+jwt")
        try:
            signing_key = self._jwks_client.get_signing_key_from_jwt(token)
            return jwt_decode(
                token,
                signing_key.key,
                algorithms=self.provider.algorithms,
                audience=self.provider.audience,
                issuer=self.provider.issuer,
                options={"verify_exp": True, "require": ["exp", "iss", "aud"]},
            )
        except jwt.PyJWTError as exc:
            raise errors.invalid_token("token verification failed") from exc


async def _verify_jwt_off_loop(
    verifier: JWTVerifier, token: str, *, require_at_jwt: Optional[bool] = None
) -> Claims:
    return await run_in_threadpool(
        functools.partial(verifier.verify, token, require_at_jwt=require_at_jwt)
    )


def _select_verifier(token: str, verifiers: List[JWTVerifier]) -> Optional[JWTVerifier]:
    if not verifiers:
        return None
    try:
        issuer = jwt.decode(token, options={"verify_signature": False}).get("iss")
    except jwt.PyJWTError:
        return None
    for verifier in verifiers:
        if verifier.provider.issuer == issuer:
            return verifier
    return None


async def authenticate_bearer_jwt(
    token: str,
    verifiers: List[JWTVerifier],
    scheme: SecuritySchemeType,
    method: AuthMethod,
    *,
    require_at_jwt: bool = False,
) -> Credential:
    verifier = _select_verifier(token, verifiers)
    if verifier is None:
        raise errors.invalid_token("no issuer match")
    claims = await _verify_jwt_off_loop(verifier, token, require_at_jwt=require_at_jwt)
    apply_role_policy(claims, verifier.provider)
    return credential_from_claims(scheme, method, token, claims)
