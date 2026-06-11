from __future__ import annotations

import base64
import binascii
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

import httpx
import jwt
from fastapi import Request
from jwt import PyJWKClient
from jwt import decode as jwt_decode

from . import errors
from .config import (
    ApiKeySchemeConfig,
    AuthConfig,
    HttpBasicConfig,
    MutualTlsConfig,
    OAuth2IntrospectionConfig,
    OidcProviderConfig,
)
from .models import (
    AuthMethod,
    ClientCertificate,
    Credential,
    CredentialRef,
    SecuritySchemeType,
)

AT_JWT_TYPES = {"at+jwt", "application/at+jwt"}


@runtime_checkable
class Authenticator(Protocol):
    scheme: SecuritySchemeType

    async def authenticate(self, request: Request) -> Optional[Credential]: ...

    def challenge(self) -> str: ...


def _extract_bearer(request: Request) -> Optional[str]:
    header = request.headers.get("authorization")
    if not header:
        return None
    scheme, _, value = header.partition(" ")
    if scheme.lower() != "bearer" or not value:
        return None
    return value


def _looks_like_jwt(token: str) -> bool:
    return token.count(".") == 2


def _normalize_audience(value: Any) -> List[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value]
    return []


def _split_scope(value: Any) -> List[str]:
    return value.split() if isinstance(value, str) else []


def _credential_from_claims(
    scheme: SecuritySchemeType,
    method: AuthMethod,
    token: str,
    claims: Dict[str, Any],
) -> Credential:
    header = jwt.get_unverified_header(token)
    return Credential(
        scheme=scheme,
        method=method,
        subject=str(claims.get("sub", "")),
        issuer=claims.get("iss"),
        audience=_normalize_audience(claims.get("aud")),
        scopes=_split_scope(claims.get("scope")),
        claims=claims,
        credential_ref=CredentialRef(
            key_id=header.get("kid"), token_id=claims.get("jti")
        ),
    )


class JwtVerifier:
    def __init__(
        self,
        provider: OidcProviderConfig,
        jwks_client: Optional[PyJWKClient] = None,
    ) -> None:
        self.provider = provider
        if jwks_client is not None:
            self._jwks_client = jwks_client
            return
        jwks_uri = (
            str(provider.jwks_uri) if provider.jwks_uri else self._discover_jwks()
        )
        self._jwks_client = PyJWKClient(jwks_uri, cache_keys=True)

    def _discover_jwks(self) -> str:
        url = f"{self.provider.issuer.rstrip('/')}/.well-known/openid-configuration"
        response = httpx.get(url, timeout=10.0)
        response.raise_for_status()
        jwks_uri = response.json().get("jwks_uri")
        if not jwks_uri:
            raise ValueError(f"discovery document missing jwks_uri: {url}")
        return str(jwks_uri)

    def verify(
        self, token: str, *, require_at_jwt: Optional[bool] = None
    ) -> Dict[str, Any]:
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
            raise errors.invalid_token(str(exc)) from exc


def _select_verifier(token: str, verifiers: List[JwtVerifier]) -> Optional[JwtVerifier]:
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


class ApiKeyAuthenticator:
    scheme = SecuritySchemeType.API_KEY

    def __init__(self, config: ApiKeySchemeConfig) -> None:
        self._header_name = config.header_name

    async def authenticate(self, request: Request) -> Optional[Credential]:
        raw = request.headers.get(self._header_name)
        if not raw:
            return None
        return Credential(
            scheme=self.scheme,
            method=AuthMethod.API_KEY,
            subject=raw,
            credential_ref=CredentialRef(key_id=raw[:10]),
            claims={"_raw_api_key": raw},
        )

    def challenge(self) -> str:
        return ""


class HttpAuthenticator:
    scheme = SecuritySchemeType.HTTP

    def __init__(
        self, basic: HttpBasicConfig, jwt_verifiers: List[JwtVerifier]
    ) -> None:
        self._basic = basic
        self._verifiers = jwt_verifiers

    async def authenticate(self, request: Request) -> Optional[Credential]:
        header = request.headers.get("authorization")
        if not header:
            return None
        scheme, _, value = header.partition(" ")
        scheme_lower = scheme.lower()
        if scheme_lower == "bearer" and value:
            return self._verify_bearer(value)
        if scheme_lower == "basic" and self._basic.enabled and value:
            return self._verify_basic(value)
        return None

    def _verify_bearer(self, token: str) -> Credential:
        verifier = _select_verifier(token, self._verifiers)
        if verifier is None:
            raise errors.invalid_token("no issuer match")
        claims = verifier.verify(token)
        return _credential_from_claims(
            self.scheme, AuthMethod.BEARER_JWT, token, claims
        )

    def _verify_basic(self, value: str) -> Credential:
        try:
            decoded = base64.b64decode(value).decode("utf-8")
        except (binascii.Error, UnicodeDecodeError) as exc:
            raise errors.unauthenticated(
                errors.basic_challenge(self._basic.realm)
            ) from exc
        username, _, password = decoded.partition(":")
        if not username:
            raise errors.unauthenticated(errors.basic_challenge(self._basic.realm))
        return Credential(
            scheme=self.scheme,
            method=AuthMethod.HTTP_BASIC,
            subject=username,
            claims={"_basic_password": password},
        )

    def challenge(self) -> str:
        bearer = errors.bearer_challenge()
        if self._basic.enabled:
            return f"{bearer}, {errors.basic_challenge(self._basic.realm)}"
        return bearer


class OAuth2Authenticator:
    scheme = SecuritySchemeType.OAUTH2

    def __init__(
        self,
        jwt_verifiers: List[JwtVerifier],
        introspection: Optional[OAuth2IntrospectionConfig],
    ) -> None:
        self._verifiers = jwt_verifiers
        self._introspection = introspection

    async def authenticate(self, request: Request) -> Optional[Credential]:
        token = _extract_bearer(request)
        if token is None:
            return None
        if _looks_like_jwt(token):
            return self._verify_at_jwt(token)
        if self._introspection is not None:
            return await self._introspect(token)
        raise errors.invalid_token()

    def _verify_at_jwt(self, token: str) -> Credential:
        verifier = _select_verifier(token, self._verifiers)
        if verifier is None:
            raise errors.invalid_token("no issuer match")
        claims = verifier.verify(token, require_at_jwt=True)
        return _credential_from_claims(
            self.scheme, AuthMethod.BEARER_JWT, token, claims
        )

    async def _introspect(self, token: str) -> Credential:
        config = self._introspection
        assert config is not None
        from litellm.llms.custom_httpx.http_handler import get_async_httpx_client
        from litellm.types.llms.custom_http import httpxSpecialProvider

        basic = base64.b64encode(
            f"{config.client_id}:{config.client_secret.get_secret_value()}".encode()
        ).decode()
        client = get_async_httpx_client(llm_provider=httpxSpecialProvider.Oauth2Check)
        response = await client.post(
            str(config.introspection_endpoint),
            data={"token": token},
            headers={"Authorization": f"Basic {basic}"},
            timeout=10.0,
        )
        if response.status_code != 200:
            raise errors.invalid_token("introspection failed")
        body = response.json()
        if not body.get("active"):
            raise errors.invalid_token("token inactive")
        return Credential(
            scheme=self.scheme,
            method=AuthMethod.OAUTH2_INTROSPECTION,
            subject=str(body.get(config.subject_field, "")),
            issuer=body.get("iss"),
            audience=_normalize_audience(body.get("aud")),
            scopes=_split_scope(body.get("scope")),
            claims=body,
        )

    def challenge(self) -> str:
        return errors.bearer_challenge()


class OidcAuthenticator:
    scheme = SecuritySchemeType.OPENID_CONNECT

    def __init__(self, jwt_verifiers: List[JwtVerifier]) -> None:
        self._verifiers = jwt_verifiers

    async def authenticate(self, request: Request) -> Optional[Credential]:
        token = _extract_bearer(request)
        if token is None:
            return None
        verifier = _select_verifier(token, self._verifiers)
        if verifier is None:
            raise errors.invalid_token("no issuer match")
        claims = verifier.verify(token)
        return _credential_from_claims(self.scheme, AuthMethod.OIDC, token, claims)

    def challenge(self) -> str:
        return errors.bearer_challenge()


class MutualTlsAuthenticator:
    scheme = SecuritySchemeType.MUTUAL_TLS

    def __init__(self, config: MutualTlsConfig) -> None:
        self._config = config

    async def authenticate(self, request: Request) -> Optional[Credential]:
        cert = self._read_client_cert(request)
        if cert is None:
            return None
        return Credential(
            scheme=self.scheme,
            method=AuthMethod.MUTUAL_TLS,
            subject=cert.subject_dn,
            client_certificate=cert,
        )

    def _read_client_cert(self, request: Request) -> Optional[ClientCertificate]:
        if self._config.forwarded_subject_header:
            dn = request.headers.get(self._config.forwarded_subject_header)
            return ClientCertificate(subject_dn=dn) if dn else None
        tls = request.scope.get("extensions", {}).get("tls", {})
        dn = tls.get("client_cert_name")
        return ClientCertificate(subject_dn=dn) if dn else None

    def challenge(self) -> str:
        return ""


def build_authenticators(config: AuthConfig) -> List[Authenticator]:
    verifiers = [JwtVerifier(provider) for provider in config.oidc_providers]
    by_scheme: Dict[SecuritySchemeType, Authenticator] = {}
    if config.api_key is not None:
        by_scheme[SecuritySchemeType.API_KEY] = ApiKeyAuthenticator(config.api_key)
    by_scheme[SecuritySchemeType.HTTP] = HttpAuthenticator(config.http_basic, verifiers)
    by_scheme[SecuritySchemeType.OPENID_CONNECT] = OidcAuthenticator(verifiers)
    by_scheme[SecuritySchemeType.OAUTH2] = OAuth2Authenticator(
        verifiers, config.oauth2_introspection
    )
    if config.mutual_tls.enabled:
        by_scheme[SecuritySchemeType.MUTUAL_TLS] = MutualTlsAuthenticator(
            config.mutual_tls
        )
    return [by_scheme[scheme] for scheme in config.scheme_order if scheme in by_scheme]
