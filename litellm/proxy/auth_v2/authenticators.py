from __future__ import annotations

import base64
import binascii
import functools
import hashlib
import hmac
import secrets
from typing import Any, Callable, Dict, List, Optional, Protocol, runtime_checkable

import jwt
from fastapi import Request
from jwt import PyJWKClient
from jwt import decode as jwt_decode
from starlette.concurrency import run_in_threadpool

from . import errors
from .config import (
    ApiKeySchemeConfig,
    AuthConfig,
    HttpBasicConfig,
    MutualTLSConfig,
    OAuth2IntrospectionConfig,
    TrustedProxyConfig,
)
from .models import (
    AuthMethod,
    ClientCertificate,
    Credential,
    CredentialRef,
    SecuritySchemeType,
)
from .network import ip_in_trusted_proxies
from .oidc.config import OIDCProviderConfig
from .rbac import filter_claim_roles

AT_JWT_TYPES = {"at+jwt", "application/at+jwt"}


def _apply_role_policy(claims: Dict[str, Any], provider: OIDCProviderConfig) -> None:
    claims["roles"] = filter_claim_roles(
        claims.get("roles"), provider.allowed_roles, provider.allow_platform_roles
    )


@runtime_checkable
class Authenticator(Protocol):
    async def authenticate(self, request: Request) -> Optional[Credential]: ...

    def challenge(self) -> str: ...


@runtime_checkable
class BasicAuthVerifier(Protocol):
    def verify(self, username: str, password: str) -> bool: ...


_PBKDF2_ITERATIONS = 600_000


def hash_basic_password(password: str, salt: Optional[str] = None) -> str:
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode(), bytes.fromhex(salt), _PBKDF2_ITERATIONS
    ).hex()
    return f"pbkdf2_sha256${_PBKDF2_ITERATIONS}${salt}${digest}"


class InMemoryBasicAuthStore:
    def __init__(self, credentials: Dict[str, str]) -> None:
        self._credentials = credentials

    def verify(self, username: str, password: str) -> bool:
        stored = self._credentials.get(username)
        if stored is None:
            return False
        try:
            _algorithm, iterations, salt, expected = stored.split("$")
            candidate = hashlib.pbkdf2_hmac(
                "sha256", password.encode(), bytes.fromhex(salt), int(iterations)
            ).hex()
        except ValueError:
            return False
        return hmac.compare_digest(candidate, expected)


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
        import httpx

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
            raise errors.invalid_token("token verification failed") from exc


async def _verify_jwt_off_loop(
    verifier: JWTVerifier, token: str, *, require_at_jwt: Optional[bool] = None
) -> Dict[str, Any]:
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


async def _authenticate_bearer_jwt(
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
    _apply_role_policy(claims, verifier.provider)
    return _credential_from_claims(scheme, method, token, claims)


class APIKeyAuthenticator:
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

    def challenge(self) -> str:
        return ""


class HttpAuthenticator:
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
            return await _authenticate_bearer_jwt(
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

    def challenge(self) -> str:
        bearer = errors.bearer_challenge()
        if self._basic.enabled:
            return f"{bearer}, {errors.basic_challenge(self._basic.realm)}"
        return bearer


def _default_introspection_client() -> Any:
    from litellm.llms.custom_httpx.http_handler import get_async_httpx_client
    from litellm.types.llms.custom_http import httpxSpecialProvider

    return get_async_httpx_client(llm_provider=httpxSpecialProvider.Oauth2Check)


class OAuth2Authenticator:
    def __init__(
        self,
        jwt_verifiers: List[JWTVerifier],
        introspection: Optional[OAuth2IntrospectionConfig],
        client_factory: Optional[Callable[[], Any]] = None,
    ) -> None:
        self._verifiers = jwt_verifiers
        self._introspection = introspection
        self._client_factory = client_factory or _default_introspection_client

    async def authenticate(self, request: Request) -> Optional[Credential]:
        token = _extract_bearer(request)
        if token is None:
            return None
        if _looks_like_jwt(token):
            return await _authenticate_bearer_jwt(
                token,
                self._verifiers,
                SecuritySchemeType.OAUTH2,
                AuthMethod.BEARER_JWT,
                require_at_jwt=True,
            )
        if self._introspection is not None:
            return await self._introspect(token)
        raise errors.invalid_token()

    async def _introspect(self, token: str) -> Credential:
        config = self._introspection
        assert config is not None
        basic = base64.b64encode(
            f"{config.client_id}:{config.client_secret.get_secret_value()}".encode()
        ).decode()
        client = self._client_factory()
        response = await client.post(
            str(config.introspection_endpoint),
            data={"token": token},
            headers={"Authorization": f"Basic {basic}"},
            timeout=10.0,
        )
        if response.status_code != 200:
            raise errors.invalid_token("introspection failed")
        try:
            body = response.json()
        except ValueError as exc:
            raise errors.invalid_token("introspection failed") from exc
        if not isinstance(body, dict) or body.get("active") is not True:
            raise errors.invalid_token("token inactive")
        token_audience = _normalize_audience(body.get("aud"))
        if config.audience and not set(token_audience) & set(config.audience):
            raise errors.invalid_token("audience mismatch")
        if config.issuer is not None and body.get("iss") != config.issuer:
            raise errors.invalid_token("issuer mismatch")
        claims = {key: value for key, value in body.items() if key != "roles"}
        return Credential(
            scheme=SecuritySchemeType.OAUTH2,
            method=AuthMethod.OAUTH2_INTROSPECTION,
            subject=str(body.get(config.subject_field, "")),
            issuer=body.get("iss"),
            audience=token_audience,
            scopes=_split_scope(body.get("scope")),
            claims=claims,
        )

    def challenge(self) -> str:
        return errors.bearer_challenge()


class OIDCAuthenticator:
    def __init__(self, jwt_verifiers: List[JWTVerifier]) -> None:
        self._verifiers = jwt_verifiers

    async def authenticate(self, request: Request) -> Optional[Credential]:
        token = _extract_bearer(request)
        if token is None:
            return None
        return await _authenticate_bearer_jwt(
            token, self._verifiers, SecuritySchemeType.OPENID_CONNECT, AuthMethod.OIDC
        )

    def challenge(self) -> str:
        return errors.bearer_challenge()


class MutualTLSAuthenticator:
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
        if self._config.forwarded_subject_header:
            peer = request.client.host if request.client else None
            if not ip_in_trusted_proxies(peer, self._network):
                return None
            dn = request.headers.get(self._config.forwarded_subject_header)
            return ClientCertificate(subject_dn=dn) if dn else None
        tls = request.scope.get("extensions", {}).get("tls", {})
        dn = tls.get("client_cert_name")
        return ClientCertificate(subject_dn=dn) if dn else None

    def challenge(self) -> str:
        return ""


def build_authenticators(
    config: AuthConfig, *, basic_verifier: Optional[BasicAuthVerifier] = None
) -> List[Authenticator]:
    verifiers = [JWTVerifier(provider) for provider in config.oidc_providers]
    by_scheme: Dict[SecuritySchemeType, Authenticator] = {}
    if config.api_key is not None:
        by_scheme[SecuritySchemeType.API_KEY] = APIKeyAuthenticator(config.api_key)
    by_scheme[SecuritySchemeType.HTTP] = HttpAuthenticator(
        config.http_basic, verifiers, basic_verifier
    )
    by_scheme[SecuritySchemeType.OPENID_CONNECT] = OIDCAuthenticator(verifiers)
    by_scheme[SecuritySchemeType.OAUTH2] = OAuth2Authenticator(
        verifiers, config.oauth2_introspection
    )
    if config.mutual_tls.enabled:
        by_scheme[SecuritySchemeType.MUTUAL_TLS] = MutualTLSAuthenticator(
            config.mutual_tls, config.network
        )
    return [by_scheme[scheme] for scheme in config.scheme_order if scheme in by_scheme]
