import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import (
    Final,
    Literal,
    Protocol,
    TypeAlias,
    cast,  # noqa: TID251  # adapter protocols cover untyped cache and HTTP handler methods
)

import httpx
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from litellm.caching.in_memory_cache import InMemoryCache
from litellm.constants import OAUTH_TOKEN_EXCHANGE_CACHE_SAFETY_MARGIN_SECONDS
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler

TokenExchangeProfile: TypeAlias = Literal["rfc8693", "jwt_bearer_obo"]

SUBJECT_TOKEN_TYPE_ACCESS_TOKEN: Final = "urn:ietf:params:oauth:token-type:access_token"
_RFC8693_GRANT_TYPE: Final = "urn:ietf:params:oauth:grant-type:token-exchange"
_JWT_BEARER_GRANT_TYPE: Final = "urn:ietf:params:oauth:grant-type:jwt-bearer"
_TOKEN_CACHE: Final = InMemoryCache(max_size_in_memory=1000, default_ttl=600)


class _TokenCache(Protocol):
    def set_cache(self, *, key: str, value: str, ttl: int) -> None: ...

    def get_cache(self, *, key: str) -> object: ...


class _SyncTokenExchangeClient(Protocol):
    def post(
        self,
        url: str,
        *,
        data: Mapping[str, str],
        timeout: float | httpx.Timeout | None,
    ) -> httpx.Response: ...


class _AsyncTokenExchangeClient(Protocol):
    async def post(
        self,
        url: str,
        *,
        data: Mapping[str, str],
        timeout: float | httpx.Timeout | None,
    ) -> httpx.Response: ...


class OAuthTokenExchangeError(Exception):
    def __init__(self, status_code: int, message: str) -> None:
        self.status_code: Final = status_code
        self.message: Final = message
        super().__init__(message)


class _OAuthTokenResponse(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True, strict=True)

    access_token: str
    expires_in: int = Field(gt=0)


class _OAuthErrorResponse(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True, strict=True)

    error: str | None = None
    error_description: str | None = None


_TOKEN_RESPONSE_ADAPTER: Final = TypeAdapter(_OAuthTokenResponse)
_ERROR_RESPONSE_ADAPTER: Final = TypeAdapter(_OAuthErrorResponse)


def _is_valid_token_endpoint(value: str) -> bool:
    try:
        endpoint: Final = httpx.URL(value)
    except httpx.InvalidURL:
        return False
    return endpoint.scheme.casefold() == "https" and bool(endpoint.host)


@dataclass(frozen=True, slots=True, repr=False)
class OAuthTokenExchangeConfig:
    token_endpoint: str
    client_id: str
    client_secret: str
    profile: TokenExchangeProfile
    scope: str | None
    audience: str | None

    def __post_init__(self) -> None:
        if not _is_valid_token_endpoint(self.token_endpoint):
            raise OAuthTokenExchangeError(
                status_code=400,
                message="token_exchange_endpoint must be an HTTPS URL with a host",
            )
        if self.profile not in ("rfc8693", "jwt_bearer_obo"):
            raise OAuthTokenExchangeError(status_code=400, message="Unsupported OAuth token exchange profile")
        if self.profile == "jwt_bearer_obo" and (self.scope is None or not self.scope.strip()):
            raise OAuthTokenExchangeError(
                status_code=400,
                message="jwt_bearer_obo requires a non-empty token exchange scope",
            )


def redact_sensitive_values(message: str, sensitive_values: tuple[str, ...]) -> str:
    redacted_values: Final = tuple(sorted((value for value in sensitive_values if value), key=len, reverse=True))
    if not redacted_values:
        return message
    pattern: Final = re.compile("|".join(re.escape(value) for value in redacted_values))
    return pattern.sub("[REDACTED]", message)


def _cache_key(config: OAuthTokenExchangeConfig, subject_token: str) -> str:
    key_material: Final = json.dumps(
        {
            "token_endpoint": config.token_endpoint,
            "client_id": config.client_id,
            "client_secret": config.client_secret,
            "profile": config.profile,
            "scope": config.scope,
            "audience": config.audience,
            "subject_token": subject_token,
        },
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(key_material.encode("utf-8")).hexdigest()


def build_token_exchange_form(
    config: OAuthTokenExchangeConfig,
    subject_token: str,
) -> dict[str, str]:  # mutable-ok: the form builder contract requires an HTTP form dictionary
    match config.profile:
        case "rfc8693":
            return {
                "grant_type": _RFC8693_GRANT_TYPE,
                "client_id": config.client_id,
                "client_secret": config.client_secret,
                "subject_token": subject_token,
                "subject_token_type": SUBJECT_TOKEN_TYPE_ACCESS_TOKEN,
                **({"scope": config.scope} if config.scope else {}),
                **({"audience": config.audience} if config.audience else {}),
            }
        case "jwt_bearer_obo":
            if config.scope is None or not config.scope.strip():
                raise OAuthTokenExchangeError(
                    status_code=400,
                    message="jwt_bearer_obo requires a non-empty token exchange scope",
                )
            return {
                "grant_type": _JWT_BEARER_GRANT_TYPE,
                "client_id": config.client_id,
                "client_secret": config.client_secret,
                "assertion": subject_token,
                "scope": config.scope,
                "requested_token_use": "on_behalf_of",
            }


def _error_message(response: httpx.Response, sensitive_values: tuple[str, ...]) -> str:
    try:
        error_response: Final = _ERROR_RESPONSE_ADAPTER.validate_python(response.json())
    except (ValidationError, ValueError):
        return redact_sensitive_values(
            "OAuth token exchange failed: unknown_error: invalid response",
            sensitive_values,
        )
    error: Final = error_response.error or "unknown_error"
    description: Final = error_response.error_description or "no description provided"
    return redact_sensitive_values(
        f"OAuth token exchange failed: {error}: {description}",
        sensitive_values,
    )


def _response_token(response: httpx.Response, config: OAuthTokenExchangeConfig, subject_token: str) -> str:
    try:
        token_response: Final = _TOKEN_RESPONSE_ADAPTER.validate_python(response.json())
    except (ValidationError, ValueError):
        raise OAuthTokenExchangeError(
            status_code=502,
            message="OAuth token exchange returned an invalid token response",
        ) from None
    effective_ttl: Final = token_response.expires_in - OAUTH_TOKEN_EXCHANGE_CACHE_SAFETY_MARGIN_SECONDS
    if effective_ttl > 0:
        cache: Final = cast(_TokenCache, _TOKEN_CACHE)  # cast-ok: InMemoryCache exposes untyped cache methods
        cache.set_cache(
            key=_cache_key(config, subject_token),
            value=token_response.access_token,
            ttl=effective_ttl,
        )
    return token_response.access_token


def _error_status_code(status_code: int) -> int:
    if status_code == 429:
        return status_code
    if 400 <= status_code < 500:
        return 401
    return status_code


def _post(
    client: HTTPHandler,
    config: OAuthTokenExchangeConfig,
    request_data: Mapping[str, str],
    timeout: float | httpx.Timeout | None,
) -> httpx.Response:
    try:
        http_client: Final = cast(  # cast-ok: HTTPHandler.post has an untyped response contract
            _SyncTokenExchangeClient,
            client,
        )
        return http_client.post(config.token_endpoint, data=dict(request_data), timeout=timeout)
    except httpx.HTTPStatusError as error:
        return error.response
    except httpx.HTTPError:
        raise OAuthTokenExchangeError(
            status_code=502,
            message="OAuth token exchange request failed",
        ) from None


async def _apost(
    client: AsyncHTTPHandler,
    config: OAuthTokenExchangeConfig,
    request_data: Mapping[str, str],
    timeout: float | httpx.Timeout | None,
) -> httpx.Response:
    try:
        http_client: Final = cast(  # cast-ok: AsyncHTTPHandler.post has an untyped response contract
            _AsyncTokenExchangeClient,
            client,
        )
        return await http_client.post(config.token_endpoint, data=dict(request_data), timeout=timeout)
    except httpx.HTTPStatusError as error:
        return error.response
    except httpx.HTTPError:
        raise OAuthTokenExchangeError(
            status_code=502,
            message="OAuth token exchange request failed",
        ) from None


def exchange_token(
    client: HTTPHandler,
    config: OAuthTokenExchangeConfig,
    subject_token: str,
    timeout: float | httpx.Timeout | None = None,
) -> str:
    cache_key: Final = _cache_key(config, subject_token)
    cache: Final = cast(_TokenCache, _TOKEN_CACHE)  # cast-ok: InMemoryCache exposes untyped cache methods
    cached_token: Final = cache.get_cache(key=cache_key)
    if isinstance(cached_token, str):
        return cached_token
    response: Final = _post(
        client=client,
        config=config,
        request_data=build_token_exchange_form(config, subject_token),
        timeout=timeout,
    )
    if response.status_code != 200:
        raise OAuthTokenExchangeError(
            status_code=_error_status_code(response.status_code),
            message=_error_message(
                response=response,
                sensitive_values=(config.client_secret, subject_token),
            ),
        )
    return _response_token(response=response, config=config, subject_token=subject_token)


async def aexchange_token(
    client: AsyncHTTPHandler,
    config: OAuthTokenExchangeConfig,
    subject_token: str,
    timeout: float | httpx.Timeout | None = None,
) -> str:
    cache_key: Final = _cache_key(config, subject_token)
    cache: Final = cast(_TokenCache, _TOKEN_CACHE)  # cast-ok: InMemoryCache exposes untyped cache methods
    cached_token: Final = cache.get_cache(key=cache_key)
    if isinstance(cached_token, str):
        return cached_token
    response: Final = await _apost(
        client=client,
        config=config,
        request_data=build_token_exchange_form(config, subject_token),
        timeout=timeout,
    )
    if response.status_code != 200:
        raise OAuthTokenExchangeError(
            status_code=_error_status_code(response.status_code),
            message=_error_message(
                response=response,
                sensitive_values=(config.client_secret, subject_token),
            ),
        )
    return _response_token(response=response, config=config, subject_token=subject_token)
