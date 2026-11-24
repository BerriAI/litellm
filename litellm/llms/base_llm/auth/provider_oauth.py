import hashlib
import json
import re
from collections.abc import Mapping
from threading import Lock
from time import monotonic
from typing import Annotated, Final, Protocol, TypeVar

import httpx
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, ValidationError

from litellm.caching.in_memory_cache import InMemoryCache
from litellm.exceptions import AuthenticationError
from litellm.llms.base_llm.auth.token_exchange import MAX_RESPONSE_BYTES, InsecureTokenUrl, validate_token_endpoint_url
from litellm.secret_managers.main import get_secret_str
from litellm.types.proxy.auth.special_headers import SpecialHeaders

_TIMEOUT_SECONDS: Final = 10.0
_LOCKS: Final = tuple(Lock() for _ in range(32))
_BEARER_TOKEN: Final = re.compile(r"[A-Za-z0-9._~+/-]+=*")
_AUTH_HEADERS: Final = frozenset(SpecialHeaders.litellm_credential_header_names()) | {"host"}
_HeaderValue = TypeVar("_HeaderValue")


class _TokenCache(Protocol):
    def get_cache(self, key: str) -> object: ...

    def set_cache(self, key: str, value: str, *, ttl: float) -> None: ...


def _new_token_cache() -> _TokenCache:
    return InMemoryCache(clock=monotonic)


_TOKENS: Final = _new_token_cache()


class _TokenResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    access_token: str
    expires_in: Annotated[FiniteFloat, Field(gt=0)] | None = None
    token_type: str = "Bearer"


def oauth_credential_free_headers(
    headers: Mapping[str, _HeaderValue] | None,
) -> dict[str, _HeaderValue]:  # mutable-ok: provider header APIs require a dict
    return {name: value for name, value in (headers or {}).items() if name.lower() not in _AUTH_HEADERS}


def _connection_value(params: Mapping[str, object], name: str) -> str | None:
    value: Final = params.get(name)
    if isinstance(value, str) and value.startswith("os.environ/"):
        return get_secret_str(value)
    return value if isinstance(value, str) else None


def _secure_url(value: str | None) -> bool:
    if not value:
        return False
    try:
        url: Final = httpx.URL(value)
        return bool(
            url.host and not url.userinfo and not isinstance(validate_token_endpoint_url(value), InsecureTokenUrl)
        )
    except (httpx.InvalidURL, ValueError):
        return False


def _fetch_token(token_url: str, client_id: str, client_secret: str, scope: str | None) -> _TokenResponse:
    response: Final = httpx.post(
        token_url,
        data={
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
            **({"scope": scope} if scope else {}),
        },
        timeout=_TIMEOUT_SECONDS,
        follow_redirects=False,
    )
    response.raise_for_status()
    if len(response.content) > MAX_RESPONSE_BYTES:
        raise ValueError("OAuth token response exceeded the size limit")
    token: Final = _TokenResponse.model_validate_json(response.content)
    if token.token_type.lower() != "bearer" or _BEARER_TOKEN.fullmatch(token.access_token) is None:
        raise ValueError("Expected a Bearer access token")
    return token


def provider_oauth_token(
    params: Mapping[str, object],
    *,
    api_base: str | None,
    provider: str,
    model: str,
) -> str | None:
    if params.get("oauth_client_credentials") is not True:
        return None

    token_url: Final = _connection_value(params, "oauth_token_url")
    client_id: Final = _connection_value(params, "oauth_client_id")
    client_secret: Final = _connection_value(params, "oauth_client_secret")
    scope: Final = _connection_value(params, "oauth_scope")
    if not token_url or not client_id or not client_secret:
        raise AuthenticationError(
            message="OAuth client credentials require oauth_token_url, oauth_client_id and oauth_client_secret.",
            llm_provider=provider,
            model=model,
        )
    if not _secure_url(token_url) or not _secure_url(api_base):
        raise AuthenticationError(
            message="OAuth client credentials require an explicit HTTPS api_base and oauth_token_url "
            "without URL credentials. HTTP is allowed on loopback.",
            llm_provider=provider,
            model=model,
        )

    cache_key: Final = hashlib.sha256(
        json.dumps((token_url, client_id, client_secret, scope, api_base)).encode()
    ).hexdigest()
    with _LOCKS[int(cache_key[:2], 16) % len(_LOCKS)]:
        cached: Final = _TOKENS.get_cache(cache_key)
        if isinstance(cached, str):
            return cached
        started: Final = monotonic()
        try:
            token: Final = _fetch_token(token_url, client_id, client_secret, scope)
        except httpx.HTTPStatusError as exc:
            raise AuthenticationError(
                message=f"OAuth client credentials token request returned HTTP {exc.response.status_code}.",
                llm_provider=provider,
                model=model,
            ) from None
        except (httpx.RequestError, httpx.InvalidURL, ValidationError, ValueError):
            raise AuthenticationError(
                message="OAuth client credentials exchange failed to return a valid Bearer access token.",
                llm_provider=provider,
                model=model,
            ) from None
        if token.expires_in is not None:
            remaining: Final = token.expires_in - (monotonic() - started)
            ttl: Final = remaining - min(30.0, token.expires_in * 0.1)
            if ttl > 0:
                _TOKENS.set_cache(cache_key, json.dumps(token.access_token), ttl=ttl)
        return token.access_token
