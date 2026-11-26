import hashlib
import json
import re
from collections.abc import Mapping
from io import BytesIO
from threading import Lock
from time import monotonic
from typing import Annotated, Final, Protocol, TypeVar

import httpx
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, ValidationError
from typing_extensions import assert_never

from litellm.caching.in_memory_cache import InMemoryCache
from litellm.exceptions import AuthenticationError
from litellm.llms.base_llm.auth.token_exchange import (
    MAX_RESPONSE_BYTES,
    InsecureTokenUrl,
    validate_token_endpoint_url,
)
from litellm.llms.base_llm.auth.types import MalformedTokenResponse, TokenEndpointError, TokenTransportError
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


_TokenResult = _TokenResponse | MalformedTokenResponse | TokenEndpointError | TokenTransportError


def _content_encoding(headers: Mapping[str, str]) -> str:
    return headers.get("content-encoding", "identity")


def _parse_token_response(response: httpx.Response) -> _TokenResponse | MalformedTokenResponse:
    if _content_encoding(response.headers).lower() != "identity":
        return MalformedTokenResponse(detail="Expected an uncompressed OAuth token response")
    try:
        with BytesIO() as body:
            for chunk in response.iter_bytes(chunk_size=65536):
                if body.tell() + len(chunk) > MAX_RESPONSE_BYTES:
                    return MalformedTokenResponse(detail="OAuth token response exceeded the size limit")
                body.write(chunk)
            token: Final = _TokenResponse.model_validate_json(body.getvalue())
    except ValidationError:
        return MalformedTokenResponse(detail="OAuth token response is not valid JSON or has invalid fields")
    if token.token_type.lower() != "bearer" or _BEARER_TOKEN.fullmatch(token.access_token) is None:
        return MalformedTokenResponse(detail="Expected a Bearer access token")
    return token


def _fetch_token(token_url: str, client_id: str, client_secret: str, scope: str | None) -> _TokenResult:
    from litellm.llms.custom_httpx.http_handler import HTTPHandler

    try:
        with (
            HTTPHandler(timeout=_TIMEOUT_SECONDS, follow_redirects=False).client as client,
            client.stream(
                "POST",
                token_url,
                data={
                    "grant_type": "client_credentials",
                    "client_id": client_id,
                    "client_secret": client_secret,
                    **({"scope": scope} if scope else {}),
                },
                headers={"Accept-Encoding": "identity"},
            ) as response,
        ):
            if not response.is_success:
                return TokenEndpointError(status_code=response.status_code, redacted_body="")
            return _parse_token_response(response)
    except (httpx.RequestError, httpx.InvalidURL, OSError, ValueError):
        return TokenTransportError(detail="OAuth token request failed")


def _require_token(result: _TokenResult, provider: str, model: str) -> _TokenResponse:
    match result:
        case _TokenResponse():
            return result
        case TokenEndpointError(status_code=status):
            message = f"OAuth client credentials token request returned HTTP {status}."
        case MalformedTokenResponse() | TokenTransportError():
            message = "OAuth client credentials exchange failed to return a valid Bearer access token."
        case _:
            assert_never(result)
    raise AuthenticationError(message=message, llm_provider=provider, model=model)


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
        token: Final = _require_token(_fetch_token(token_url, client_id, client_secret, scope), provider, model)
        lifetime: Final = token.expires_in
        if lifetime is None:
            return token.access_token
        remaining: Final = lifetime - (monotonic() - started)
        ttl: Final = remaining - min(30.0, lifetime * 0.1)
        if ttl > 0:
            _TOKENS.set_cache(cache_key, json.dumps(token.access_token), ttl=ttl)
        return token.access_token
