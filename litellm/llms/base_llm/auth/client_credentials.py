"""Fetches a fresh RFC 6749 client_credentials assertion for Anthropic's ``keycloak`` identity
source: LiteLLM authenticates to Keycloak as its own confidential client and presents the
resulting ``access_token`` as the workload assertion (Phase 1 decision 2).

The client secret is the operator-supplied pointer at ``KeycloakSource.client_secret_ref``,
resolved the same way every other WIF secret pointer already is (env, a Credential, or whatever
secret manager ``litellm.secret_manager_client`` is globally configured to, Vault included).
Every fetch is a fresh HTTP POST; nothing here caches a fetched token, since the outer
token-exchange engine already caches the Anthropic token it buys with one -- see decision 2's
"no Keycloak-side cache" ruling.
"""

import base64
import threading
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, TypeAlias
from urllib.parse import urlencode

import httpx
from pydantic import BaseModel, SecretStr, ValidationError
from typing_extensions import assert_never

from litellm.llms.base_llm.auth.identity_source import KeycloakSource
from litellm.llms.base_llm.auth.token_exchange import (
    MAX_RESPONSE_BYTES,
    redact_oauth_error_body,
    validate_token_endpoint_url,
)
from litellm.llms.base_llm.auth.types import InsecureTokenUrl, SyncTokenPoster

if TYPE_CHECKING:
    from litellm.llms.custom_httpx.http_handler import HTTPHandler

SecretReader: TypeAlias = Callable[[str], str | None]  # mutable-ok: Callable param-list syntax, not a list

_GRANT_TYPE: Final = "client_credentials"
_TIMEOUT_SECONDS: Final = 30.0
_FORM_CONTENT_TYPE: Final = "application/x-www-form-urlencoded"


class _ClientCredentialsResponse(BaseModel):
    access_token: str


def _default_secret_reader(ref: str) -> str | None:
    from litellm.secret_managers.main import get_secret_str

    return get_secret_str(ref)


class _HttpxSyncKeycloakPoster:
    """Dedicated HTTPHandler for the Keycloak token POST: no ``logging_obj`` (so litellm's
    request/response logging never sees the client secret or the fetched token), redirects
    disabled. A separate instance from the outer engine's own poster, since this is a genuinely
    new HTTP call site whose no-logging guarantee must be built here, not assumed inherited."""

    def __init__(self) -> None:
        self._lock: Final = threading.Lock()
        self._handler: HTTPHandler | None = None

    def _handler_instance(self) -> "HTTPHandler":
        from litellm.llms.custom_httpx.http_handler import HTTPHandler

        with self._lock:
            if self._handler is None:
                handler: Final = HTTPHandler(timeout=httpx.Timeout(timeout=30.0, connect=5.0))
                handler.client.follow_redirects = False
                self._handler = handler
            return self._handler

    def post(self, url: str, *, content: bytes, headers: Mapping[str, str], timeout: float) -> httpx.Response:
        try:
            response: Final[httpx.Response | None] = self._handler_instance().post(  # pyright: ignore[reportUnknownMemberType]  # HTTPHandler.post is legacy-untyped; the result is validated below
                url,
                content=content,
                headers=dict(headers),  # mutable-ok: HTTPHandler.post requires a concrete dict
                timeout=timeout,
            )
        except httpx.HTTPStatusError as e:
            return e.response
        if response is None:
            raise httpx.TransportError("keycloak token endpoint returned no response")
        return response


_DEFAULT_POSTER: Final[SyncTokenPoster] = _HttpxSyncKeycloakPoster()


def _basic_auth_header(client_id: str, client_secret: str) -> str:
    return "Basic " + base64.b64encode(f"{client_id}:{client_secret}".encode()).decode("ascii")


def _prepared_request(config: KeycloakSource, client_secret: str) -> tuple[bytes, Mapping[str, str]]:
    scope_field: Final[Mapping[str, str]] = (
        MappingProxyType({"scope": config.scope}) if config.scope else MappingProxyType({})
    )
    match config.auth_method:
        case "client_secret_basic":
            return (
                urlencode(MappingProxyType({"grant_type": _GRANT_TYPE, **scope_field})).encode(),
                MappingProxyType(
                    {
                        "content-type": _FORM_CONTENT_TYPE,
                        "authorization": _basic_auth_header(config.client_id, client_secret),
                    }
                ),
            )
        case "client_secret_post":
            return (
                urlencode(
                    MappingProxyType(
                        {
                            "grant_type": _GRANT_TYPE,
                            "client_id": config.client_id,
                            "client_secret": client_secret,
                            **scope_field,
                        }
                    )
                ).encode(),
                MappingProxyType({"content-type": _FORM_CONTENT_TYPE}),
            )
        case _:
            assert_never(config.auth_method)


def _resolve_client_secret(config: KeycloakSource, secret_reader: SecretReader) -> str:
    secret: Final = secret_reader(config.client_secret_ref)
    if not secret:
        raise ValueError(f"keycloak client secret {config.client_secret_ref} could not be read")
    return secret


def _endpoint_error_message(config: KeycloakSource, response: httpx.Response, client_secret: str) -> str:
    endpoint_error: Final = redact_oauth_error_body(response.status_code, response.text, SecretStr(client_secret))
    return f"keycloak token endpoint {config.token_url} returned HTTP {endpoint_error.status_code}: {endpoint_error.redacted_body}"


def _parse_success_body(response: httpx.Response) -> str:
    if len(response.content) > MAX_RESPONSE_BYTES:
        raise ValueError("keycloak token response exceeded the size cap")
    try:
        parsed: Final = _ClientCredentialsResponse.model_validate_json(response.content)
    except ValidationError as e:
        raise ValueError("keycloak token response failed schema validation") from e
    token: Final = parsed.access_token.strip()
    if not token:
        raise ValueError("keycloak token response carried an empty access_token")
    return token


def fetch_keycloak_assertion(
    config: KeycloakSource,
    *,
    poster: SyncTokenPoster = _DEFAULT_POSTER,
    secret_reader: SecretReader = _default_secret_reader,
) -> str:
    """POSTs one fresh client_credentials grant and returns the resulting ``access_token`` as the
    workload assertion; the caller must not cache the result -- see the module docstring."""
    match validate_token_endpoint_url(config.token_url):
        case InsecureTokenUrl(host=host):
            raise ValueError(f"keycloak token_url must use https; refusing to send the client secret to host {host!r}")
        case _:
            pass
    client_secret: Final = _resolve_client_secret(config, secret_reader)
    content, headers = _prepared_request(config, client_secret)
    try:
        response: Final = poster.post(config.token_url, content=content, headers=headers, timeout=_TIMEOUT_SECONDS)
    except Exception as e:  # noqa: BLE001  # injected posters may raise beyond httpx; every failure becomes a ValueError
        raise ValueError(f"could not reach the keycloak token endpoint {config.token_url}: {type(e).__name__}") from e
    if not 200 <= response.status_code < 300:
        raise ValueError(_endpoint_error_message(config, response, client_secret))
    return _parse_success_body(response)


def keycloak_assertion_source(
    config: KeycloakSource,
    *,
    poster: SyncTokenPoster = _DEFAULT_POSTER,
    secret_reader: SecretReader = _default_secret_reader,
) -> Callable[[], str]:
    """A zero-arg closure that fetches fresh on every call: the shape an ``oidc/keycloak/...``
    ref dispatches to once wired into ``TokenExchangeSpec.assertion_source`` (Phase 1 decision 7)
    -- the caller parses the config and closes this function over it, with no registry involved."""
    return lambda: fetch_keycloak_assertion(config, poster=poster, secret_reader=secret_reader)
