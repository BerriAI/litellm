import re
from dataclasses import dataclass
from enum import Enum
from typing import Final

_AWS4_PREFIX: Final = "AWS4-HMAC-SHA256"
_AWS4_BEARER_CREDENTIAL: Final = re.compile(r"Credential=Bearer\s+([^/\s,]+)")
_AWS4_CREDENTIAL: Final = re.compile(r"Credential=([^/\s,]+)")
_TOKEN_SCHEMES: Final = frozenset({"bearer", "basic"})
_WEBSOCKET_SUBPROTOCOL_KEY_PREFIX: Final = "openai-insecure-api-key."


class KeySource(Enum):
    CUSTOM_HEADER = "custom_header"
    LITELLM_HEADER = "x-litellm-api-key"
    AUTHORIZATION = "authorization"
    API_KEY = "api-key"
    X_API_KEY = "x-api-key"
    GOOGLE = "x-goog-api-key"
    APIM = "ocp-apim-subscription-key"
    QUERY = "key"
    PASS_THROUGH = "pass_through_endpoint"


@dataclass(frozen=True, slots=True)
class PresentedKeys:
    custom_header: str | None = None
    litellm_header: str | None = None
    authorization: str | None = None
    api_key: str | None = None
    x_api_key: str | None = None
    google: str | None = None
    apim: str | None = None
    query: str | None = None
    pass_through: str | None = None


@dataclass(frozen=True, slots=True)
class SelectedKey:
    source: KeySource
    presented: str
    token: str


class WebSocketDenial(Enum):
    NO_API_KEY = "no_api_key"
    INVALID_AUTHORIZATION = "invalid_authorization"


@dataclass(frozen=True, slots=True)
class WebSocketKeys:
    authorization: str | None = None
    api_key: str | None = None
    subprotocols: str | None = None


def _aws4_access_key(value: str) -> str | None:
    match: Final = _AWS4_BEARER_CREDENTIAL.search(value) or _AWS4_CREDENTIAL.search(value)
    return match.group(1) if match else None


def _split_scheme(value: str) -> tuple[str, str] | None:
    scheme, separator, credential = value.partition(" ")
    return (scheme.lower(), credential) if separator else None


def scheme_credential(value: str) -> str | None:
    if value.startswith(_AWS4_PREFIX):
        return _aws4_access_key(value)
    split: Final = _split_scheme(value)
    if split is None:
        return None
    scheme, credential = split
    return credential if scheme in _TOKEN_SCHEMES else None


def litellm_header_token(value: str) -> str:
    token: Final = scheme_credential(value)
    return value if token is None else token


def _raw_key(keys: PresentedKeys) -> SelectedKey | None:
    raw_sources: Final = (
        (KeySource.API_KEY, keys.api_key),
        (KeySource.X_API_KEY, keys.x_api_key),
        (KeySource.GOOGLE, keys.google),
        (KeySource.APIM, keys.apim),
        (KeySource.QUERY, keys.query or None),
        (KeySource.PASS_THROUGH, keys.pass_through),
    )
    return next(
        (
            SelectedKey(source=source, presented=value, token=value)
            for source, value in raw_sources
            if value is not None
        ),
        None,
    )


def select_api_key(keys: PresentedKeys) -> SelectedKey | None:
    if keys.custom_header is not None:
        return SelectedKey(
            source=KeySource.CUSTOM_HEADER,
            presented=keys.custom_header,
            token=scheme_credential(keys.custom_header) or "",
        )
    if keys.litellm_header is not None:
        return SelectedKey(
            source=KeySource.LITELLM_HEADER,
            presented=keys.litellm_header,
            token=litellm_header_token(keys.litellm_header),
        )
    if keys.authorization:
        return SelectedKey(
            source=KeySource.AUTHORIZATION,
            presented=keys.authorization,
            token=scheme_credential(keys.authorization) or "",
        )
    return _raw_key(keys)


def _subprotocol_api_key(subprotocols: str | None) -> str | None:
    protocols: Final = tuple(protocol.strip() for protocol in (subprotocols or "").split(","))
    return next(
        (
            protocol.removeprefix(_WEBSOCKET_SUBPROTOCOL_KEY_PREFIX)
            for protocol in protocols
            if protocol.startswith(_WEBSOCKET_SUBPROTOCOL_KEY_PREFIX)
        ),
        None,
    )


def select_websocket_api_key(keys: WebSocketKeys) -> str | WebSocketDenial:
    if keys.authorization:
        split: Final = _split_scheme(keys.authorization)
        if split is None or split[0] != "bearer":
            return WebSocketDenial.INVALID_AUTHORIZATION
        return split[1].strip()
    return keys.api_key or _subprotocol_api_key(keys.subprotocols) or WebSocketDenial.NO_API_KEY
