from __future__ import annotations

from collections.abc import Iterable
from typing import Final
from urllib.parse import urlsplit, urlunsplit

PARITY_PROVIDER_HOST: Final = "parity-provider.invalid"

HOP_BY_HOP_HEADERS: Final[frozenset[str]] = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "trailers",
        "transfer-encoding",
        "upgrade",
    }
)

REQUEST_DROPPED_HEADERS: Final[frozenset[str]] = HOP_BY_HOP_HEADERS | {
    "host",
    "content-length",
    "accept-encoding",
}

RESPONSE_DROPPED_HEADERS: Final[frozenset[str]] = HOP_BY_HOP_HEADERS | {
    "content-encoding",
    "content-length",
    "set-cookie",
}


def connection_header_names(headers: Iterable[tuple[str, str]]) -> frozenset[str]:
    return frozenset(
        token.strip().lower()
        for name, value in headers
        if name.lower() == "connection"
        for token in value.split(",")
        if token.strip()
    )


def dropped_request_headers(headers: Iterable[tuple[str, str]]) -> frozenset[str]:
    materialized: Final = tuple(headers)
    return REQUEST_DROPPED_HEADERS | connection_header_names(materialized)


def dropped_response_headers(headers: Iterable[tuple[str, str]]) -> frozenset[str]:
    materialized: Final = tuple(headers)
    return RESPONSE_DROPPED_HEADERS | connection_header_names(materialized)


def is_streaming_response(content_type: str) -> bool:
    return "text/event-stream" in content_type.lower()


def normalized_response_header(name: str, value: str) -> str:
    if name.lower() not in {"location", "operation-location"}:
        return value
    parsed: Final = urlsplit(value)
    if not parsed.netloc:
        return value
    return urlunsplit(("http", PARITY_PROVIDER_HOST, parsed.path, parsed.query, parsed.fragment))


def local_response_header(name: str, value: str, provider_url: str) -> str:
    if name.lower() not in {"location", "operation-location"}:
        return value
    parsed: Final = urlsplit(value)
    if parsed.hostname != PARITY_PROVIDER_HOST:
        return value
    return f"{provider_url}{parsed.path}{'?' + parsed.query if parsed.query else ''}"
