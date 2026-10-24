from collections.abc import Mapping
from typing import Final

HOP_HEADERS: Final = frozenset(
    (
        "host",
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    )
)


def clean_headers(headers: Mapping[str, str]) -> Mapping[str, str]:
    connection: Final = frozenset(token.strip().lower() for token in headers.get("connection", "").split(","))
    return {key: value for key, value in headers.items() if key.lower() not in HOP_HEADERS | connection}
