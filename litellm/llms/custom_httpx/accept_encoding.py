"""
httpx keeps one zstd decompressor per response, so a streamed body that sends each event as its own
zstd frame fails on the second frame with `cannot use a decompressobj multiple times`. httpx offers
zstd whenever `zstandard` is importable (langchain, langsmith and `httpx[zstd]` all bring it in), so
the clients litellm builds offer every encoding httpx decodes except zstd.
"""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

_ZSTD: Final = "zstd"
_ALWAYS_DECODABLE: Final = "gzip, deflate"


def httpx_accept_encoding() -> str:
    try:
        from httpx._client import ACCEPT_ENCODING
    except ImportError:
        return _ALWAYS_DECODABLE
    return ACCEPT_ENCODING


def decodable_accept_encoding(advertised: str) -> str:
    supported: Final = tuple(
        part.strip() for part in advertised.split(",") if part.strip() and part.split(";")[0].strip() != _ZSTD
    )
    return ", ".join(supported) or "identity"


ACCEPT_ENCODING_HEADER: Final[Mapping[str, str]] = MappingProxyType(
    {"Accept-Encoding": decodable_accept_encoding(httpx_accept_encoding())}
)
