"""
The `Accept-Encoding` every litellm-built httpx client asks upstreams for.

httpx builds one zstd decompressor per response and reuses it for every chunk, so a body whose
chunks are separate zstd frames, which is how a streaming provider that compresses one frame per
event answers, dies on the second frame with `cannot use a decompressobj multiple times`. httpx
offers zstd whenever the optional `zstandard` package is importable, which any install carrying
langchain, langsmith or `httpx[zstd]` does, so litellm asks for everything httpx offers but that.
"""

import os
from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

_ZSTD_CONTENT_ENCODING: Final = "zstd"

_ALWAYS_DECODABLE_ACCEPT_ENCODING: Final = "gzip, deflate"

ACCEPT_ENCODING_OVERRIDE_ENV_VAR: Final = "LITELLM_ACCEPT_ENCODING"


def httpx_accept_encoding() -> str:
    """
    httpx's own value, which is private to httpx and tracks which optional decoder packages are
    installed. A release that renames it leaves litellm asking for the two encodings httpx can
    always decode rather than failing to import.
    """
    try:
        from httpx._client import ACCEPT_ENCODING
    except ImportError:
        return _ALWAYS_DECODABLE_ACCEPT_ENCODING

    return ACCEPT_ENCODING


def decodable_accept_encoding(advertised: str) -> str:
    supported: Final = tuple(
        part.strip()
        for part in advertised.split(",")
        if part.strip() and part.split(";")[0].strip() != _ZSTD_CONTENT_ENCODING
    )
    return ", ".join(supported) or "identity"


DECODABLE_ACCEPT_ENCODING: Final = decodable_accept_encoding(httpx_accept_encoding())


def accept_encoding_header() -> Mapping[str, str]:
    """
    Set `LITELLM_ACCEPT_ENCODING` to send something else, which is the way back to zstd for a
    deployment whose upstreams answer with one zstd frame per response rather than per event.
    """
    override: Final = os.environ.get(ACCEPT_ENCODING_OVERRIDE_ENV_VAR)

    return MappingProxyType({"Accept-Encoding": override if override else DECODABLE_ACCEPT_ENCODING})
