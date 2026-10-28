import hashlib
import hmac
from collections.abc import Mapping
from typing import Final
from urllib.parse import parse_qsl

_UNRESERVED: Final = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_.~"


def _encoded(value: str, safe: bytes) -> str:
    return "".join(chr(byte) if byte in safe else f"%{byte:02X}" for byte in value.encode("utf-8"))


def encoded_path(value: str) -> str:
    return _encoded(value, _UNRESERVED + b"/")


def canonical_query(query: str) -> str:
    """The SigV4 canonical query string: pairs sorted by name, each name and value URI-encoded."""
    return "&".join(
        _encoded(name, _UNRESERVED) + "=" + _encoded(value, _UNRESERVED)
        for name, value in sorted(parse_qsl(query, keep_blank_values=True))
    )


def signature(
    method: str,
    path: str,
    headers: Mapping[str, str],
    signed: str,
    body: bytes,
    secret: str,
    scope: str,
    query: str = "",
) -> tuple[str, str]:
    """AWS SigV4 equations, independent of botocore and LiteLLM's signer; `query` is already canonical."""
    canonical_headers: Final = "".join(
        name + ":" + " ".join(headers[name].split()) + "\n" for name in signed.split(";")
    )
    canonical: Final = "\n".join((method, path, query, canonical_headers, signed, hashlib.sha256(body).hexdigest()))
    canonical_hash: Final = hashlib.sha256(canonical.encode()).hexdigest()
    date, region, service, terminator = scope.split("/")
    assert terminator == "aws4_request"
    key = ("AWS4" + secret).encode()
    for part in (date, region, service, terminator):
        key = hmac.new(key, part.encode(), hashlib.sha256).digest()
    to_sign: Final = "\n".join(("AWS4-HMAC-SHA256", headers["x-amz-date"], scope, canonical_hash))
    return canonical_hash, hmac.new(key, to_sign.encode(), hashlib.sha256).hexdigest()
