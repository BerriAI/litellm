import hashlib
import hmac
from collections.abc import Mapping
from typing import Final


def encoded_path(value: str) -> str:
    safe: Final = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_.~/"
    return "".join(chr(byte) if byte in safe else f"%{byte:02X}" for byte in value.encode("utf-8"))


def signature(
    method: str, path: str, headers: Mapping[str, str], signed: str, body: bytes, secret: str, scope: str,
) -> tuple[str, str]:
    """AWS SigV4 equations, independent of botocore and LiteLLM's signer."""
    canonical_headers: Final = "".join(name + ":" + " ".join(headers[name].split()) + "\n" for name in signed.split(";"))
    canonical: Final = "\n".join((method, path, "", canonical_headers, signed, hashlib.sha256(body).hexdigest()))
    canonical_hash: Final = hashlib.sha256(canonical.encode()).hexdigest()
    date, region, service, terminator = scope.split("/")
    assert terminator == "aws4_request"
    key = ("AWS4" + secret).encode()
    for part in (date, region, service, terminator):
        key = hmac.new(key, part.encode(), hashlib.sha256).digest()
    to_sign: Final = "\n".join(("AWS4-HMAC-SHA256", headers["x-amz-date"], scope, canonical_hash))
    return canonical_hash, hmac.new(key, to_sign.encode(), hashlib.sha256).hexdigest()
