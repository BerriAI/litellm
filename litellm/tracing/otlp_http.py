"""OTLP/HTTP framing: request content encoding and the response body the exporter expects."""

import gzip
import json
import zlib
from io import BytesIO
from typing import Final

from typing_extensions import ReadOnly, TypedDict

from litellm.constants import OTLP_MAX_BODY_BYTES
from litellm.rust_bridge.trace.storage import encode_error


class InvalidOTLPPayloadError(ValueError):
    pass


class TracingPayloadTooLargeError(Exception):
    pass


class OTLPError(TypedDict):
    message: ReadOnly[str]


def decompress(body: bytes, content_encoding: str | None) -> bytes:
    if len(body) > OTLP_MAX_BODY_BYTES:
        raise TracingPayloadTooLargeError(f"OTLP body exceeds {OTLP_MAX_BODY_BYTES} bytes")
    if content_encoding is None or content_encoding.lower() == "identity":
        return body
    if content_encoding.lower() != "gzip":
        raise InvalidOTLPPayloadError("Unsupported OTLP content encoding")
    try:
        with gzip.GzipFile(fileobj=BytesIO(body)) as stream:
            payload: Final = stream.read(OTLP_MAX_BODY_BYTES + 1)
    except (EOFError, OSError, zlib.error) as error:
        raise InvalidOTLPPayloadError("Invalid OTLP gzip body") from error
    if len(payload) > OTLP_MAX_BODY_BYTES:
        raise TracingPayloadTooLargeError(f"OTLP body exceeds {OTLP_MAX_BODY_BYTES} bytes")
    return payload


def encode_otlp_response(content_type: str | None, error: str | None = None) -> tuple[bytes, str]:
    media_type: Final = (content_type or "application/x-protobuf").split(";", 1)[0].strip().lower()
    if media_type == "application/json":
        response: Final[OTLPError] = {"message": error or ""}
        return (json.dumps(response).encode() if error else b"{}"), "application/json"
    if error is None:
        return b"", "application/x-protobuf"
    return encode_error(error), "application/x-protobuf"
