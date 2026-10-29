import json
from typing import Final

from typing_extensions import ReadOnly, TypedDict

from litellm.rust_bridge import _native


class OTLPError(TypedDict):
    message: ReadOnly[str]


def encode_error(message: str) -> bytes:
    return _native.trace_encode_error(message)


def encode_otlp_response(content_type: str | None, error: str | None = None) -> tuple[bytes, str]:
    media_type: Final = (content_type or "application/x-protobuf").split(";", 1)[0].strip().lower()
    if media_type == "application/json":
        response: Final[OTLPError] = {"message": error or ""}
        return (json.dumps(response).encode() if error else b"{}"), "application/json"
    if error is None:
        return b"", "application/x-protobuf"
    return encode_error(error), "application/x-protobuf"
