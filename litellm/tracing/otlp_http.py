import json
from typing import Final, Protocol, runtime_checkable

from pydantic import ConfigDict, TypeAdapter
from typing_extensions import ReadOnly, TypedDict

from litellm.rust_bridge.loader import get_native_bridge


@runtime_checkable
class NativeOtlpError(Protocol):
    def trace_encode_error(self, message: str) -> bytes: ...


_NATIVE: Final[TypeAdapter[NativeOtlpError]] = TypeAdapter(
    NativeOtlpError, config=ConfigDict(arbitrary_types_allowed=True)
)


class OTLPError(TypedDict):
    message: ReadOnly[str]


def encode_error(message: str) -> bytes:
    native: Final = get_native_bridge()
    return b"" if native is None else _NATIVE.validate_python(native).trace_encode_error(message)


def encode_otlp_response(content_type: str | None, error: str | None = None) -> tuple[bytes, str]:
    media_type: Final = (content_type or "application/x-protobuf").split(";", 1)[0].strip().lower()
    if media_type == "application/json":
        response: Final[OTLPError] = {"message": error or ""}
        return (json.dumps(response).encode() if error else b"{}"), "application/json"
    if error is None:
        return b"", "application/x-protobuf"
    return encode_error(error), "application/x-protobuf"
