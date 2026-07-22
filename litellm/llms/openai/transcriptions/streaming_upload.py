"""
Streaming multipart upload for OpenAI-compatible transcription providers.

Rebuilds the multipart/form-data body as an async generator so the file part streams to the
provider as it arrives from the client, instead of being fully buffered first. The OpenAI SDK has
no streaming-upload mode, so this path posts via httpx directly (through litellm's AsyncHTTPHandler,
preserving proxy/ssl behaviour). Providers accept the resulting `Transfer-Encoding: chunked` body
(verified against OpenAI and vLLM).
"""

import uuid
from typing import AsyncIterator, Mapping

from litellm.litellm_core_utils.audio_utils.streaming_multipart import (
    StreamingMultipartUpload,
)

FormValue = str | int | float | bool | list


def _escape_header_param(value: str) -> str:
    """Escape a multipart Content-Disposition param the way httpx does, so a client-controlled
    name/filename with a quote or newline can't break out of the header."""
    return value.replace("\\", "\\\\").replace('"', "%22").replace("\r", "%0D").replace("\n", "%0A")


def _field_part(boundary: str, name: str, value: str) -> bytes:
    return (
        f'--{boundary}\r\nContent-Disposition: form-data; name="{_escape_header_param(name)}"\r\n\r\n{value}\r\n'
    ).encode()


def _encode_field(boundary: str, name: str, value: FormValue) -> bytes:
    if isinstance(value, list):
        return b"".join(_field_part(boundary, f"{name}[]", str(item)) for item in value)
    return _field_part(boundary, name, str(value))


def _field_parts(boundary: str, fields: Mapping[str, FormValue]) -> bytes:
    return b"".join(_encode_field(boundary, name, value) for name, value in fields.items())


def multipart_content_type(boundary: str) -> str:
    return f"multipart/form-data; boundary={boundary}"


async def stream_multipart_body(
    boundary: str,
    fields: Mapping[str, FormValue],
    filename: str,
    file_content_type: str | None,
    upload: StreamingMultipartUpload,
) -> AsyncIterator[bytes]:
    # The file part goes first so any form field that follows the file in the source body is still
    # captured (multipart is order-independent per RFC 7578): the fields are emitted after the file
    # streams, by which point `upload.fields` has parsed the whole body.
    yield (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{_escape_header_param(filename)}"\r\n'
        f"Content-Type: {file_content_type or 'application/octet-stream'}\r\n\r\n"
    ).encode()
    async for chunk in upload.stream():
        yield chunk
    yield b"\r\n"
    # Merge in any field that only appeared after the file part in the source body.
    merged = {**fields, **{name: value for name, value in upload.fields.items() if name not in fields}}
    yield _field_parts(boundary, merged)
    yield f"--{boundary}--\r\n".encode()


def new_boundary() -> str:
    return f"----litellmstream{uuid.uuid4().hex}"
