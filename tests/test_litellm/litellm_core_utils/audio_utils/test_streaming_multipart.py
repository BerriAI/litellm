import asyncio
import os
import sys
from typing import AsyncIterator

import pytest

sys.path.insert(0, os.path.abspath("../../../.."))

from litellm.litellm_core_utils.audio_utils.streaming_multipart import (
    FieldPartTooLarge,
    FilePartTooLarge,
    open_transcription_multipart,
)

BOUNDARY = b"----testboundary1234"


def build_body(fields: dict[str, str], filename: str, file_bytes: bytes, file_last: bool = True) -> bytes:
    field_parts = [
        (
            f"--{BOUNDARY.decode()}\r\n"
            f'Content-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'
        ).encode()
        for k, v in fields.items()
    ]
    file_part = (
        f"--{BOUNDARY.decode()}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: audio/wav\r\n\r\n"
    ).encode() + file_bytes + b"\r\n"
    epilogue = f"--{BOUNDARY.decode()}--\r\n".encode()
    ordered = ([*field_parts, file_part] if file_last else [file_part, *field_parts])
    return b"".join(ordered) + epilogue


async def chunked(body: bytes, size: int) -> AsyncIterator[bytes]:
    for i in range(0, len(body), size):
        yield body[i : i + size]


async def _drain(agen) -> list[bytes]:
    return [chunk async for chunk in agen]


@pytest.mark.asyncio
async def test_fields_extracted_and_file_streamed_across_chunk_boundaries():
    file_bytes = bytes(range(256)) * 40  # 10240 bytes, spans many small chunks
    body = build_body({"model": "my-whisper", "language": "en"}, "audio.wav", file_bytes)

    parsed = await open_transcription_multipart(chunked(body, 64), BOUNDARY, max_file_bytes=None)

    assert parsed.fields == {"model": "my-whisper", "language": "en"}
    assert parsed.filename == "audio.wav"
    assert parsed.file_content_type == "audio/wav"

    pieces = await _drain(parsed.stream())
    assert b"".join(pieces) == file_bytes
    # streamed incrementally, not one blob
    assert len(pieces) > 1


@pytest.mark.asyncio
async def test_tee_buffer_matches_and_survives_for_retry():
    file_bytes = os.urandom(5000)
    body = build_body({"model": "m"}, "a.wav", file_bytes)

    parsed = await open_transcription_multipart(chunked(body, 128), BOUNDARY, max_file_bytes=None)
    streamed = b"".join(await _drain(parsed.stream()))

    assert streamed == file_bytes
    assert parsed.exhausted is True
    # retry path: full bytes still available after streaming
    assert await parsed.getvalue() == file_bytes


@pytest.mark.asyncio
async def test_getvalue_without_streaming_drains_remainder():
    file_bytes = os.urandom(3333)
    body = build_body({"model": "m"}, "a.wav", file_bytes)

    parsed = await open_transcription_multipart(chunked(body, 100), BOUNDARY, max_file_bytes=None)
    # never call stream(); go straight to getvalue (unsupported-provider buffer path)
    assert await parsed.getvalue() == file_bytes


@pytest.mark.asyncio
async def test_size_limit_aborts_mid_stream():
    file_bytes = os.urandom(10000)
    body = build_body({"model": "m"}, "a.wav", file_bytes)

    parsed = await open_transcription_multipart(chunked(body, 256), BOUNDARY, max_file_bytes=4096)

    with pytest.raises(FilePartTooLarge):
        await _drain(parsed.stream())
    # sticky: a retry via getvalue() must re-raise, not silently return the truncated prefix
    with pytest.raises(FilePartTooLarge):
        await parsed.getvalue()


@pytest.mark.asyncio
async def test_field_after_file_is_captured_not_dropped():
    # A field after the file part (non-standard ordering) must still be parsed and the full file
    # streamed, so re-emitting fields after the file loses nothing (no regression vs request.form()).
    file_bytes = os.urandom(2000)
    body = build_body({"model": "m", "language": "en"}, "a.wav", file_bytes, file_last=False)

    parsed = await open_transcription_multipart(chunked(body, 128), BOUNDARY, max_file_bytes=None)
    streamed = b"".join(await _drain(parsed.stream()))

    assert streamed == file_bytes
    # the fields that followed the file are captured once the stream is drained
    assert parsed.fields == {"model": "m", "language": "en"}
    assert await parsed.getvalue() == file_bytes


@pytest.mark.asyncio
async def test_oversized_form_field_is_rejected():
    # Non-file fields are parsed during the auth pre-read, so an unbounded field must be rejected.
    huge = "x" * (2 * 1024 * 1024)
    body = build_body({"model": huge}, "a.wav", os.urandom(64))
    with pytest.raises(FieldPartTooLarge):
        await open_transcription_multipart(chunked(body, 4096), BOUNDARY, max_file_bytes=None)
