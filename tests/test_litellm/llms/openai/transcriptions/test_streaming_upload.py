import os
import sys
from typing import AsyncIterator

import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))

from litellm.litellm_core_utils.audio_utils.streaming_multipart import (
    open_transcription_multipart,
)
from litellm.llms.openai.transcriptions.streaming_upload import (
    _escape_header_param,
    multipart_content_type,
    new_boundary,
    stream_multipart_body,
)

INBOUND = b"----inbound99"


def _inbound_body(audio: bytes, filename: str) -> bytes:
    b = INBOUND.decode()
    return (
        (
            f"--{b}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
            f"Content-Type: audio/wav\r\n\r\n"
        ).encode()
        + audio
        + f"\r\n--{b}--\r\n".encode()
    )


async def _chunks(body: bytes) -> AsyncIterator[bytes]:
    yield body


async def _upload(audio: bytes, filename: str = "a.wav"):
    return await open_transcription_multipart(_chunks(_inbound_body(audio, filename)), INBOUND, None)


def test_escape_header_param_neutralizes_quotes_and_newlines():
    assert _escape_header_param('meeting"final.wav') == "meeting%22final.wav"
    assert _escape_header_param("a\r\nb") == "a%0D%0Ab"


@pytest.mark.asyncio
async def test_malicious_filename_cannot_break_the_header():
    audio = os.urandom(64)
    upload = await _upload(audio, filename='evil".wav')
    boundary = new_boundary()

    body = b"".join(
        [
            chunk
            async for chunk in stream_multipart_body(
                boundary, {"model": "m"}, upload.filename or "a", "audio/wav", upload
            )
        ]
    )
    header, _, _rest = body.partition(b"\r\n\r\n")
    # the raw double-quote is escaped, so the filename cannot inject an extra header attribute
    assert b'filename="evil%22.wav"' in body
    assert b'filename="evil".wav"' not in body


@pytest.mark.asyncio
async def test_list_field_becomes_repeated_bracket_parts():
    audio = os.urandom(32)
    upload = await _upload(audio)
    boundary = new_boundary()
    body = b"".join(
        [
            chunk
            async for chunk in stream_multipart_body(
                boundary, {"model": "m", "timestamp_granularities": ["word", "segment"]}, "a.wav", "audio/wav", upload
            )
        ]
    )
    assert body.count(b'name="timestamp_granularities[]"') == 2
    assert b"word" in body and b"segment" in body
    # the streamed file bytes are present between the preamble and the closing boundary
    assert audio in body
    assert body.rstrip().endswith(f"--{boundary}--".encode())


def test_multipart_content_type_carries_boundary():
    assert multipart_content_type("xyz") == "multipart/form-data; boundary=xyz"
