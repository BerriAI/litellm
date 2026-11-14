"""Shared helpers for the `test_dec1_media_*` contracts: synthetic media bytes, multipart decoding of what the
proxy sent upstream, scripted provider replies, and the spend rows a media call leaves behind."""

from __future__ import annotations

import io
import json
import struct
import wave
import zlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from email.message import Message
from email.parser import BytesParser
from email.policy import HTTP
from hashlib import sha256
from types import MappingProxyType
from typing import Final, cast

import httpx
from integration._support.client import JSON_OBJECT, Gateway, eventually, object_value, string_value
from integration._support.database import read_rows
from integration._support.wire import Reply, Request, Wire
from pydantic import JsonValue

OPENAI_KEY: Final = "synthetic-openai-key"
AZURE_KEY: Final = "synthetic-azure-key"
PROMPT: Final = "a lighthouse made of glass at dawn"
B64_IMAGE: Final = "aW50ZWdyYXRpb24taW1hZ2U="
SPEECH_BYTES: Final = bytes(range(256)) * 16
MP4_BYTES: Final = b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom" + bytes(range(256)) * 4
PDF_BYTES: Final = b"%PDF-1.4\n%integration\n1 0 obj<<>>endobj\ntrailer<<>>\n%%EOF\n"
SPEND_COLUMNS: Final = (
    "request_id, call_type, spend, model, model_group, custom_llm_provider, status, "
    "prompt_tokens, completion_tokens, total_tokens"
)


def _png_chunk(chunk_type: bytes, chunk_data: bytes) -> bytes:
    checksum: Final = zlib.crc32(chunk_type + chunk_data) & 0xFFFFFFFF
    return struct.pack(">I", len(chunk_data)) + chunk_type + chunk_data + struct.pack(">I", checksum)


def png(pixel: tuple[int, int, int, int] = (255, 0, 0, 255)) -> bytes:
    """A valid 1x1 RGBA PNG whose single pixel is ``pixel``, so two uploads can be told apart byte for byte."""
    header: Final = struct.pack(">IIBBBBB", 1, 1, 8, 6, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", header)
        + _png_chunk(b"IDAT", zlib.compress(b"\x00" + bytes(pixel)))
        + _png_chunk(b"IEND", b"")
    )


def silent_wav(seconds: float, marker: bytes = b"") -> bytes:
    """A 16 kHz mono PCM WAV of ``seconds`` of silence; ``marker`` is appended so each test uploads unique bytes."""
    output: Final = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\x00\x00" * round(16000 * seconds))
    return output.getvalue() + marker


@dataclass(frozen=True, slots=True)
class FilePart:
    name: str
    filename: str
    content_type: str
    content: bytes


def multipart_parts(request: Request) -> tuple[Message, ...]:
    envelope: Final = f"content-type: {request.headers['content-type']}\r\n\r\n".encode() + request.body
    parsed: Final = BytesParser(policy=HTTP).parsebytes(envelope)
    assert parsed.is_multipart(), request.headers["content-type"]
    return tuple(parsed.iter_parts())


def text_fields(request: Request) -> tuple[tuple[str, str], ...]:
    """Every non-file multipart field the upstream received, sorted by name then value."""
    return tuple(
        sorted(
            (
                cast(str, part.get_param("name", header="content-disposition")),
                cast(bytes, part.get_payload(decode=True)).decode(),
            )
            for part in multipart_parts(request)
            if part.get_filename() is None
        )
    )


def file_parts(request: Request) -> tuple[FilePart, ...]:
    """Every file part the upstream received, in the order it arrived on the wire."""
    return tuple(
        FilePart(
            name=cast(str, part.get_param("name", header="content-disposition")),
            filename=filename,
            content_type=part.get_content_type(),
            content=cast(bytes, part.get_payload(decode=True)),
        )
        for part in multipart_parts(request)
        if (filename := part.get_filename()) is not None
    )


def json_body(request: Request) -> dict[str, JsonValue]:
    return JSON_OBJECT.validate_json(request.body)


def json_reply(body: Mapping[str, JsonValue], status: int = 200, headers: Mapping[str, str] | None = None) -> Reply:
    return Reply(status=status, body=json.dumps(body).encode(), headers=MappingProxyType(dict(headers or {})))


def openai_error_reply(status: int, message: str, error_type: str, code: str | None = None) -> Reply:
    """An OpenAI-shaped error body, the shape every OpenAI-compatible media upstream answers a failure with."""
    return json_reply({"error": {"message": message, "type": error_type, "param": None, "code": code}}, status)


def image_reply(*images: Mapping[str, JsonValue], **extra: JsonValue) -> Reply:
    return json_reply({"created": 1700000000, "data": [dict(image) for image in images], **extra})


def targets(wire: Wire) -> list[tuple[str, str]]:
    return [(request.method, request.target) for request in wire.drain()]


def error_of(response: httpx.Response) -> dict[str, JsonValue]:
    return object_value(JSON_OBJECT.validate_json(response.content)["error"])


def error_message(response: httpx.Response) -> str:
    return string_value(error_of(response)["message"])


def response_cost(response: httpx.Response) -> float:
    return float(response.headers["x-litellm-response-cost"])


def key_hash(key: str) -> str:
    return sha256(key.encode()).hexdigest()


def spend_rows(key: str) -> list[dict[str, JsonValue]]:
    return read_rows(
        f'SELECT {SPEND_COLUMNS} FROM "LiteLLM_SpendLogs" WHERE api_key = %s ORDER BY "startTime"',  # noqa: S608  # fixed column list
        (key_hash(key),),
    )


def logged_rows(key: str, count: int = 1) -> list[dict[str, JsonValue]]:
    """Polls until the key has exactly ``count`` spend rows; the rows land asynchronously after the response."""
    return eventually(lambda: spend_rows(key), lambda rows: len(rows) == count, seconds=70)


def key_spend(key: str) -> float:
    rows: Final = read_rows('SELECT spend FROM "LiteLLM_VerificationToken" WHERE token = %s', (key_hash(key),))
    assert len(rows) == 1, rows
    return float(str(rows[0]["spend"]))


def settled_key_spend(key: str, expected: float) -> float:
    return eventually(lambda: key_spend(key), lambda spend: abs(spend - expected) < 1e-9, seconds=70)


def post_files(
    gateway: Gateway,
    path: str,
    fields: Mapping[str, str],
    files: Sequence[tuple[str, tuple[str, bytes, str]]],
    *,
    key: str | None = None,
) -> httpx.Response:
    """Multipart POST that can repeat a file field name, which ``Gateway.request_multipart`` cannot."""
    return gateway.client.post(
        path,
        data=dict(fields),
        files=list(files),
        headers={"Authorization": f"Bearer {gateway.key if key is None else key}"},
    )
