"""Shared helpers for the `test_dec1_media2_*` contracts: audio, speech, OCR, video and image edit routes on the big
providers, each driven through the proxy against a scripted upstream (`wire_server`).

Most byte, multipart and spend helpers are re-exported from `dec1_media_support` so both waves read the upstream
request the same way. What is new here: a form reader that accepts either multipart or urlencoded bodies (the
OpenAI video create call sends whichever the HTTP client picks when no file is attached), canned provider replies
for Deepgram, ElevenLabs, Google long running operations and Gemini `generateContent` images, and a decoder for the
proxy's wrapped video ids."""

from __future__ import annotations

import base64
import json
from collections.abc import Mapping, Sequence
from typing import Final
from urllib.parse import parse_qs, unquote, urlsplit

import httpx
from integration._support.client import JSON_OBJECT, Gateway, eventually, string_value
from integration._support.database import read_rows
from integration._support.wire import Reply, Request
from integration.providers.dec1_media_support import (
    AZURE_KEY,
    B64_IMAGE,
    MP4_BYTES,
    OPENAI_KEY,
    PDF_BYTES,
    SPEECH_BYTES,
    FilePart,
    error_message,
    error_of,
    file_parts,
    json_body,
    json_reply,
    key_hash,
    key_spend,
    logged_rows,
    multipart_parts,
    openai_error_reply,
    png,
    response_cost,
    settled_key_spend,
    silent_wav,
    spend_rows,
    targets,
    text_fields,
)
from pydantic import JsonValue

__all__ = (
    "AZURE_KEY",
    "B64_IMAGE",
    "DEEPGRAM_KEY",
    "ELEVENLABS_KEY",
    "GEMINI_KEY",
    "GROQ_KEY",
    "MISTRAL_KEY",
    "MP4_BYTES",
    "OPENAI_KEY",
    "PDF_BYTES",
    "SPEECH_BYTES",
    "FilePart",
    "body_json",
    "created_video_id",
    "data_url",
    "decoded_video",
    "deepgram_reply",
    "elevenlabs_stt_reply",
    "error_message",
    "error_of",
    "file_parts",
    "form_fields",
    "gemini_image_reply",
    "image_reply_body",
    "json_body",
    "json_reply",
    "key_hash",
    "key_spend",
    "logged_rows",
    "mistral_ocr_reply",
    "multipart_parts",
    "openai_error_reply",
    "path_of",
    "png",
    "post_upload",
    "query_of",
    "response_cost",
    "row_by_call_id",
    "settled_key_spend",
    "silent_wav",
    "spend_rows",
    "targets",
    "text_fields",
    "transcribe",
)

GROQ_KEY: Final = "synthetic-groq-key"
DEEPGRAM_KEY: Final = "synthetic-deepgram-key"
ELEVENLABS_KEY: Final = "synthetic-elevenlabs-key"
MISTRAL_KEY: Final = "synthetic-mistral-key"
GEMINI_KEY: Final = "synthetic-gemini-key"
TRANSCRIPTIONS: Final = "/v1/audio/transcriptions"


def path_of(request: Request) -> str:
    return unquote(urlsplit(request.target).path)


def query_of(request: Request) -> dict[str, list[str]]:
    return parse_qs(urlsplit(request.target).query, keep_blank_values=True)


def form_fields(request: Request) -> dict[str, str]:
    """Every text field of a form body, whether the client encoded it as multipart or as urlencoded."""
    content_type: Final = request.headers.get("content-type", "")
    if content_type.startswith("multipart/form-data"):
        return dict(text_fields(request))
    return {name: values[-1] for name, values in parse_qs(request.body.decode()).items()}


def data_url(media_type: str, content: bytes) -> str:
    return f"data:{media_type};base64,{base64.b64encode(content).decode()}"


def transcribe(
    gateway: Gateway,
    fields: Mapping[str, str] | Sequence[tuple[str, str]],
    audio: bytes,
    *,
    filename: str = "clip.wav",
    content_type: str = "audio/wav",
    key: str | None = None,
) -> httpx.Response:
    """Multipart POST to `/v1/audio/transcriptions`; a sequence of pairs lets a field repeat (`timestamp_granularities[]`)."""
    return gateway.client.post(
        TRANSCRIPTIONS,
        data=dict(fields) if isinstance(fields, Mapping) else list(fields),
        files={"file": (filename, audio, content_type)},
        headers={"Authorization": f"Bearer {gateway.key if key is None else key}"},
    )


def post_upload(
    gateway: Gateway,
    path: str,
    fields: Mapping[str, str],
    files: Sequence[tuple[str, tuple[str, bytes, str]]],
    *,
    key: str | None = None,
) -> httpx.Response:
    return gateway.client.post(
        path,
        data=dict(fields),
        files=list(files),
        headers={"Authorization": f"Bearer {gateway.key if key is None else key}"},
    )


def deepgram_reply(
    transcript: str = "hello from deepgram",
    duration: float = 42.5,
    words: Sequence[Mapping[str, JsonValue]] = (
        {"word": "hello", "start": 0.0, "end": 0.4, "confidence": 0.99},
        {"word": "from", "start": 0.5, "end": 0.7, "confidence": 0.98},
        {"word": "deepgram", "start": 0.8, "end": 1.3, "confidence": 0.97},
    ),
    detected_language: str | None = None,
) -> Reply:
    """Deepgram `/v1/listen` answer: one channel, one alternative, `metadata.duration` in seconds."""
    channel: Final[dict[str, JsonValue]] = {
        "alternatives": [{"transcript": transcript, "confidence": 0.98, "words": [dict(word) for word in words]}]
    }
    if detected_language is not None:
        channel["detected_language"] = detected_language
    return json_reply(
        {
            "metadata": {"request_id": "dg-req-1", "duration": duration, "channels": 1},
            "results": {"channels": [channel]},
        }
    )


def elevenlabs_stt_reply(text: str = "hello from scribe", language_code: str = "eng") -> Reply:
    """ElevenLabs `/v1/speech-to-text` answer with words and one spacing entry that must be filtered out."""
    return json_reply(
        {
            "language_code": language_code,
            "language_probability": 0.99,
            "text": text,
            "words": [
                {"text": "hello", "start": 0.0, "end": 0.4, "type": "word"},
                {"text": " ", "start": 0.4, "end": 0.5, "type": "spacing"},
                {"text": "scribe", "start": 0.5, "end": 1.1, "type": "word"},
            ],
        }
    )


def mistral_ocr_reply(
    pages: int = 2,
    *,
    model: str = "mistral-ocr-latest",
    usage: Mapping[str, JsonValue] | None = None,
    **extra: JsonValue,
) -> Reply:
    body: Final[dict[str, JsonValue]] = {
        "pages": [
            {
                "index": index,
                "markdown": f"# page {index}",
                "images": [],
                "dimensions": {"dpi": 200, "height": 1100, "width": 850},
            }
            for index in range(pages)
        ],
        "model": model,
        "usage_info": dict(usage) if usage is not None else {"pages_processed": pages, "doc_size_bytes": 2048},
        "document_annotation": None,
        **extra,
    }
    return Reply(body=json.dumps(body).encode())


def gemini_image_reply(
    *images: str, prompt_tokens: int = 20, image_tokens: int = 1290, text: str | None = None
) -> Reply:
    """Gemini `generateContent` answer whose parts carry ``images`` as inline base64 PNG data."""
    parts: Final[list[JsonValue]] = [{"inlineData": {"mimeType": "image/png", "data": image}} for image in images]
    if text is not None:
        parts.insert(0, {"text": text})
    return json_reply(
        {
            "candidates": [{"content": {"role": "model", "parts": parts}, "finishReason": "STOP"}],
            "usageMetadata": {
                "promptTokenCount": prompt_tokens,
                "candidatesTokenCount": image_tokens,
                "totalTokenCount": prompt_tokens + image_tokens,
                "candidatesTokensDetails": [{"modality": "IMAGE", "tokenCount": image_tokens}],
            },
            "modelVersion": "gemini-2.5-flash-image",
        }
    )


def decoded_video(video_id: str) -> Mapping[str, str | None]:
    """The provider, model id and provider-side id the proxy packed into a `video_...` id."""
    from litellm.types.videos.utils import decode_video_id_with_provider

    return {
        name: value if isinstance(value, str) else None
        for name, value in decode_video_id_with_provider(video_id).items()
    }


def row_by_call_id(call_id: str) -> dict[str, JsonValue]:
    """The one spend row the proxy writes for ``call_id`` (its `x-litellm-call-id`), polled until it lands."""
    rows: Final = eventually(
        lambda: read_rows(
            "SELECT request_id, call_type, spend, model, model_group, custom_llm_provider, status, api_base "
            'FROM "LiteLLM_SpendLogs" WHERE request_id = %s',
            (call_id,),
        ),
        lambda values: len(values) == 1,
        seconds=70,
    )
    return rows[0]


def body_json(response: httpx.Response) -> dict[str, JsonValue]:
    return JSON_OBJECT.validate_json(response.content)


def created_video_id(response: httpx.Response) -> str:
    assert response.status_code == 200, response.text
    return string_value(body_json(response)["id"])


def image_reply_body(*images: str) -> bytes:
    """An OpenAI images response body carrying ``images`` (default one ``B64_IMAGE``) as `b64_json` entries."""
    return json.dumps(
        {"created": 1700000000, "data": [{"b64_json": image} for image in (images or (B64_IMAGE,))]}
    ).encode()
