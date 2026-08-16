from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Final, Required
from urllib.parse import urlencode

from pydantic import TypeAdapter
from starlette.websockets import WebSocket
from typing_extensions import ReadOnly, TypedDict

from litellm.litellm_core_utils.url_utils import encode_url_path_segment
from litellm.secret_managers.main import get_secret_str

if TYPE_CHECKING:
    from websockets.asyncio.client import ClientConnection

_WS_BASE_URL: Final = "wss://api.elevenlabs.io"
_WS_PATH: Final = "/v1/text-to-speech/{voice_id}/stream-input"


class VoiceSettings(TypedDict, total=False):
    stability: ReadOnly[float]
    similarity_boost: ReadOnly[float]
    style: ReadOnly[float]
    use_speaker_boost: ReadOnly[bool]
    speed: ReadOnly[float]


class GenerationConfig(TypedDict, total=False):
    chunk_length_schedule: ReadOnly[list[float]]


class _TtsClientMessage(TypedDict, total=False):
    text: Required[ReadOnly[str]]
    flush: ReadOnly[bool]
    try_trigger_generation: ReadOnly[bool]
    voice_settings: ReadOnly[VoiceSettings]
    generator_config: ReadOnly[GenerationConfig]


class _TtsServerMessage(TypedDict, total=False):
    audio: ReadOnly[str]
    isFinal: ReadOnly[bool]  # camelCase matches ElevenLabs API field name


_CLIENT_MSG_ADAPTER: Final = TypeAdapter(_TtsClientMessage)
_SERVER_MSG_ADAPTER: Final = TypeAdapter(_TtsServerMessage)


def build_elevenlabs_ws_url(
    model: str,
    voice_id: str,
    output_format: str,
    api_base: str | None = None,
) -> str:
    raw_base: Final = (api_base or get_secret_str("ELEVENLABS_API_BASE") or _WS_BASE_URL).rstrip("/")
    ws_base: Final = raw_base.replace("https://", "wss://").replace("http://", "ws://")
    encoded_voice: Final = encode_url_path_segment(voice_id, field_name="voice_id")
    path: Final = _WS_PATH.format(voice_id=encoded_voice)
    query: Final = urlencode({"model_id": model, "output_format": output_format})
    return f"{ws_base}{path}?{query}"


async def _relay_client_to_upstream(
    client_ws: WebSocket,
    upstream: ClientConnection,
) -> tuple[int, ...]:
    chunk_lengths: list[int] = []  # mutable-ok: local accumulator, converted to immutable tuple on return
    async for raw in client_ws.iter_text():
        msg = _CLIENT_MSG_ADAPTER.validate_json(raw)  # rebind-ok: loop-body, rebound each iteration
        await upstream.send(json.dumps(dict(msg)))
        text = msg.get("text", "")  # rebind-ok: loop-body, rebound each iteration
        chunk_lengths.append(len(text))
        if not text:
            break
    return tuple(chunk_lengths)


async def _relay_upstream_to_client(
    upstream: ClientConnection,
    client_ws: WebSocket,
) -> None:
    async for raw in upstream:
        payload = raw if isinstance(raw, str) else raw.decode()  # rebind-ok: loop-body, rebound each iteration
        await client_ws.send_text(payload)
        msg = _SERVER_MSG_ADAPTER.validate_json(payload)  # rebind-ok: loop-body, rebound each iteration
        if msg.get("isFinal"):
            break


async def stream_input_tts(
    *,
    client_ws: WebSocket,
    model: str,
    voice_id: str,
    output_format: str = "mp3_44100_128",
    api_key: str | None = None,
    api_base: str | None = None,
    voice_settings: VoiceSettings | None = None,
    generation_config: GenerationConfig | None = None,
) -> int:
    """
    Relay a streaming-input TTS session between a client WebSocket and ElevenLabs.

    The proxy sends BOS automatically on connect using the provided voice/generation
    settings. The client then sends text chunks as {"text": "..."} and signals end of
    stream with {"text": ""}. Audio responses (JSON with base64 audio) are forwarded
    back to the client as-is.

    Returns the total number of text characters sent (used for per-character cost tracking).
    """
    import websockets

    key: Final = api_key or get_secret_str("ELEVENLABS_API_KEY")
    if key is None:
        raise ValueError("ElevenLabs API key is required. Set ELEVENLABS_API_KEY.")

    url: Final = build_elevenlabs_ws_url(model, voice_id, output_format, api_base)

    bos: dict[str, object] = {"text": " "}  # mutable-ok: fields added conditionally before first send
    if voice_settings is not None:
        bos["voice_settings"] = voice_settings
    if generation_config is not None:
        bos["generation_config"] = generation_config

    async with websockets.connect(url, additional_headers={"xi-api-key": key}) as upstream:
        await upstream.send(json.dumps(bos))

        task_c2u: Final = asyncio.create_task(_relay_client_to_upstream(client_ws, upstream))
        task_u2c: Final = asyncio.create_task(_relay_upstream_to_client(upstream, client_ws))

        # Wait for whichever relay finishes first, then cancel the other.
        # This prevents the client-to-upstream relay from hanging if ElevenLabs
        # sends isFinal before the client sends EOS, or if either side disconnects.
        await asyncio.wait({task_c2u, task_u2c}, return_when=asyncio.FIRST_COMPLETED)
        task_c2u.cancel()
        task_u2c.cancel()
        results: Final = await asyncio.gather(task_c2u, task_u2c, return_exceptions=True)

    c2u_result: Final = results[0]
    chunk_lengths: Final = c2u_result if isinstance(c2u_result, tuple) else ()
    return sum(chunk_lengths)
