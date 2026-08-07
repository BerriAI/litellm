"""
Vertex AI Chirp 3 HD TTS — WebSocket-to-gRPC bridge.

Exposes a WebSocket endpoint so callers can stream text in real-time (e.g., from
a streaming LLM response) and receive audio chunks as they are synthesized,
without waiting for the full text to be available.

Wire protocol (client → server):
    1. First message (text/JSON):
       {"model": "vertex_ai/chirp-3-hd",
        "voice": "es-ES-Chirp3-HD-Charon",
        "encoding": "OGG_OPUS",
        "vertex_project": "...", "vertex_location": "us-central1"}
    2. Subsequent messages (text/JSON):
       {"type": "text", "content": "Siguiente frase parcial del LLM..."}
    3. Termination: send {"type": "end"} OR simply close the WebSocket

Wire protocol (server → client):
    binary: <audio chunk bytes>   (OGG_OPUS, MULAW, or ALAW — never LINEAR16/MP3)
    text/JSON: {"type": "done", "chunks": N, "total_bytes": M, "chars": K}
    text/JSON: {"type": "error", "message": "..."}

google-cloud-texttospeech is imported lazily so litellm core stays usable without it.
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any, Final

from litellm.llms.vertex_ai.common_utils import VertexAIError
from litellm.llms.vertex_ai.text_to_speech.transformation import (
    extract_language_code_from_voice,
)
from litellm.llms.vertex_ai.vertex_llm_base import VertexBase

if TYPE_CHECKING:
    from fastapi import WebSocket

_INSTALL_HINT: Final = (
    "google-cloud-texttospeech is not installed. "
    "Install with: pip install 'litellm[tts-vertex-chirp-grpc]'"
)

# streaming_synthesize only supports these encodings — LINEAR16 and MP3 are rejected.
_SUPPORTED_ENCODINGS: Final = frozenset({"OGG_OPUS", "MULAW", "ALAW"})
_DEFAULT_ENCODING: Final = "OGG_OPUS"
_DEFAULT_VOICE: Final = "en-US-Chirp3-HD-Charon"


def _import_texttospeech() -> Any:
    try:
        from google.cloud import texttospeech
        return texttospeech
    except ImportError as e:
        raise VertexAIError(status_code=500, message=_INSTALL_HINT) from e


class ChirpTtsWebSocketSession(VertexBase):
    """Bridges one WebSocket connection to a gRPC streaming_synthesize session."""

    async def run(
        self,
        websocket: WebSocket,
        user_api_key_dict: Any,
    ) -> None:
        await websocket.accept()
        try:
            await self._session(websocket, user_api_key_dict)
        except Exception as e:  # noqa: BLE001  # any session error → error msg to client
            await _send_error(websocket, str(e))
        finally:
            try:
                await websocket.close()
            except Exception:  # noqa: BLE001,S110  # already closed — ignore
                pass

    async def _session(self, websocket: WebSocket, user_api_key_dict: Any) -> None:
        # First message must be the config
        raw_config = await websocket.receive_text()
        config_msg = json.loads(raw_config)

        voice_name: Final = config_msg.get("voice", _DEFAULT_VOICE)
        encoding_raw: Final = config_msg.get("encoding", _DEFAULT_ENCODING).upper()
        encoding_name: Final = encoding_raw if encoding_raw in _SUPPORTED_ENCODINGS else _DEFAULT_ENCODING

        litellm_params: Final = {
            "vertex_project": config_msg.get("vertex_project"),
            "vertex_location": config_msg.get("vertex_location"),
            "vertex_credentials": config_msg.get("vertex_credentials"),
        }

        credentials, _ = self.load_auth(
            credentials=self.safe_get_vertex_ai_credentials(litellm_params),
            project_id=self.safe_get_vertex_ai_project(litellm_params),
        )

        tts: Final = _import_texttospeech()
        language_code: Final = extract_language_code_from_voice(voice_name)
        encoding_enum: Final = getattr(
            tts.AudioEncoding, encoding_name, tts.AudioEncoding.OGG_OPUS
        )

        streaming_config: Final = tts.StreamingSynthesizeConfig(
            voice=tts.VoiceSelectionParams(
                name=voice_name,
                language_code=language_code,
            ),
            streaming_audio_config=tts.StreamingAudioConfig(
                audio_encoding=encoding_enum,
            ),
        )

        # Queue bridges the WebSocket receive loop and the gRPC request generator.
        # None is the sentinel signalling end-of-stream.
        text_queue: Final[asyncio.Queue[str | None]] = asyncio.Queue()

        async def request_generator():
            yield tts.StreamingSynthesizeRequest(streaming_config=streaming_config)
            while True:
                chunk = await text_queue.get()
                if chunk is None:
                    break
                yield tts.StreamingSynthesizeRequest(
                    input=tts.StreamingSynthesisInput(text=chunk)
                )

        client: Final = tts.TextToSpeechAsyncClient(credentials=credentials)
        response_stream: Final = await client.streaming_synthesize(
            requests=request_generator(), timeout=3600
        )

        total_chars = 0
        total_audio_chunks = 0
        total_audio_bytes = 0

        async def receive_from_client() -> None:
            nonlocal total_chars
            while True:
                try:
                    message = await websocket.receive()
                except Exception:  # noqa: BLE001  # client disconnected unexpectedly
                    await text_queue.put(None)
                    return

                if message.get("type") == "websocket.disconnect":
                    await text_queue.put(None)
                    return

                raw_text = message.get("text")
                if not raw_text:
                    continue

                try:
                    msg = json.loads(raw_text)
                except (json.JSONDecodeError, AttributeError):
                    continue

                msg_type = msg.get("type")
                if msg_type == "end":
                    await text_queue.put(None)
                    return
                if msg_type == "text":
                    content: Final = msg.get("content", "")
                    if content:
                        total_chars += len(content)
                        await text_queue.put(content)

        async def forward_to_client() -> None:
            nonlocal total_audio_chunks, total_audio_bytes
            async for response in response_stream:
                if response.audio_content:
                    chunk: Final = bytes(response.audio_content)
                    total_audio_chunks += 1
                    total_audio_bytes += len(chunk)
                    await websocket.send_bytes(chunk)

        receive_task: Final = asyncio.create_task(receive_from_client())
        forward_task: Final = asyncio.create_task(forward_to_client())
        done, pending = await asyncio.wait(
            {receive_task, forward_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()
        for task in done:
            exc = task.exception()
            if exc is not None:
                raise exc

        await websocket.send_text(
            json.dumps({
                "type": "done",
                "chunks": total_audio_chunks,
                "total_bytes": total_audio_bytes,
                "chars": total_chars,
            })
        )


async def _send_error(websocket: WebSocket, message: str) -> None:
    try:
        await websocket.send_text(json.dumps({"type": "error", "message": message}))
    except Exception:  # noqa: BLE001,S110  # WebSocket may already be closed
        pass
