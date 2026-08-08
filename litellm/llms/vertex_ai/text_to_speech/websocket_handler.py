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

_SUPPORTED_ENCODINGS: Final = frozenset({"OGG_OPUS", "MULAW", "ALAW"})
_DEFAULT_ENCODING: Final = "OGG_OPUS"
_DEFAULT_VOICE: Final = "en-US-Chirp3-HD-Charon"
_TEXT_QUEUE_MAX: Final = 200


def _import_texttospeech() -> Any:
    try:
        from google.cloud import texttospeech
        return texttospeech
    except ImportError as e:
        raise VertexAIError(status_code=500, message=_INSTALL_HINT) from e


class ChirpTtsWebSocketSession(VertexBase):

    async def run(
        self,
        websocket: WebSocket,
        user_api_key_dict: Any,
    ) -> None:
        await websocket.accept()
        try:
            await self._session(websocket, user_api_key_dict)
        except Exception as e:  # noqa: BLE001  # any session error -> error msg to client
            await _send_error(websocket, str(e))
        finally:
            try:
                await websocket.close()
            except Exception:  # noqa: BLE001,S110  # already closed, ignore
                pass

    async def _session(self, websocket: WebSocket, user_api_key_dict: Any) -> None:
        raw_config = await websocket.receive_text()
        config_msg = json.loads(raw_config)

        voice_name: Final = config_msg.get("voice", _DEFAULT_VOICE)
        encoding_raw: Final = config_msg.get("encoding", _DEFAULT_ENCODING).upper()
        encoding_name: Final = encoding_raw if encoding_raw in _SUPPORTED_ENCODINGS else _DEFAULT_ENCODING

        # Credentials are always resolved server-side via ADC or env vars.
        # vertex_credentials is intentionally not accepted from the client message
        # to prevent SSRF attacks via external-account credential injection.
        server_params: Final = {
            "vertex_project": config_msg.get("vertex_project"),
            "vertex_location": config_msg.get("vertex_location"),
        }

        credentials, _ = self.load_auth(
            credentials=self.safe_get_vertex_ai_credentials(server_params),
            project_id=self.safe_get_vertex_ai_project(server_params),
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

        text_queue: Final[asyncio.Queue[str | None]] = asyncio.Queue(maxsize=_TEXT_QUEUE_MAX)

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

        total_chars = 0        # mutable-ok: accumulated across receive loop iterations
        total_audio_chunks = 0  # mutable-ok: accumulated across forward loop iterations
        total_audio_bytes = 0   # mutable-ok: accumulated across forward loop iterations

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

        # Wait for whichever direction finishes first, then drain the other.
        # receive_task ending signals input is closed; wait for forward_task to
        # drain remaining audio before returning so no chunks are dropped.
        # forward_task ending means the gRPC stream closed; cancel receive_task.
        done, pending = await asyncio.wait(
            {receive_task, forward_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if receive_task in done and forward_task in pending:
            # Input done: give the gRPC response stream a short window to drain
            # any audio chunks already in-flight, then cancel.
            try:
                await asyncio.wait_for(asyncio.shield(forward_task), timeout=0.5)
            except (asyncio.TimeoutError, Exception):  # noqa: BLE001,S110  # drain window expired or error already sent
                pass
            finally:
                forward_task.cancel()
                try:
                    await forward_task
                except (asyncio.CancelledError, Exception):  # noqa: BLE001,S110  # expected on cancel
                    pass
        else:
            receive_task.cancel()
            try:
                await receive_task
            except asyncio.CancelledError:
                pass

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
