from __future__ import annotations

import asyncio
import json
import time
from typing import TYPE_CHECKING, Any, Final

from litellm.llms.vertex_ai.audio_transcription.transformation import (
    DEFAULT_SPEECH_TO_TEXT_LOCATION,
    vertex_model_to_speech_v2_name,
)
from litellm.llms.vertex_ai.common_utils import VertexAIError, validate_vertex_location
from litellm.llms.vertex_ai.vertex_llm_base import VertexBase

if TYPE_CHECKING:
    from fastapi import WebSocket

_INSTALL_HINT: Final = (
    "google-cloud-speech is not installed. "
    "Install with: pip install 'litellm[stt-vertex-chirp-grpc]'"
)

_DEFAULT_SAMPLE_RATE: Final = 16000
_DEFAULT_ENCODING: Final = "LINEAR16"
_AUDIO_QUEUE_MAX: Final = 200


def _import_speech_v2() -> Any:
    try:
        from google.cloud import speech_v2
        return speech_v2
    except ImportError as e:
        raise VertexAIError(status_code=500, message=_INSTALL_HINT) from e


class ChirpSttWebSocketSession(VertexBase):

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

        model: Final = config_msg.get("model", "vertex_ai/latest_long")
        language: Final = config_msg.get("language", "es-ES")
        sample_rate: Final = int(config_msg.get("sample_rate", _DEFAULT_SAMPLE_RATE))
        encoding_name: Final = config_msg.get("encoding", _DEFAULT_ENCODING).upper()

        # Credentials are always resolved server-side via ADC or env vars.
        # vertex_credentials is intentionally not accepted from the client message
        # to prevent SSRF attacks via external-account credential injection.
        server_params: Final = {
            "vertex_project": config_msg.get("vertex_project"),
            "vertex_location": config_msg.get("vertex_location"),
        }

        credentials, project_id = self.load_auth(
            credentials=self.safe_get_vertex_ai_credentials(server_params),
            project_id=self.safe_get_vertex_ai_project(server_params),
        )

        raw_location: Final = (
            self.safe_get_vertex_ai_location(server_params) or DEFAULT_SPEECH_TO_TEXT_LOCATION
        )
        try:
            validate_vertex_location(raw_location)
        except ValueError as e:
            raise VertexAIError(status_code=400, message=str(e)) from e

        speech_v2: Final = _import_speech_v2()
        cs: Final = speech_v2.types.cloud_speech
        client: Final = speech_v2.SpeechAsyncClient(credentials=credentials)

        model_name: Final = vertex_model_to_speech_v2_name(model)
        recognizer: Final = f"projects/{project_id}/locations/global/recognizers/_"

        encoding_enum: Final = getattr(
            cs.ExplicitDecodingConfig.AudioEncoding,
            encoding_name,
            cs.ExplicitDecodingConfig.AudioEncoding.LINEAR16,
        )
        streaming_config: Final = cs.StreamingRecognitionConfig(
            config=cs.RecognitionConfig(
                explicit_decoding_config=cs.ExplicitDecodingConfig(
                    encoding=encoding_enum,
                    sample_rate_hertz=sample_rate,
                    audio_channel_count=1,
                ),
                model=model_name,
                language_codes=[language],
            ),
            streaming_features=cs.StreamingRecognitionFeatures(interim_results=True),
        )

        audio_queue: Final[asyncio.Queue[bytes | None]] = asyncio.Queue(maxsize=_AUDIO_QUEUE_MAX)

        async def request_generator():
            yield cs.StreamingRecognizeRequest(
                recognizer=recognizer, streaming_config=streaming_config
            )
            while True:
                chunk = await audio_queue.get()
                if chunk is None:
                    break
                yield cs.StreamingRecognizeRequest(audio=chunk)

        response_stream: Final = await client.streaming_recognize(
            requests=request_generator(), timeout=3600
        )

        total_audio_bytes = 0  # mutable-ok: accumulated across receive loop iterations
        start_time: Final = time.perf_counter()

        async def receive_from_client() -> None:
            nonlocal total_audio_bytes
            while True:
                try:
                    message = await websocket.receive()
                except Exception:  # noqa: BLE001  # client disconnected unexpectedly
                    await audio_queue.put(None)
                    return

                if message.get("type") == "websocket.disconnect":
                    await audio_queue.put(None)
                    return

                raw_bytes = message.get("bytes")
                raw_text = message.get("text")

                if raw_bytes:
                    total_audio_bytes += len(raw_bytes)
                    await audio_queue.put(raw_bytes)
                elif raw_text:
                    try:
                        msg = json.loads(raw_text)
                        if msg.get("type") == "end":
                            await audio_queue.put(None)
                            return
                    except (json.JSONDecodeError, AttributeError):
                        pass

        async def forward_to_client() -> None:
            async for response in response_stream:
                for result in response.results:
                    if not result.alternatives:
                        continue
                    alt: Final = result.alternatives[0]
                    payload: dict[str, Any] = {
                        "type": "final" if result.is_final else "interim",
                        "transcript": alt.transcript,
                    }
                    if result.is_final:
                        if alt.confidence:
                            payload["confidence"] = round(alt.confidence, 4)
                        if result.language_code:
                            payload["language"] = result.language_code
                        payload["latency_s"] = round(time.perf_counter() - start_time, 3)
                    else:
                        payload["stability"] = round(getattr(result, "stability", 0.0), 4)
                    await websocket.send_text(json.dumps(payload))

        receive_task: Final = asyncio.create_task(receive_from_client())
        forward_task: Final = asyncio.create_task(forward_to_client())

        # Wait for whichever direction finishes first, then drain the other.
        # receive_task ending signals the input stream is closed; we still wait
        # for forward_task so trailing final results are not dropped.
        # forward_task ending means the gRPC stream closed; we then stop reading.
        done, pending = await asyncio.wait(
            {receive_task, forward_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if receive_task in done and forward_task in pending:
            # Input done: give the gRPC response stream a short window to drain
            # any final results that were already in-flight, then cancel.
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
            # gRPC stream ended first: stop receiving.
            receive_task.cancel()
            try:
                await receive_task
            except asyncio.CancelledError:
                pass

        for task in done:
            exc = task.exception()
            if exc is not None:
                raise exc

        duration_s: Final = total_audio_bytes / (sample_rate * 2)
        await websocket.send_text(
            json.dumps({"type": "done", "audio_duration_s": round(duration_s, 3)})
        )


async def _send_error(websocket: WebSocket, message: str) -> None:
    try:
        await websocket.send_text(json.dumps({"type": "error", "message": message}))
    except Exception:  # noqa: BLE001,S110  # WebSocket may already be closed
        pass
