from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Coroutine
from typing import TYPE_CHECKING, Any, Final

import httpx

from litellm.llms.vertex_ai.common_utils import VertexAIError
from litellm.llms.vertex_ai.text_to_speech.transformation import (
    extract_language_code_from_voice,
)
from litellm.llms.vertex_ai.vertex_llm_base import VertexBase

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.types.llms.openai import HttpxBinaryResponseContent

_INSTALL_HINT: Final = (
    "google-cloud-texttospeech is not installed. "
    "Install with: pip install 'litellm[tts-vertex-chirp-grpc]'"
)


def _import_texttospeech() -> Any:
    try:
        from google.cloud import texttospeech

        return texttospeech
    except ImportError as e:
        raise VertexAIError(status_code=500, message=_INSTALL_HINT) from e


async def _build_request_stream(
    tts_module: Any,
    voice_name: str,
    language_code: str,
    text: str,
    audio_encoding: str,
    speaking_rate: float | None,
) -> AsyncIterator[Any]:
    # google-cloud-texttospeech exposes types directly on the module,
    # not via a .types.cloud_tts sub-module like speech_v2 does.
    streaming_audio_config: Final = tts_module.StreamingAudioConfig(
        audio_encoding=getattr(tts_module.AudioEncoding, audio_encoding, tts_module.AudioEncoding.LINEAR16),
        **({"speaking_rate": speaking_rate} if speaking_rate is not None else {}),
    )
    config: Final = tts_module.StreamingSynthesizeConfig(
        voice=tts_module.VoiceSelectionParams(
            name=voice_name,
            language_code=language_code,
        ),
        streaming_audio_config=streaming_audio_config,
    )

    async def _gen() -> AsyncIterator[Any]:
        yield tts_module.StreamingSynthesizeRequest(streaming_config=config)
        yield tts_module.StreamingSynthesizeRequest(input=tts_module.StreamingSynthesisInput(text=text))

    return _gen()


async def _collect_audio(response_stream: Any) -> bytes:
    chunks: list[bytes] = []  # mutable-ok: local accumulator
    async for response in response_stream:
        if response.audio_content:
            chunks.append(bytes(response.audio_content))
    return b"".join(chunks)


class VertexAIChirpGrpcTextToSpeech(VertexBase):

    def speech(
        self,
        input: str,
        voice_name: str,
        optional_params: dict,
        litellm_params: dict,
        logging_obj: LiteLLMLoggingObj,
        timeout: float | httpx.Timeout | None,
        aspeech: bool = False,
    ) -> HttpxBinaryResponseContent | Coroutine[Any, Any, HttpxBinaryResponseContent]:
        if aspeech:
            return self._run_async(
                input=input,
                voice_name=voice_name,
                optional_params=optional_params,
                litellm_params=litellm_params,
                logging_obj=logging_obj,
                timeout=timeout,
            )
        return asyncio.run(
            self._run_async(
                input=input,
                voice_name=voice_name,
                optional_params=optional_params,
                litellm_params=litellm_params,
                logging_obj=logging_obj,
                timeout=timeout,
            )
        )

    async def _run_async(
        self,
        input: str,
        voice_name: str,
        optional_params: dict,
        litellm_params: dict,
        logging_obj: LiteLLMLoggingObj,
        timeout: float | httpx.Timeout | None,
    ) -> HttpxBinaryResponseContent:
        from litellm.types.llms.openai import HttpxBinaryResponseContent

        credentials, _ = self.load_auth(
            credentials=self.safe_get_vertex_ai_credentials(litellm_params),
            project_id=self.safe_get_vertex_ai_project(litellm_params),
        )

        tts_module: Final = _import_texttospeech()
        language_code: Final = extract_language_code_from_voice(voice_name)
        # Streaming synthesis only supports OGG_OPUS, MULAW, and ALAW.
        # LINEAR16 and MP3 are rejected with "Unsupported audio encoding".
        # Fall back to OGG_OPUS when the caller requests an incompatible encoding.
        _STREAMING_ENCODINGS: Final = frozenset({"OGG_OPUS", "MULAW", "ALAW"})
        _requested: Final = optional_params.get("audioEncoding", "OGG_OPUS")
        audio_encoding: Final = _requested if _requested in _STREAMING_ENCODINGS else "OGG_OPUS"
        speaking_rate: Final = _parse_speaking_rate(optional_params.get("speakingRate"))

        logging_obj.pre_call(
            input=input,
            api_key=None,
            additional_args={
                "voice_name": voice_name,
                "language_code": language_code,
                "audio_encoding": audio_encoding,
            },
        )

        client: Final = tts_module.TextToSpeechAsyncClient(credentials=credentials)
        request_stream: Final = await _build_request_stream(
            tts_module=tts_module,
            voice_name=voice_name,
            language_code=language_code,
            text=input,
            audio_encoding=audio_encoding,
            speaking_rate=speaking_rate,
        )

        timeout_seconds: Final = _resolve_timeout(timeout)
        try:
            response_stream: Final = await client.streaming_synthesize(
                requests=request_stream,
                timeout=timeout_seconds,
            )
            audio_bytes: Final = await _collect_audio(response_stream)
        except VertexAIError:
            raise
        except Exception as e:
            raise VertexAIError(status_code=500, message=str(e)) from e

        logging_obj.post_call(
            input=input,
            api_key=None,
            additional_args={"voice_name": voice_name},
            original_response=f"<audio bytes: {len(audio_bytes)}>",
        )

        return HttpxBinaryResponseContent(
            httpx.Response(status_code=200, content=audio_bytes)
        )


def _parse_speaking_rate(value: Any) -> float | None:
    if value is None:
        return None
    try:
        rate: Final = float(value)
        return rate if rate != 1.0 else None
    except (TypeError, ValueError):
        return None


def _resolve_timeout(timeout: float | httpx.Timeout | None) -> float | None:
    if timeout is None:
        return None
    if isinstance(timeout, httpx.Timeout):
        return timeout.read
    return float(timeout)
