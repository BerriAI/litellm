from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Coroutine
from typing import TYPE_CHECKING, Any, Final

from litellm.litellm_core_utils.audio_utils.utils import (
    get_audio_file_name,
    process_audio_file,
)
from litellm.llms.vertex_ai.audio_transcription.transformation import (
    DEFAULT_SPEECH_TO_TEXT_LOCATION,
    vertex_model_to_speech_v2_name,
)
from litellm.llms.vertex_ai.common_utils import VertexAIError, validate_vertex_location
from litellm.llms.vertex_ai.vertex_llm_base import VertexBase
from litellm.types.llms.vertex_ai_speech_to_text import ChirpGrpcRequestConfig
from litellm.types.utils import FileTypes, TranscriptionResponse
from litellm.utils import convert_to_model_response_object

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj

_INSTALL_HINT: Final = (
    "google-cloud-speech is not installed. "
    "Install with: pip install 'litellm[stt-vertex-chirp-grpc]'"
)

_DEFAULT_CHUNK_BYTES: Final = 25_600


def _import_speech_v2() -> Any:
    try:
        from google.cloud import speech_v2

        return speech_v2
    except ImportError as e:
        raise VertexAIError(status_code=500, message=_INSTALL_HINT) from e


def _build_grpc_request_config(
    model: str,
    project_id: str,
    location: str,
    language: str | None,
) -> ChirpGrpcRequestConfig:
    model_name: Final = vertex_model_to_speech_v2_name(model)
    # Chirp 3 HD requires the global recognizer path regardless of the
    # regional endpoint the user configured for other Vertex AI calls.
    recognizer: Final = f"projects/{project_id}/locations/global/recognizers/_"
    language_codes: Final = (language,) if language else ("en-US",)
    return ChirpGrpcRequestConfig(
        recognizer=recognizer,
        model_name=model_name,
        language_codes=language_codes,
        enable_automatic_punctuation=True,
        chunk_bytes=_DEFAULT_CHUNK_BYTES,
    )


def _iter_audio_chunks(audio_bytes: bytes, chunk_bytes: int) -> AsyncIterator[bytes]:
    async def _gen() -> AsyncIterator[bytes]:
        for offset in range(0, len(audio_bytes), chunk_bytes):
            chunk: Final = audio_bytes[offset : offset + chunk_bytes]
            if chunk:
                yield chunk

    return _gen()


async def _build_request_stream(
    speech_v2: Any,
    config: ChirpGrpcRequestConfig,
    audio_bytes: bytes,
) -> AsyncIterator[Any]:
    cs: Final = speech_v2.types.cloud_speech
    recognition_config: Final = cs.RecognitionConfig(
        auto_decoding_config=cs.AutoDetectDecodingConfig(),
        model=config.model_name,
        language_codes=list(config.language_codes),
        features=cs.RecognitionFeatures(
            enable_automatic_punctuation=config.enable_automatic_punctuation,
        ),
    )
    streaming_config: Final = cs.StreamingRecognitionConfig(config=recognition_config)

    async def _gen() -> AsyncIterator[Any]:
        yield cs.StreamingRecognizeRequest(
            recognizer=config.recognizer,
            streaming_config=streaming_config,
        )
        async for chunk in _iter_audio_chunks(audio_bytes, config.chunk_bytes):
            yield cs.StreamingRecognizeRequest(audio=chunk)

    return _gen()


async def _collect_transcripts(response_stream: Any) -> tuple[str, str | None]:
    parts: list[str] = []  # mutable-ok: local accumulator, not externally visible
    detected_language: str | None = None
    async for response in response_stream:
        for result in response.results:
            if not result.is_final:
                continue
            if result.alternatives:
                parts.append(result.alternatives[0].transcript)
            if detected_language is None and result.language_code:
                detected_language = result.language_code
    return " ".join(parts).strip(), detected_language


class VertexAIChirpGrpcTranscription(VertexBase):

    def audio_transcriptions(
        self,
        model: str,
        audio_file: FileTypes,
        optional_params: dict,
        litellm_params: dict,
        model_response: TranscriptionResponse,
        timeout: float,
        logging_obj: LiteLLMLoggingObj,
        api_key: str | None,
        api_base: str | None,
        atranscription: bool = False,
    ) -> TranscriptionResponse | Coroutine[Any, Any, TranscriptionResponse]:
        if atranscription:
            return self._run_async(
                model=model,
                audio_file=audio_file,
                optional_params=optional_params,
                litellm_params=litellm_params,
                model_response=model_response,
                timeout=timeout,
                logging_obj=logging_obj,
                api_key=api_key,
                api_base=api_base,
            )
        return asyncio.run(
            self._run_async(
                model=model,
                audio_file=audio_file,
                optional_params=optional_params,
                litellm_params=litellm_params,
                model_response=model_response,
                timeout=timeout,
                logging_obj=logging_obj,
                api_key=api_key,
                api_base=api_base,
            )
        )

    async def _run_async(
        self,
        model: str,
        audio_file: FileTypes,
        optional_params: dict,
        litellm_params: dict,
        model_response: TranscriptionResponse,
        timeout: float,
        logging_obj: LiteLLMLoggingObj,
        api_key: str | None,
        api_base: str | None,
    ) -> TranscriptionResponse:
        location: Final = self._resolve_location(litellm_params)
        credentials, project_id = self.load_auth(
            credentials=self.safe_get_vertex_ai_credentials(litellm_params),
            project_id=self.safe_get_vertex_ai_project(litellm_params),
        )

        speech_v2: Final = _import_speech_v2()
        config: Final = _build_grpc_request_config(
            model=model,
            project_id=project_id,
            location=location,
            language=optional_params.get("language"),
        )

        processed: Final = process_audio_file(audio_file)

        logging_obj.pre_call(
            input=None,
            api_key=api_key,
            additional_args={
                "recognizer": config.recognizer,
                "model_name": config.model_name,
                "language_codes": config.language_codes,
            },
        )

        client: Final = speech_v2.SpeechAsyncClient(credentials=credentials)
        request_stream: Final = await _build_request_stream(
            speech_v2=speech_v2,
            config=config,
            audio_bytes=processed.file_content,
        )

        try:
            response_stream: Final = await client.streaming_recognize(
                requests=request_stream,
                timeout=timeout if timeout else None,
            )
            transcript, detected_language = await _collect_transcripts(response_stream)
        except VertexAIError:
            raise
        except Exception as e:
            raise VertexAIError(status_code=500, message=str(e)) from e

        response: Final = TranscriptionResponse(text=transcript)
        response["task"] = "transcribe"
        if detected_language is not None:
            response["language"] = detected_language

        stringified: Final = dict(response)

        logging_obj.post_call(
            input=get_audio_file_name(audio_file),
            api_key=api_key,
            additional_args={"recognizer": config.recognizer},
            original_response=stringified,
        )

        hidden_params: Final = {
            "model": model,
            "custom_llm_provider": "vertex_ai",
        }

        return convert_to_model_response_object(  # pyright: ignore[reportReturnType]  # stubs lack audio_transcription overload
            response_object=stringified,
            model_response_object=model_response,
            hidden_params=hidden_params,
            response_type="audio_transcription",
        )

    @staticmethod
    def _resolve_location(litellm_params: dict) -> str:
        raw: Final = VertexBase.safe_get_vertex_ai_location(litellm_params) or DEFAULT_SPEECH_TO_TEXT_LOCATION
        try:
            return validate_vertex_location(raw)
        except ValueError as e:
            raise VertexAIError(status_code=400, message=str(e)) from e
