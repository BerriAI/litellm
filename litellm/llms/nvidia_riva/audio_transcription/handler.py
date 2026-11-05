"""
NVIDIA Riva STT handler.

This module bridges litellm's transcription dispatch to NVIDIA Riva's gRPC
streaming ASR API. We do *not* go through ``base_llm_http_handler`` because
Riva is gRPC-only: HTTP-shaped abstractions (``httpx.Response``,
``api_base/v1/...`` URLs, multipart bodies) do not apply.

The handler is intentionally a thin orchestration layer:

1. Resample the inbound audio to 16 kHz mono LINEAR_PCM (Riva's required
   wire format).
2. Build ``RecognitionConfig`` / ``StreamingRecognitionConfig`` protobufs
   from the structured dict produced by
   :class:`NvidiaRivaAudioTranscriptionConfig`.
3. Open a gRPC channel honoring NVCF (TLS by default) vs self-hosted (any
   host:port, optional TLS) modes, and attach the ``authorization`` and NVCF
   ``function-id`` metadata to each call exactly once.
4. Stream the audio through Riva's ``StreamingRecognize`` RPC and aggregate
   ``is_final`` results into a single transcript, or, when ``riva_offline``
   is set, send the whole buffer through the unary ``Recognize`` RPC for NIMs
   that only ship offline engines.
5. Return a normalized ``TranscriptionResponse`` with ``duration`` exposed
   on ``_hidden_params`` so cost calculation works.

``grpcio`` and ``protobuf`` are imported lazily so ``litellm`` core remains
usable without the optional STT extras installed.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable, Sequence
from contextlib import closing
from typing import TYPE_CHECKING, Final

import httpx

import litellm
from litellm.constants import COMPLETION_HTTP_FALLBACK_SECONDS
from litellm.litellm_core_utils.audio_utils.utils import (
    get_audio_file_name,
    process_audio_file,
)
from litellm.litellm_core_utils.completion_timeout import CompletionTimeout
from litellm.llms.nvidia_riva.audio_transcription.audio_utils import (
    resample_to_riva_pcm,
)
from litellm.llms.nvidia_riva.audio_transcription.transformation import (
    RIVA_TARGET_NUM_CHANNELS,
    RIVA_TARGET_SAMPLE_RATE_HZ,
    NvidiaRivaAudioTranscriptionConfig,
    RivaFinalResult,
    RivaWordTiming,
)
from litellm.llms.nvidia_riva.common_utils import (
    NvidiaRivaException,
    grpc_error_to_litellm_exception,
)
from litellm.secret_managers.main import str_to_bool
from litellm.types.utils import FileTypes, TranscriptionResponse
from litellm.utils import convert_to_model_response_object

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import (
        Logging as LiteLLMLoggingObj,
    )

_RIVA_INSTALL_HINT = "NVIDIA Riva client is not installed. Install with `pip install 'litellm[stt-nvidia-riva]'`."


def _use_ssl(optional_params: dict) -> bool:
    """
    NVCF endpoints are TLS-only, so ``nvcf_function_id`` defaults ``use_ssl``
    to True. Self-hosted Riva defaults to plaintext but honors an explicit
    ``use_ssl``, since TLS-terminating ingresses in front of it are common.
    """
    use_ssl_override: Final = optional_params.get("use_ssl")
    if use_ssl_override is not None:
        return bool(use_ssl_override)
    return bool(optional_params.get("nvcf_function_id"))


def _parse_riva_offline(value: object, drop_params: bool) -> bool:
    match value:
        case None:
            return False
        case bool():
            return value
        case str() if (parsed := str_to_bool(value)) is not None:
            return parsed
        case _ if drop_params:
            return False
        case _:
            raise NvidiaRivaException(
                status_code=400,
                message=f"riva_offline must be true or false, got {value!r}. Set drop_params to ignore it.",
            )


def _deadline_seconds(timeout: float | str | httpx.Timeout, drop_params: bool) -> float:
    try:
        limit: Final = CompletionTimeout.as_limit(timeout)
    except ValueError:
        if drop_params:
            return COMPLETION_HTTP_FALLBACK_SECONDS
        raise NvidiaRivaException(
            status_code=400,
            message=f"timeout must be a number of seconds, got {timeout!r}. Set drop_params to ignore it.",
        ) from None
    if not isinstance(limit, httpx.Timeout):
        return limit
    return COMPLETION_HTTP_FALLBACK_SECONDS if limit.read is None else limit.read


class NvidiaRivaAudioTranscription:
    """Sync + async entry point for Riva ASR."""

    def audio_transcriptions(
        self,
        model: str,
        audio_file: FileTypes,
        optional_params: dict,
        litellm_params: dict,
        model_response: TranscriptionResponse,
        timeout: float | str | httpx.Timeout | None,
        logging_obj: LiteLLMLoggingObj,
        api_key: str | None,
        api_base: str | None,
        atranscription: bool = False,
        provider_config: NvidiaRivaAudioTranscriptionConfig | None = None,
    ):
        if provider_config is None:
            provider_config = NvidiaRivaAudioTranscriptionConfig()

        if atranscription:
            return self.async_audio_transcriptions(
                model=model,
                audio_file=audio_file,
                optional_params=optional_params,
                litellm_params=litellm_params,
                model_response=model_response,
                timeout=timeout,
                logging_obj=logging_obj,
                api_key=api_key,
                api_base=api_base,
                provider_config=provider_config,
            )

        return self._run_sync(
            model=model,
            audio_file=audio_file,
            optional_params=optional_params,
            litellm_params=litellm_params,
            model_response=model_response,
            timeout=timeout,
            logging_obj=logging_obj,
            api_key=api_key,
            api_base=api_base,
            provider_config=provider_config,
            atranscription=atranscription,
        )

    async def async_audio_transcriptions(
        self,
        model: str,
        audio_file: FileTypes,
        optional_params: dict,
        litellm_params: dict,
        model_response: TranscriptionResponse,
        timeout: float | str | httpx.Timeout | None,
        logging_obj: LiteLLMLoggingObj,
        api_key: str | None,
        api_base: str | None,
        provider_config: NvidiaRivaAudioTranscriptionConfig | None = None,
    ) -> TranscriptionResponse:
        return await asyncio.to_thread(
            self._run_sync,
            model=model,
            audio_file=audio_file,
            optional_params=optional_params,
            litellm_params=litellm_params,
            model_response=model_response,
            timeout=timeout,
            logging_obj=logging_obj,
            api_key=api_key,
            api_base=api_base,
            provider_config=provider_config or NvidiaRivaAudioTranscriptionConfig(),
            atranscription=True,
        )

    def _run_sync(
        self,
        model: str,
        audio_file: FileTypes,
        optional_params: dict,
        litellm_params: dict,
        model_response: TranscriptionResponse,
        timeout: float | str | httpx.Timeout | None,
        logging_obj: LiteLLMLoggingObj,
        api_key: str | None,
        api_base: str | None,
        provider_config: NvidiaRivaAudioTranscriptionConfig,
        atranscription: bool = False,
    ) -> TranscriptionResponse:
        if not api_base:
            raise NvidiaRivaException(
                status_code=400,
                message=(
                    "NVIDIA Riva requires `api_base` (host:port for the gRPC "
                    "endpoint, e.g. `grpc.nvcf.nvidia.com:443` or "
                    "`localhost:50051`). Set it in litellm_params or via "
                    "NVIDIA_RIVA_API_BASE."
                ),
            )

        processed: Final = process_audio_file(audio_file)
        resampled: Final = resample_to_riva_pcm(processed.file_content)

        request_payload: Final = provider_config.transform_audio_transcription_request(
            model=model,
            audio_file=audio_file,
            optional_params=optional_params,
            litellm_params={
                **litellm_params,
                "api_base": api_base,
                "api_key": api_key,
            },
        ).data
        if not isinstance(request_payload, dict):
            raise NvidiaRivaException(
                status_code=500,
                message="NvidiaRivaAudioTranscriptionConfig produced an unexpected request payload type.",
            )

        recognition_config_dict: Final[dict[str, object]] = request_payload["recognition_config"]
        # The wire format is fixed by our resampler; override anything stale
        # the caller passed in so the gRPC config matches the bytes we send.
        recognition_config_dict["sample_rate_hertz"] = RIVA_TARGET_SAMPLE_RATE_HZ
        recognition_config_dict["audio_channel_count"] = RIVA_TARGET_NUM_CHANNELS
        recognition_config_dict["encoding"] = "LINEAR_PCM"

        response_format: Final = request_payload.get("response_format") or "json"
        timestamp_granularities: Final = request_payload.get("timestamp_granularities")
        drop_params: Final = (
            "drop_params" in litellm_params and litellm_params["drop_params"] is True
        ) or litellm.drop_params is True
        riva_offline: Final = _parse_riva_offline(optional_params.get("riva_offline"), drop_params=drop_params)

        try:
            from litellm.llms.nvidia_riva.audio_transcription import riva_grpc
        except ImportError as e:
            raise NvidiaRivaException(status_code=500, message=_RIVA_INSTALL_HINT) from e

        recognition_config: Final = riva_grpc.recognition_config(recognition_config_dict)
        logging_obj.pre_call(
            input=None,
            api_key=api_key,
            additional_args={
                "api_base": api_base,
                "atranscription": atranscription,
                "complete_input_dict": {
                    "recognition_config": recognition_config_dict,
                    "nvcf_function_id_set": bool(optional_params.get("nvcf_function_id")),
                    "use_ssl": optional_params.get("use_ssl"),
                    "riva_offline": riva_offline,
                },
            },
        )

        metadata: Final = riva_grpc.call_metadata(api_key, optional_params.get("nvcf_function_id"))
        try:
            deadline: Final = None if timeout is None else _deadline_seconds(timeout, drop_params)
            with closing(riva_grpc.open_channel(api_base, _use_ssl(optional_params))) as channel:
                responses: Final = (
                    riva_grpc.recognize(channel, recognition_config, resampled.pcm_bytes, metadata, deadline)
                    if riva_offline
                    else riva_grpc.streaming_recognize(
                        channel, recognition_config, resampled.pcm_bytes, metadata, deadline
                    )
                )
                final_results: Final = self._collect_final_results(responses, require_final=not riva_offline)
        except NvidiaRivaException:
            raise
        except Exception as e:
            raise grpc_error_to_litellm_exception(e) from e

        transcription: Final = NvidiaRivaAudioTranscriptionConfig.build_transcription_response(
            final_results=final_results,
            response_format=response_format,
            duration_seconds=resampled.duration_seconds,
            timestamp_granularities=timestamp_granularities,
        )

        stringified_response: Final = dict(transcription)

        logging_obj.post_call(
            input=get_audio_file_name(audio_file),
            api_key=api_key,
            additional_args={"complete_input_dict": recognition_config_dict},
            original_response=stringified_response,
        )

        hidden_params: Final = {
            "model": model,
            "custom_llm_provider": "nvidia_riva",
            "audio_transcription_duration": resampled.duration_seconds,
        }

        final_response: Final[TranscriptionResponse] = convert_to_model_response_object(
            response_object=stringified_response,
            model_response_object=model_response,
            hidden_params=hidden_params,
            response_type="audio_transcription",
        )

        return final_response

    @staticmethod
    def _collect_final_results(stream: Iterable[object], require_final: bool = True) -> Sequence[RivaFinalResult]:
        """
        Walk the gRPC responses, ignore empty / non-final chunks, and return a
        list of normalized final-result dicts. Matching the user's note: the
        ``id`` blocks with no ``results`` are streaming heartbeats and must
        be skipped.
        """
        final_results: Final[list[RivaFinalResult]] = []
        for response in stream:
            results = getattr(response, "results", None) or []
            for result in results:
                if require_final and not getattr(result, "is_final", False):
                    continue
                alternatives = getattr(result, "alternatives", None) or []
                if not alternatives:
                    continue
                top = alternatives[0]
                transcript = getattr(top, "transcript", "") or ""
                words_proto = getattr(top, "words", None) or []
                final_results.append(
                    {
                        "transcript": transcript,
                        "words": [
                            RivaWordTiming(
                                word=getattr(word, "word", ""),
                                start_time_ms=int(getattr(word, "start_time", 0) or 0),
                                end_time_ms=int(getattr(word, "end_time", 0) or 0),
                            )
                            for word in words_proto
                        ],
                    }
                )
        return final_results
