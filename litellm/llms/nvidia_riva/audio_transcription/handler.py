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
3. Construct ``riva.client.Auth`` honoring NVCF (function-id metadata + TLS)
   vs self-hosted (any host:port, optional TLS) modes.
4. Stream the audio through Riva's ``streaming_response_generator`` and
   aggregate ``is_final`` results into a single transcript.
5. Return a normalized ``TranscriptionResponse`` with ``duration`` exposed
   on ``_hidden_params`` so cost calculation works.

``riva-client`` is imported lazily so ``litellm`` core remains usable
without the optional STT extras installed.
"""

import asyncio
import inspect
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from litellm.litellm_core_utils.audio_utils.utils import (
    get_audio_file_name,
    process_audio_file,
)
from litellm.llms.nvidia_riva.audio_transcription.audio_utils import (
    resample_to_riva_pcm,
)
from litellm.llms.nvidia_riva.audio_transcription.transformation import (
    NvidiaRivaAudioTranscriptionConfig,
    RIVA_TARGET_NUM_CHANNELS,
    RIVA_TARGET_SAMPLE_RATE_HZ,
)
from litellm.llms.nvidia_riva.common_utils import (
    NvidiaRivaException,
    grpc_error_to_litellm_exception,
)
from litellm.types.utils import FileTypes, TranscriptionResponse
from litellm.utils import convert_to_model_response_object

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import (
        Logging as LiteLLMLoggingObj,
    )

# Stream audio to Riva in ~50 ms slices (1600 samples at 16 kHz). Matches
# NVIDIA's recommended chunk size for streaming ASR — small enough for
# responsive endpointing, large enough to keep per-RPC overhead low.
_DEFAULT_CHUNK_SAMPLES = 1600
_DEFAULT_CHUNK_BYTES = _DEFAULT_CHUNK_SAMPLES * 2  # int16 = 2 bytes/sample


_RIVA_INSTALL_HINT = (
    "NVIDIA Riva client is not installed. "
    "Install with `pip install 'litellm[stt-nvidia-riva]'`."
)


class NvidiaRivaAudioTranscription:
    """Sync + async entry point for Riva ASR."""

    def audio_transcriptions(
        self,
        model: str,
        audio_file: FileTypes,
        optional_params: dict,
        litellm_params: dict,
        model_response: TranscriptionResponse,
        timeout: float,
        logging_obj: "LiteLLMLoggingObj",
        api_key: Optional[str],
        api_base: Optional[str],
        atranscription: bool = False,
        provider_config: Optional[NvidiaRivaAudioTranscriptionConfig] = None,
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
        timeout: float,
        logging_obj: "LiteLLMLoggingObj",
        api_key: Optional[str],
        api_base: Optional[str],
        provider_config: Optional[NvidiaRivaAudioTranscriptionConfig] = None,
    ) -> TranscriptionResponse:
        # ``riva-client`` exposes a sync streaming generator, so we offload
        # the blocking call to a worker thread to keep the event loop free.
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
        timeout: float,
        logging_obj: "LiteLLMLoggingObj",
        api_key: Optional[str],
        api_base: Optional[str],
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

        processed = process_audio_file(audio_file)
        resampled = resample_to_riva_pcm(processed.file_content)

        request_payload = provider_config.transform_audio_transcription_request(
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

        recognition_config_dict: Dict[str, Any] = request_payload["recognition_config"]
        # The wire format is fixed by our resampler; override anything stale
        # the caller passed in so the gRPC config matches the bytes we send.
        recognition_config_dict["sample_rate_hertz"] = RIVA_TARGET_SAMPLE_RATE_HZ
        recognition_config_dict["audio_channel_count"] = RIVA_TARGET_NUM_CHANNELS
        recognition_config_dict["encoding"] = "LINEAR_PCM"

        response_format = request_payload.get("response_format") or "json"
        timestamp_granularities = request_payload.get("timestamp_granularities")

        riva_module, riva_asr_module = _import_riva()
        auth_obj = self._construct_auth(
            riva_module=riva_module,
            api_base=api_base,
            api_key=api_key,
            optional_params=optional_params,
        )

        recognition_config = self._build_recognition_config_proto(
            riva_asr_module=riva_asr_module,
            recognition_config_dict=recognition_config_dict,
        )
        streaming_config = riva_asr_module.StreamingRecognitionConfig(
            config=recognition_config, interim_results=False
        )

        logging_obj.pre_call(
            input=None,
            api_key=api_key,
            additional_args={
                "api_base": api_base,
                "atranscription": atranscription,
                "complete_input_dict": {
                    "recognition_config": recognition_config_dict,
                    "nvcf_function_id_set": bool(
                        optional_params.get("nvcf_function_id")
                    ),
                    "use_ssl": optional_params.get("use_ssl"),
                },
            },
        )

        try:
            asr_service = riva_module.ASRService(auth_obj)
            audio_chunks = self._iter_audio_chunks(resampled.pcm_bytes)
            stream_kwargs: Dict[str, Any] = {
                "audio_chunks": audio_chunks,
                "streaming_config": streaming_config,
            }
            # Forward the deadline so the stream cannot block forever if the
            # server stalls. Older riva-client versions do not accept a
            # ``timeout`` kwarg, so pass it only when supported.
            if timeout is not None and self._supports_timeout_kwarg(
                asr_service.streaming_response_generator
            ):
                stream_kwargs["timeout"] = float(timeout)
            stream = asr_service.streaming_response_generator(**stream_kwargs)
            final_results = self._collect_final_results(stream)
        except NvidiaRivaException:
            raise
        except Exception as e:
            raise grpc_error_to_litellm_exception(e) from e

        transcription = NvidiaRivaAudioTranscriptionConfig.build_transcription_response(
            final_results=final_results,
            response_format=response_format,
            duration_seconds=resampled.duration_seconds,
            timestamp_granularities=timestamp_granularities,
        )

        stringified_response = dict(transcription)

        logging_obj.post_call(
            input=get_audio_file_name(audio_file),
            api_key=api_key,
            additional_args={"complete_input_dict": recognition_config_dict},
            original_response=stringified_response,
        )

        hidden_params = {
            "model": model,
            "custom_llm_provider": "nvidia_riva",
            "audio_transcription_duration": resampled.duration_seconds,
        }

        final_response: TranscriptionResponse = convert_to_model_response_object(  # type: ignore
            response_object=stringified_response,
            model_response_object=model_response,
            hidden_params=hidden_params,
            response_type="audio_transcription",
        )

        return final_response

    def _construct_auth(
        self,
        riva_module: Any,
        api_base: str,
        api_key: Optional[str],
        optional_params: dict,
    ) -> Any:
        """
        Build a ``riva.client.Auth`` object.

        - When ``nvcf_function_id`` is provided we attach the NVCF
          ``function-id`` and bearer ``authorization`` metadata, and default
          ``use_ssl`` to True (NVCF endpoints are TLS-only).
        - Otherwise (self-hosted) we default ``use_ssl`` to False but still
          honor an explicit override — self-hosted Riva behind an ingress
          with TLS termination is a real deployment topology.
        """
        nvcf_function_id = optional_params.get("nvcf_function_id")
        use_ssl_override = optional_params.get("use_ssl")
        use_ssl = (
            bool(use_ssl_override)
            if use_ssl_override is not None
            else bool(nvcf_function_id)
        )

        metadata: List[Tuple[str, str]] = []
        if nvcf_function_id:
            metadata.append(("function-id", str(nvcf_function_id)))
        if api_key:
            metadata.append(("authorization", f"Bearer {api_key}"))

        try:
            return riva_module.Auth(
                uri=api_base, use_ssl=use_ssl, metadata_args=metadata
            )
        except TypeError:
            # Older riva-client signatures used positional-only args.
            return riva_module.Auth(None, use_ssl, api_base, metadata)

    def _build_recognition_config_proto(
        self, riva_asr_module: Any, recognition_config_dict: Dict[str, Any]
    ):
        encoding_name = (
            recognition_config_dict.get("encoding") or "LINEAR_PCM"
        ).upper()
        encoding_enum = getattr(
            riva_asr_module.AudioEncoding,
            encoding_name,
            riva_asr_module.AudioEncoding.LINEAR_PCM,
        )

        config = riva_asr_module.RecognitionConfig(
            encoding=encoding_enum,
            sample_rate_hertz=int(recognition_config_dict["sample_rate_hertz"]),
            language_code=recognition_config_dict["language_code"],
            audio_channel_count=int(recognition_config_dict["audio_channel_count"]),
            enable_automatic_punctuation=bool(
                recognition_config_dict.get("enable_automatic_punctuation", True)
            ),
            enable_word_time_offsets=bool(
                recognition_config_dict.get("enable_word_time_offsets", False)
            ),
            max_alternatives=int(recognition_config_dict.get("max_alternatives", 1)),
            model=recognition_config_dict.get("model", "") or "",
            verbatim_transcripts=bool(
                recognition_config_dict.get("verbatim_transcripts", False)
            ),
            profanity_filter=bool(
                recognition_config_dict.get("profanity_filter", False)
            ),
        )

        endpointing = recognition_config_dict.get("endpointing_config")
        if isinstance(endpointing, dict) and endpointing:
            try:
                ep = riva_asr_module.EndpointingConfig(**endpointing)
                config.endpointing_config.CopyFrom(ep)
            except Exception:
                # If the user supplied an unknown EndpointingConfig field
                # (older Riva server), fall back to Riva's defaults rather
                # than failing the whole request.
                pass

        return config

    @staticmethod
    def _supports_timeout_kwarg(callable_obj: Any) -> bool:
        try:
            sig = inspect.signature(callable_obj)
        except (TypeError, ValueError):
            return False
        params = sig.parameters
        if "timeout" in params:
            return True
        return any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())

    @staticmethod
    def _iter_audio_chunks(pcm_bytes: bytes):
        for offset in range(0, len(pcm_bytes), _DEFAULT_CHUNK_BYTES):
            chunk = pcm_bytes[offset : offset + _DEFAULT_CHUNK_BYTES]
            if not chunk:
                continue
            yield chunk

    @staticmethod
    def _collect_final_results(stream) -> List[Dict[str, Any]]:
        """
        Walk the gRPC stream, ignore empty / non-final chunks, and return a
        list of normalized final-result dicts. Matching the user's note: the
        ``id`` blocks with no ``results`` are streaming heartbeats and must
        be skipped.
        """
        final_results: List[Dict[str, Any]] = []
        for response in stream:
            results = getattr(response, "results", None) or []
            for result in results:
                if not getattr(result, "is_final", False):
                    continue
                alternatives = getattr(result, "alternatives", None) or []
                if not alternatives:
                    continue
                top = alternatives[0]
                transcript = getattr(top, "transcript", "") or ""
                words_proto = getattr(top, "words", None) or []
                words = []
                for word in words_proto:
                    words.append(
                        {
                            "word": getattr(word, "word", ""),
                            "start_time_ms": int(getattr(word, "start_time", 0) or 0),
                            "end_time_ms": int(getattr(word, "end_time", 0) or 0),
                        }
                    )
                final_results.append({"transcript": transcript, "words": words})
        return final_results


def _import_riva():
    """
    Lazy import of ``riva.client`` and ``riva.client.proto.riva_asr_pb2``.

    We try the SDK first (preferred) and fall back to importing the proto
    module separately when the SDK packaging changes between versions.
    """
    try:
        import riva.client as riva_client  # type: ignore
    except ImportError as e:
        raise NvidiaRivaException(status_code=500, message=_RIVA_INSTALL_HINT) from e

    riva_asr_module = riva_client
    if not hasattr(riva_asr_module, "RecognitionConfig"):
        try:
            import riva.client.proto.riva_asr_pb2 as riva_asr_pb2  # type: ignore

            riva_asr_module = riva_asr_pb2
        except ImportError as e:
            raise NvidiaRivaException(
                status_code=500, message=_RIVA_INSTALL_HINT
            ) from e

    return riva_client, riva_asr_module
