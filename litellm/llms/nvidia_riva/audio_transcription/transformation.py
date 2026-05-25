"""
Translates from OpenAI's `/v1/audio/transcriptions` to NVIDIA Riva's gRPC
streaming recognize API.

Riva is gRPC-only, so unlike most providers in this directory the request
"transformation" produced here is a structured dict consumed directly by the
gRPC handler (rather than HTTP form-data). The handler builds Riva
``RecognitionConfig`` / ``StreamingRecognitionConfig`` protobufs from this
dict at call time.

Reference: https://docs.nvidia.com/deeplearning/riva/user-guide/docs/asr/asr-overview.html
"""

from typing import Any, Dict, List, Optional, Union

from httpx import Headers, Response

from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.types.llms.openai import (
    AllMessageValues,
    OpenAIAudioTranscriptionOptionalParams,
)
from litellm.types.utils import FileTypes, TranscriptionResponse

from ...base_llm.audio_transcription.transformation import (
    AudioTranscriptionRequestData,
    BaseAudioTranscriptionConfig,
)
from ..common_utils import NvidiaRivaException

# Riva expects a fixed wire format for the audio chunks we stream in.
RIVA_TARGET_SAMPLE_RATE_HZ = 16000
RIVA_TARGET_NUM_CHANNELS = 1
RIVA_TARGET_ENCODING = "LINEAR_PCM"


class NvidiaRivaAudioTranscriptionConfig(BaseAudioTranscriptionConfig):
    """
    Config for NVIDIA Riva ASR (gRPC).

    Supports both NVCF-hosted (``api_base=grpc.nvcf.nvidia.com:443`` +
    ``nvcf_function_id``) and self-hosted deployments (any ``host:port``,
    optional TLS via ``use_ssl``).
    """

    def get_supported_openai_params(
        self, model: str
    ) -> List[OpenAIAudioTranscriptionOptionalParams]:
        # Riva natively understands language + word timestamps.
        # `response_format` is honored at response-shaping time in the handler.
        return ["language", "response_format", "timestamp_granularities"]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        for key, value in non_default_params.items():
            if value is None:
                continue

            if key == "language":
                optional_params["language_code"] = self._normalize_language_code(value)
            elif key == "timestamp_granularities":
                # OpenAI accepts ["word"], ["segment"], or both. Riva only
                # natively exposes word timing, so we toggle it on whenever
                # "word" is requested. Segment timing is reconstructed in the
                # response transformer.
                if isinstance(value, list) and "word" in value:
                    optional_params["enable_word_time_offsets"] = True
                optional_params["timestamp_granularities"] = value
            elif key == "response_format":
                # Stored verbatim; consumed by transform_audio_transcription_response.
                optional_params["response_format"] = value
            else:
                optional_params[key] = value

        return optional_params

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, Headers]
    ) -> BaseLLMException:
        return NvidiaRivaException(
            message=error_message, status_code=status_code, headers=headers
        )

    def transform_audio_transcription_request(
        self,
        model: str,
        audio_file: FileTypes,
        optional_params: dict,
        litellm_params: dict,
    ) -> AudioTranscriptionRequestData:
        """
        Build a structured dict that the gRPC handler consumes. We do *not*
        construct protobufs here, so this module remains importable without
        ``nvidia-riva-client`` being installed (matching how other providers
        defer SDK imports to handler-call time).
        """
        recognition_config = self._build_recognition_config_dict(
            model=model,
            optional_params=optional_params,
        )

        endpointing_config = self._build_endpointing_config_dict(optional_params)
        if endpointing_config is not None:
            recognition_config["endpointing_config"] = endpointing_config

        request_payload: Dict[str, Any] = {
            "recognition_config": recognition_config,
            "response_format": optional_params.get("response_format") or "json",
            "timestamp_granularities": optional_params.get("timestamp_granularities"),
        }

        return AudioTranscriptionRequestData(data=request_payload, files=None)

    def transform_audio_transcription_response(
        self,
        raw_response: Response,
    ) -> TranscriptionResponse:
        # Not used: Riva responses come from a gRPC stream, not an httpx
        # response. The handler calls _build_transcription_response directly.
        raise NotImplementedError(
            "NvidiaRivaAudioTranscriptionConfig.transform_audio_transcription_response "
            "is not used. The handler builds the TranscriptionResponse directly "
            "from Riva's gRPC streaming results."
        )

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> dict:
        # gRPC auth is constructed in the handler, not via HTTP headers.
        return headers

    def _build_recognition_config_dict(
        self, model: str, optional_params: dict
    ) -> Dict[str, Any]:
        """
        Build the Riva ``RecognitionConfig`` shape as a plain dict.

        ``model`` is intentionally left empty when the user has not supplied
        ``riva_model_name``. Riva auto-selects the right deployment from
        ``language_code`` + ``sample_rate_hertz``. NVIDIA's internal
        deployment names (e.g. ``parakeet-1.1b-en-US-asr-streaming-...``)
        change across NIM versions, regions, and self-hosted builds, so
        hardcoding any name here would break unpredictably.
        """
        return {
            "language_code": optional_params.get("language_code", "en-US"),
            "sample_rate_hertz": optional_params.get(
                "sample_rate_hertz", RIVA_TARGET_SAMPLE_RATE_HZ
            ),
            "encoding": optional_params.get("encoding", RIVA_TARGET_ENCODING),
            "audio_channel_count": optional_params.get(
                "audio_channel_count", RIVA_TARGET_NUM_CHANNELS
            ),
            "enable_automatic_punctuation": optional_params.get(
                "enable_automatic_punctuation", True
            ),
            "enable_word_time_offsets": bool(
                optional_params.get("enable_word_time_offsets", False)
            ),
            "max_alternatives": optional_params.get("max_alternatives", 1),
            "model": optional_params.get("riva_model_name", ""),
            "verbatim_transcripts": optional_params.get("verbatim_transcripts", False),
            "profanity_filter": optional_params.get("profanity_filter", False),
        }

    def _build_endpointing_config_dict(
        self, optional_params: dict
    ) -> Optional[Dict[str, Any]]:
        """
        Translate an OpenAI-style ``chunking_strategy`` into Riva's
        ``EndpointingConfig`` shape, or pass through an explicit
        ``endpointing_config`` dict.

        Returns ``None`` when neither is provided so Riva uses its built-in
        VAD defaults.
        """
        explicit = optional_params.get("endpointing_config")
        if isinstance(explicit, dict):
            return dict(explicit)

        chunking = optional_params.get("chunking_strategy")
        if chunking in (None, "auto"):
            return None

        if isinstance(chunking, dict) and chunking.get("type") == "server_vad":
            config: Dict[str, Any] = {}
            if "threshold" in chunking:
                threshold = float(chunking["threshold"])
                config["start_threshold"] = threshold
                config["stop_threshold"] = threshold
            if "silence_duration_ms" in chunking:
                config["stop_history"] = int(chunking["silence_duration_ms"])
            if "prefix_padding_ms" in chunking:
                config["stop_history_eou"] = int(chunking["prefix_padding_ms"])
            return config or None

        return None

    @staticmethod
    def _normalize_language_code(language: str) -> str:
        """
        OpenAI accepts bare ISO-639 codes like ``en``; Riva requires BCP-47
        like ``en-US``. Normalize the most common bare codes; pass through
        anything that already looks like BCP-47.
        """
        if not isinstance(language, str) or not language:
            return "en-US"
        if "-" in language:
            return language
        bare_to_bcp47 = {
            "en": "en-US",
            "es": "es-ES",
            "de": "de-DE",
            "fr": "fr-FR",
            "it": "it-IT",
            "pt": "pt-BR",
            "ja": "ja-JP",
            "ko": "ko-KR",
            "zh": "zh-CN",
            "ru": "ru-RU",
            "hi": "hi-IN",
            "ar": "ar-SA",
        }
        return bare_to_bcp47.get(language.lower(), language)

    @staticmethod
    def build_transcription_response(
        final_results: List[Dict[str, Any]],
        response_format: str,
        duration_seconds: Optional[float],
        timestamp_granularities: Optional[List[str]],
    ) -> TranscriptionResponse:
        """
        Aggregate a list of normalized "final result" dicts into a
        ``TranscriptionResponse`` shaped for the requested ``response_format``.

        Each entry in ``final_results`` is expected to look like::

            {
                "transcript": str,
                "words": [{"word": str, "start_time_ms": int, "end_time_ms": int}, ...],
            }

        which the handler produces by walking the gRPC stream and keeping
        only ``result.is_final`` entries (empty/non-final chunks are
        ignored).
        """
        full_transcript = "".join(
            (item.get("transcript") or "") for item in final_results
        ).strip()

        response = TranscriptionResponse(text=full_transcript)
        response["task"] = "transcribe"

        if response_format == "verbose_json":
            words: List[Dict[str, Any]] = []
            if timestamp_granularities and "word" in timestamp_granularities:
                for item in final_results:
                    for word in item.get("words", []) or []:
                        words.append(
                            {
                                "word": word.get("word", ""),
                                "start": (float(word.get("start_time_ms", 0)) / 1000.0),
                                "end": float(word.get("end_time_ms", 0)) / 1000.0,
                            }
                        )
            if words:
                response["words"] = words
            if duration_seconds is not None:
                response["duration"] = duration_seconds

        return response
