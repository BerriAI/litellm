"""
Deepgram streaming Speech-to-Text (`/v1/listen`) WebSocket passthrough logging handler

Computes duration-based cost tracking for Deepgram's realtime `/v1/listen` WebSocket,
mirroring the batch `/v1/audio/transcriptions` pricing (input_cost_per_second). Deepgram
reports the amount of audio it processed in a final `Metadata` message; when that message
is missing (for example an abruptly closed stream) the end timestamp of the last transcript
segment is used instead.
"""

from datetime import datetime

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import PassThroughEndpointLoggingTypedDict
from litellm.types.utils import ModelResponse, Usage
from litellm.utils import get_model_info

DEEPGRAM_DEFAULT_MODEL = "nova-3"


class DeepgramLivePassthroughLoggingHandler:
    """Cost tracking and logging for the Deepgram streaming `/v1/listen` passthrough."""

    @staticmethod
    def _coerce_seconds(value: object) -> float | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            return float(value)
        return None

    @staticmethod
    def extract_audio_duration_seconds(
        websocket_messages: list[dict],
    ) -> float:
        """
        Determine how many seconds of audio Deepgram processed.

        Prefers the `duration` reported in the final `Metadata` message (Deepgram's own
        billed quantity). Falls back to the furthest `start + duration` seen across
        `Results` messages so a stream that ends before the Metadata frame still bills
        for the audio that was transcribed.
        """
        metadata_duration: float | None = None
        transcript_end = 0.0

        for message in websocket_messages:
            if not isinstance(message, dict):
                continue
            message_type = message.get("type")
            if message_type == "Metadata":
                duration = DeepgramLivePassthroughLoggingHandler._coerce_seconds(message.get("duration"))
                if duration is not None:
                    metadata_duration = duration
            elif message_type == "Results":
                start = DeepgramLivePassthroughLoggingHandler._coerce_seconds(message.get("start")) or 0.0
                duration = DeepgramLivePassthroughLoggingHandler._coerce_seconds(message.get("duration")) or 0.0
                transcript_end = max(transcript_end, start + duration)

        if metadata_duration is not None:
            return metadata_duration
        return transcript_end

    @staticmethod
    def extract_transcript(websocket_messages: list[dict]) -> str:
        """Concatenate the finalized transcript segments for logging visibility."""
        segments = tuple(
            alternative.get("transcript", "")
            for message in websocket_messages
            if isinstance(message, dict) and message.get("type") == "Results" and message.get("is_final") is True
            for alternative in message.get("channel", {}).get("alternatives", [])[:1]
            if isinstance(alternative, dict) and alternative.get("transcript")
        )
        return " ".join(segment for segment in segments if segment).strip()

    @staticmethod
    def _cost_per_second(model: str) -> float:
        try:
            model_info = get_model_info(model=model, custom_llm_provider="deepgram")
        except Exception as e:  # noqa: BLE001  # get_model_info raises varied errors for unmapped models; treat as unpriced
            verbose_proxy_logger.warning(f"Deepgram live passthrough: no pricing for '{model}', billing $0. Error: {e}")
            return 0.0
        return model_info.get("input_cost_per_second") or 0.0

    def deepgram_live_passthrough_handler(
        self,
        websocket_messages: list[dict],
        start_time: datetime,
        end_time: datetime,
        **kwargs,
    ) -> PassThroughEndpointLoggingTypedDict:
        model = kwargs.get("model") or DEEPGRAM_DEFAULT_MODEL
        prefixed_model = model if model.startswith("deepgram/") else f"deepgram/{model}"

        duration_seconds = self.extract_audio_duration_seconds(websocket_messages)
        response_cost = duration_seconds * self._cost_per_second(prefixed_model)

        litellm_model_response = ModelResponse(
            id=f"deepgram-listen-{start_time.timestamp()}",
            created=int(start_time.timestamp()),
            model=prefixed_model,
            usage=Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
            choices=[],
        )

        updated_kwargs = {
            **kwargs,
            "response_cost": response_cost,
            "model": model,
            "custom_llm_provider": "deepgram",
        }

        verbose_proxy_logger.debug(
            f"Deepgram live passthrough cost tracking - model: {prefixed_model}, "
            f"audio_seconds: {duration_seconds:.3f}, transcript: '{self.extract_transcript(websocket_messages)}', "
            f"cost: ${response_cost:.6f}"
        )

        return {
            "result": litellm_model_response,
            "kwargs": updated_kwargs,
        }
