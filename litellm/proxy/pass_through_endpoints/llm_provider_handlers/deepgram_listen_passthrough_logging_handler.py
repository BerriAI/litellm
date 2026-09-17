"""
Cost tracking for Deepgram's streaming ``/v1/listen`` WebSocket. Deepgram bills the audio it processed, which it
reports as ``duration`` on the closing ``Metadata`` frame; a stream that ends without one is billed on the furthest
``start + duration`` across its ``Results`` frames
"""

import math
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Final
from urllib.parse import parse_qs, urlparse

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.constants import DEEPGRAM_LISTEN_DEFAULT_MODEL
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.proxy._types import PassThroughEndpointLoggingTypedDict
from litellm.types.utils import TranscriptionResponse

DEEPGRAM_LISTEN_ROUTE_SUFFIX: Final = "/listen"


def _seconds(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) and value >= 0 else None


def _results_frame_end(frame: Mapping[str, object]) -> float | None:
    start: Final = _seconds(frame.get("start"))
    duration: Final = _seconds(frame.get("duration"))
    return None if start is None or duration is None else start + duration


def _final_transcript(frame: Mapping[str, object]) -> str | None:
    if frame.get("is_final") is not True:
        return None
    channel: Final = frame.get("channel")
    alternatives: Final = channel.get("alternatives") if isinstance(channel, Mapping) else None
    first: Final = alternatives[0] if isinstance(alternatives, list) and alternatives else None
    transcript: Final = first.get("transcript") if isinstance(first, Mapping) else None
    return transcript if isinstance(transcript, str) and transcript else None


def deepgram_listen_audio_seconds(websocket_messages: Sequence[Mapping[str, object]]) -> float:
    metadata_durations: Final = tuple(
        duration
        for frame in websocket_messages
        if frame.get("type") == "Metadata"
        if (duration := _seconds(frame.get("duration"))) is not None
    )
    if metadata_durations:
        return metadata_durations[-1]
    return max(
        (
            end
            for frame in websocket_messages
            if frame.get("type") == "Results"
            if (end := _results_frame_end(frame)) is not None
        ),
        default=0.0,
    )


def deepgram_listen_transcript(websocket_messages: Sequence[Mapping[str, object]]) -> str:
    return " ".join(
        transcript
        for frame in websocket_messages
        if frame.get("type") == "Results"
        if (transcript := _final_transcript(frame)) is not None
    )


def deepgram_listen_model(upstream_url: str) -> str:
    models: Final = parse_qs(urlparse(upstream_url).query).get("model")
    return models[0] if models else DEEPGRAM_LISTEN_DEFAULT_MODEL


def _audio_cost(response: TranscriptionResponse, model: str) -> float | None:
    try:
        return litellm.completion_cost(
            completion_response=response,
            model=model,
            custom_llm_provider=litellm.LlmProviders.DEEPGRAM.value,
            call_type="transcription",
        )
    except Exception as e:  # noqa: BLE001  # an unpriced model must not lose the spend row, only its cost
        verbose_proxy_logger.warning("Deepgram listen passthrough: no pricing for model '%s': %s", model, e)
        return None


class DeepgramListenPassthroughLoggingHandler:
    @staticmethod
    def is_deepgram_listen_route(url_route: str) -> bool:
        path: Final = urlparse(url_route).path
        return path.startswith("/deepgram/") and path.endswith(DEEPGRAM_LISTEN_ROUTE_SUFFIX)

    def deepgram_listen_passthrough_handler(
        self,
        websocket_messages: Sequence[Mapping[str, object]],
        logging_obj: LiteLLMLoggingObj,
        upstream_url: str,
        kwargs: Mapping[str, object] = MappingProxyType({}),
    ) -> PassThroughEndpointLoggingTypedDict:
        model: Final = deepgram_listen_model(upstream_url)
        audio_seconds: Final = deepgram_listen_audio_seconds(websocket_messages)
        response: Final = TranscriptionResponse(text=deepgram_listen_transcript(websocket_messages))
        response._hidden_params["audio_transcription_duration"] = audio_seconds  # pyright: ignore[reportPrivateUsage]  # the cost calculator reads the billed duration off the response's hidden params
        response_cost: Final = _audio_cost(response, model)
        response._hidden_params["response_cost"] = response_cost  # pyright: ignore[reportPrivateUsage]  # the logger reads a precomputed cost off the response's hidden params

        provider: Final = litellm.LlmProviders.DEEPGRAM.value
        logging_obj.model = model  # rebind-ok: the spend logger reads model and cost off the shared logging object
        logging_obj.model_call_details["model"] = model  # rebind-ok: same shared logging object
        logging_obj.model_call_details["custom_llm_provider"] = provider  # rebind-ok: same shared logging object
        logging_obj.model_call_details["response_cost"] = response_cost  # rebind-ok: same shared logging object
        verbose_proxy_logger.debug(
            "Deepgram listen passthrough cost tracking: model %s, audio seconds %s, cost %s",
            model,
            audio_seconds,
            response_cost,
        )
        logging_result: Final[PassThroughEndpointLoggingTypedDict] = {
            "result": response,
            "kwargs": {
                **kwargs,
                "model": model,
                "custom_llm_provider": provider,
                "response_cost": response_cost,
            },
        }
        return logging_result
