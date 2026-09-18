from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Final
from urllib.parse import urlparse

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.deepgram.common_utils import (
    deepgram_listen_addon_pricing_models,
    deepgram_listen_audio_seconds,
    deepgram_listen_base_pricing_models,
    deepgram_listen_channel_count,
    deepgram_listen_model,
    deepgram_listen_transcript,
)
from litellm.proxy._types import PassThroughEndpointLoggingTypedDict
from litellm.types.utils import TranscriptionResponse

DEEPGRAM_LISTEN_ROUTE_SUFFIX: Final = "/listen"


def _registry_cost(response: TranscriptionResponse, pricing_model: str) -> float | None:
    try:
        return litellm.completion_cost(
            completion_response=response,
            model=pricing_model,
            custom_llm_provider=litellm.LlmProviders.DEEPGRAM.value,
            call_type="transcription",
        )
    except Exception as e:  # noqa: BLE001  # an unpriced entry must not lose the spend row, only its cost
        verbose_proxy_logger.debug("Deepgram listen passthrough: no registry price for '%s': %s", pricing_model, e)
        return None


def _audio_cost(response: TranscriptionResponse, upstream_url: str) -> float | None:
    base_cost: Final = next(
        (
            cost
            for pricing_model in deepgram_listen_base_pricing_models(upstream_url)
            if (cost := _registry_cost(response, pricing_model)) is not None
        ),
        None,
    )
    if base_cost is None:
        verbose_proxy_logger.warning(
            "Deepgram listen passthrough: no pricing for model '%s'", deepgram_listen_model(upstream_url)
        )
        return None
    addon_costs: Final = tuple(
        _registry_cost(response, pricing_model) for pricing_model in deepgram_listen_addon_pricing_models(upstream_url)
    )
    return base_cost + sum(cost for cost in addon_costs if cost is not None)


class DeepgramListenPassthroughLoggingHandler:
    @staticmethod
    def is_deepgram_listen_route(url_route: str) -> bool:
        path: Final = urlparse(url_route).path
        return "/deepgram/" in path and path.endswith(DEEPGRAM_LISTEN_ROUTE_SUFFIX)

    def deepgram_listen_passthrough_handler(
        self,
        websocket_messages: Sequence[Mapping[str, object]],
        logging_obj: LiteLLMLoggingObj,
        upstream_url: str,
        kwargs: Mapping[str, object] = MappingProxyType({}),
    ) -> PassThroughEndpointLoggingTypedDict:
        model: Final = deepgram_listen_model(upstream_url)
        audio_seconds: Final = deepgram_listen_audio_seconds(websocket_messages)
        channels: Final = deepgram_listen_channel_count(websocket_messages, upstream_url)
        billed_seconds: Final = audio_seconds * channels
        response: Final = TranscriptionResponse(text=deepgram_listen_transcript(websocket_messages))
        response._hidden_params["audio_transcription_duration"] = billed_seconds  # pyright: ignore[reportPrivateUsage]  # the cost calculator reads the billed duration off the response's hidden params
        response_cost: Final = _audio_cost(response, upstream_url)
        response._hidden_params["response_cost"] = response_cost  # pyright: ignore[reportPrivateUsage]  # the logger reads a precomputed cost off the response's hidden params

        provider: Final = litellm.LlmProviders.DEEPGRAM.value
        logging_obj.model = model  # rebind-ok: the spend logger reads model and cost off the shared logging object
        logging_obj.model_call_details["model"] = model  # rebind-ok: same shared logging object
        logging_obj.model_call_details["custom_llm_provider"] = provider  # rebind-ok: same shared logging object
        logging_obj.model_call_details["response_cost"] = response_cost  # rebind-ok: same shared logging object
        verbose_proxy_logger.debug(
            "Deepgram listen passthrough cost tracking: model %s, audio seconds %s, channels %s, cost %s",
            model,
            audio_seconds,
            channels,
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
