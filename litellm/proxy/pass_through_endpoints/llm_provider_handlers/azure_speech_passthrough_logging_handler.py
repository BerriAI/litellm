from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Final
from urllib.parse import urlparse

import httpx

from litellm._logging import verbose_proxy_logger
from litellm.constants import (
    AZURE_SPEECH_BATCH_MODEL,
    AZURE_SPEECH_CUSTOM_LLM_PROVIDER,
    AZURE_SPEECH_PRICING_MODEL,
    AZURE_SPEECH_SHORT_AUDIO_MODEL,
    AZURE_SPEECH_SHORT_AUDIO_PATH_PREFIX,
    AZURE_SPEECH_TICKS_PER_SECOND,
)
from litellm.cost_calculator import transcription_cost
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.litellm_core_utils.litellm_logging import (
    get_standard_logging_object_payload,
)
from litellm.proxy._types import PassThroughEndpointLoggingTypedDict
from litellm.types.utils import StandardPassThroughResponseObject


class AzureSpeechPassthroughLoggingHandler:
    @staticmethod
    def _is_short_audio_route(url_route: str) -> bool:
        return urlparse(url_route).path.startswith(AZURE_SPEECH_SHORT_AUDIO_PATH_PREFIX)

    @staticmethod
    def _model_from_url_route(url_route: str) -> str:
        if AzureSpeechPassthroughLoggingHandler._is_short_audio_route(url_route):
            return f"{AZURE_SPEECH_CUSTOM_LLM_PROVIDER}/{AZURE_SPEECH_SHORT_AUDIO_MODEL}"
        return f"{AZURE_SPEECH_CUSTOM_LLM_PROVIDER}/{AZURE_SPEECH_BATCH_MODEL}"

    @staticmethod
    def _recognized_audio_seconds(response_body: Mapping[str, object] | Sequence[object] | None) -> float:
        if not isinstance(response_body, Mapping):
            return 0.0
        offset: Final = response_body.get("Offset")
        duration: Final = response_body.get("Duration")
        if not isinstance(offset, int) or not isinstance(duration, int):
            return 0.0
        return (offset + duration) / AZURE_SPEECH_TICKS_PER_SECOND

    @staticmethod
    def _response_cost(url_route: str, response_body: Mapping[str, object] | Sequence[object] | None) -> float:
        if not AzureSpeechPassthroughLoggingHandler._is_short_audio_route(url_route):
            return 0.0
        audio_seconds: Final = AzureSpeechPassthroughLoggingHandler._recognized_audio_seconds(response_body)
        if audio_seconds <= 0.0:
            return 0.0
        try:
            prompt_cost, completion_cost = transcription_cost(
                model=AZURE_SPEECH_PRICING_MODEL,
                custom_llm_provider="azure",
                duration=audio_seconds,
            )
        except Exception as e:  # noqa: BLE001  # a missing price entry must not drop the spend log row
            verbose_proxy_logger.warning(
                "No price for %s, logging Azure Speech call at zero cost: %s", AZURE_SPEECH_PRICING_MODEL, e
            )
            return 0.0
        return prompt_cost + completion_cost

    @staticmethod
    def azure_speech_passthrough_handler(
        httpx_response: httpx.Response,
        response_body: Mapping[str, object] | Sequence[object] | None,
        logging_obj: LiteLLMLoggingObj,
        url_route: str,
        result: str,
        start_time: datetime,
        end_time: datetime,
        cache_hit: bool,
        request_body: Mapping[str, object],
        **kwargs: object,  # kwargs-ok: the passthrough logging dispatch forwards shared logging kwargs to every handler
    ) -> PassThroughEndpointLoggingTypedDict:
        try:
            model_name: Final = AzureSpeechPassthroughLoggingHandler._model_from_url_route(url_route)
            response_cost: Final = AzureSpeechPassthroughLoggingHandler._response_cost(url_route, response_body)

            updated_kwargs: Final = {  # mutable-ok: the logging pipeline requires a plain kwargs dict
                **kwargs,
                "model": model_name,
                "custom_llm_provider": AZURE_SPEECH_CUSTOM_LLM_PROVIDER,
                "response_cost": response_cost,
            }
            logging_obj.model_call_details.update(
                model=model_name,
                custom_llm_provider=AZURE_SPEECH_CUSTOM_LLM_PROVIDER,
                response_cost=response_cost,
            )

            standard_logging_object: Final = get_standard_logging_object_payload(
                kwargs=updated_kwargs,
                init_response_obj=StandardPassThroughResponseObject(response=result),
                start_time=start_time,
                end_time=end_time,
                logging_obj=logging_obj,
                status="success",
            )

            handler_payload: Final[PassThroughEndpointLoggingTypedDict] = {
                "result": StandardPassThroughResponseObject(response=result),
                "kwargs": {**updated_kwargs, "standard_logging_object": standard_logging_object},
            }
        except Exception as e:  # noqa: BLE001  # logging must never fail the forwarded request
            verbose_proxy_logger.exception("Error in Azure Speech passthrough logging handler: %s", e)
            fallback_payload: Final[PassThroughEndpointLoggingTypedDict] = {
                "result": StandardPassThroughResponseObject(response=result),
                "kwargs": kwargs,
            }
            return fallback_payload
        return handler_payload
