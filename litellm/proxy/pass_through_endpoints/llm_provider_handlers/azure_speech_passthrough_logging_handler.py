from collections.abc import Mapping
from datetime import datetime
from typing import Final
from urllib.parse import urlparse

import httpx

from litellm._logging import verbose_proxy_logger
from litellm.constants import (
    AZURE_SPEECH_BATCH_MODEL,
    AZURE_SPEECH_CUSTOM_LLM_PROVIDER,
    AZURE_SPEECH_SHORT_AUDIO_MODEL,
    AZURE_SPEECH_SHORT_AUDIO_PATH_PREFIX,
)
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.litellm_core_utils.litellm_logging import (
    get_standard_logging_object_payload,
)
from litellm.proxy._types import PassThroughEndpointLoggingTypedDict
from litellm.types.utils import StandardPassThroughResponseObject


class AzureSpeechPassthroughLoggingHandler:
    @staticmethod
    def _model_from_url_route(url_route: str) -> str:
        path: Final = urlparse(url_route).path
        if path.startswith(AZURE_SPEECH_SHORT_AUDIO_PATH_PREFIX):
            return f"{AZURE_SPEECH_CUSTOM_LLM_PROVIDER}/{AZURE_SPEECH_SHORT_AUDIO_MODEL}"
        return f"{AZURE_SPEECH_CUSTOM_LLM_PROVIDER}/{AZURE_SPEECH_BATCH_MODEL}"

    @staticmethod
    def azure_speech_passthrough_handler(
        httpx_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        url_route: str,
        result: str,
        start_time: datetime,
        end_time: datetime,
        cache_hit: bool,
        request_body: Mapping[str, object],
        **kwargs: object,  # kwargs-ok: the passthrough logging dispatch forwards shared logging kwargs to every handler
    ) -> PassThroughEndpointLoggingTypedDict:
        """
        Records model and provider for an Azure AI Speech REST call. Azure bills per audio
        hour after the fact and neither the short-audio response nor the batch job carries
        a billable duration this path can trust, so response_cost is recorded as 0.0 rather
        than estimated.
        """
        try:
            model_name: Final = AzureSpeechPassthroughLoggingHandler._model_from_url_route(url_route)

            updated_kwargs: Final = {  # mutable-ok: the logging pipeline requires a plain kwargs dict
                **kwargs,
                "model": model_name,
                "custom_llm_provider": AZURE_SPEECH_CUSTOM_LLM_PROVIDER,
                "response_cost": 0.0,
            }
            logging_obj.model_call_details.update(
                model=model_name,
                custom_llm_provider=AZURE_SPEECH_CUSTOM_LLM_PROVIDER,
                response_cost=0.0,
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
