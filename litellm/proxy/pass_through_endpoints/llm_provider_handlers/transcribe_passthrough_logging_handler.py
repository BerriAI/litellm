from collections.abc import Mapping
from datetime import datetime
from functools import lru_cache
from typing import Final

import httpx

from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.litellm_core_utils.litellm_logging import (
    get_standard_logging_object_payload,
)
from litellm.proxy._types import PassThroughEndpointLoggingTypedDict
from litellm.types.utils import StandardPassThroughResponseObject

TRANSCRIBE_TARGET_PREFIX: Final = "Transcribe"
TRANSCRIBE_CUSTOM_LLM_PROVIDER: Final = "transcribe"


@lru_cache(maxsize=1)
def transcribe_supported_operations() -> frozenset[str]:
    """
    Operation names of the Amazon Transcribe JSON 1.1 API, read from the botocore
    service model so the allowlist tracks the installed SDK instead of a hand-typed copy.
    """
    from botocore.session import get_session

    return frozenset(get_session().get_service_model("transcribe").operation_names)


class TranscribePassthroughLoggingHandler:
    @staticmethod
    def _operation_from_response(httpx_response: httpx.Response) -> str:
        target: Final = httpx_response.request.headers.get("x-amz-target", "")
        return target.split(".")[-1]

    @staticmethod
    def transcribe_passthrough_handler(
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
        Records model and provider for an Amazon Transcribe control-plane call. Transcribe
        bills per second of audio once a job finishes, which no request or response on this
        path carries, so response_cost is recorded as 0.0 rather than estimated.
        """
        try:
            operation: Final = TranscribePassthroughLoggingHandler._operation_from_response(httpx_response)
            model_name: Final = f"{TRANSCRIBE_CUSTOM_LLM_PROVIDER}/{operation}"

            updated_kwargs: Final = {  # mutable-ok: the logging pipeline requires a plain kwargs dict
                **kwargs,
                "model": model_name,
                "custom_llm_provider": TRANSCRIBE_CUSTOM_LLM_PROVIDER,
                "response_cost": 0.0,
            }
            logging_obj.model_call_details.update(
                model=model_name,
                custom_llm_provider=TRANSCRIBE_CUSTOM_LLM_PROVIDER,
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
            verbose_proxy_logger.exception("Error in Amazon Transcribe passthrough logging handler: %s", e)
            fallback_payload: Final[PassThroughEndpointLoggingTypedDict] = {
                "result": StandardPassThroughResponseObject(response=result),
                "kwargs": kwargs,
            }
            return fallback_payload
        return handler_payload
