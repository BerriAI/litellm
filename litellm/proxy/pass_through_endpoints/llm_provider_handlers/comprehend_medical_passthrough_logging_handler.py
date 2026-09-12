import math
from collections.abc import Mapping
from datetime import datetime
from types import MappingProxyType
from typing import Final

import httpx

from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.litellm_core_utils.litellm_logging import (
    get_standard_logging_object_payload,
)
from litellm.proxy._types import PassThroughEndpointLoggingTypedDict
from litellm.types.utils import StandardPassThroughResponseObject

COMPREHEND_MEDICAL_CHARS_PER_UNIT: Final = 100
COMPREHEND_MEDICAL_COST_PER_UNIT_USD: Final[Mapping[str, float]] = MappingProxyType(
    {
        "DetectEntitiesV2": 0.01,
        "DetectPHI": 0.0014,
        "InferICD10CM": 0.0005,
        "InferRxNorm": 0.00025,
        "InferSNOMEDCT": 0.0075,
    }
)
COMPREHEND_MEDICAL_SUPPORTED_OPERATIONS: Final = frozenset(COMPREHEND_MEDICAL_COST_PER_UNIT_USD)


class ComprehendMedicalPassthroughLoggingHandler:
    @staticmethod
    def _operation_from_response(httpx_response: httpx.Response) -> str:
        target: Final = httpx_response.request.headers.get("x-amz-target", "")
        return target.split(".")[-1]

    @staticmethod
    def get_cost_for_operation(operation: str, text: str) -> float:
        cost_per_unit: Final = COMPREHEND_MEDICAL_COST_PER_UNIT_USD.get(operation)
        if cost_per_unit is None:
            return 0.0
        units: Final = max(1, math.ceil(len(text) / COMPREHEND_MEDICAL_CHARS_PER_UNIT))
        return units * cost_per_unit

    @staticmethod
    def comprehend_medical_passthrough_handler(
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
        Prices a Comprehend Medical sync operation from the request text length
        (billed per started 100-character unit, 1-unit minimum) and records
        model, provider, and cost on the logging payload.
        """
        try:
            operation: Final = ComprehendMedicalPassthroughLoggingHandler._operation_from_response(httpx_response)
            text: Final = request_body.get("Text")
            response_cost: Final = ComprehendMedicalPassthroughLoggingHandler.get_cost_for_operation(
                operation=operation,
                text=text if isinstance(text, str) else "",
            )
            model_name: Final = f"comprehendmedical/{operation}"

            updated_kwargs: Final = {  # mutable-ok: the logging pipeline requires a plain kwargs dict
                **kwargs,
                "model": model_name,
                "custom_llm_provider": "comprehendmedical",
                "response_cost": response_cost,
            }
            logging_obj.model_call_details.update(
                model=model_name,
                custom_llm_provider="comprehendmedical",
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
        except Exception as e:
            verbose_proxy_logger.exception("Error in Comprehend Medical passthrough logging handler: %s", e)
            fallback_payload: Final[PassThroughEndpointLoggingTypedDict] = {
                "result": StandardPassThroughResponseObject(response=result),
                "kwargs": kwargs,
            }
            return fallback_payload
        return handler_payload
