import datetime
from typing import Any, Optional

from litellm.litellm_core_utils.core_helpers import process_response_headers
from litellm.litellm_core_utils.llm_response_utils.get_api_base import get_api_base
from litellm.litellm_core_utils.logging_utils import LiteLLMLoggingObject
from litellm.types.utils import EmbeddingResponse, ModelResponse, TranscriptionResponse


class ResponseMetadata:
    """
    Handles setting and managing `_hidden_params`, `response_time_ms`, and `litellm_overhead_time_ms` for LiteLLM responses
    """

    def __init__(self, result: Any):
        self.result = result
        self._hidden_params = getattr(result, "_hidden_params", {}) or {}

    @property
    def supports_response_time(self) -> bool:
        """Check if response type supports timing metrics"""
        return (
            isinstance(self.result, ModelResponse)
            or isinstance(self.result, EmbeddingResponse)
            or isinstance(self.result, TranscriptionResponse)
        )

    def set_hidden_params(
        self, logging_obj: LiteLLMLoggingObject, model: Optional[str], kwargs: dict
    ) -> None:
        """Set hidden parameters on the response"""
        self._hidden_params.update(
            {
                "litellm_call_id": getattr(logging_obj, "litellm_call_id", None),
                "model_id": kwargs.get("model_info", {}).get("id", None),
                "api_base": get_api_base(model=model or "", optional_params=kwargs),
                "response_cost": logging_obj._response_cost_calculator(
                    result=self.result
                ),
                "additional_headers": process_response_headers(
                    self._hidden_params.get("additional_headers", {})
                ),
            }
        )

    def set_timing_metrics(
        self,
        start_time: datetime.datetime,
        end_time: datetime.datetime,
        logging_obj: LiteLLMLoggingObject,
    ) -> None:
        """Set response timing metrics"""
        total_response_time_ms = (end_time - start_time).total_seconds() * 1000

        # Set total response time if supported
        if self.supports_response_time:
            self.result._response_ms = total_response_time_ms

        # Calculate LiteLLM overhead
        llm_api_duration_ms = logging_obj.model_call_details.get("llm_api_duration_ms")
        if llm_api_duration_ms is not None:
            overhead_ms = round(total_response_time_ms - llm_api_duration_ms, 4)
            self._hidden_params.update(
                {
                    "litellm_overhead_time_ms": overhead_ms,
                    "_response_ms": total_response_time_ms,
                }
            )

    def apply(self) -> None:
        """Apply metadata to the response object"""
        self.result._hidden_params = self._hidden_params


def update_response_metadata(
    result: Any,
    logging_obj: LiteLLMLoggingObject,
    model: Optional[str],
    kwargs: dict,
    start_time: datetime.datetime,
    end_time: datetime.datetime,
) -> None:
    """
    Updates response metadata including hidden params and timing metrics
    """
    if result is None:
        return

    metadata = ResponseMetadata(result)
    metadata.set_hidden_params(logging_obj, model, kwargs)
    metadata.set_timing_metrics(start_time, end_time, logging_obj)
    metadata.apply()
