import datetime
from typing import Any, Optional, Union

from litellm.litellm_core_utils.core_helpers import process_response_headers
from litellm.litellm_core_utils.llm_response_utils.get_api_base import get_api_base
from litellm.litellm_core_utils.logging_utils import LiteLLMLoggingObject
from litellm.types.utils import (
    EmbeddingResponse,
    HiddenParams,
    ModelResponse,
    TranscriptionResponse,
)


class ResponseMetadata:
    """
    Handles setting and managing `_hidden_params`, `response_time_ms`, and `litellm_overhead_time_ms` for LiteLLM responses
    """

    def __init__(self, result: Any):
        self.result = result
        self._hidden_params: Union[HiddenParams, dict] = (
            getattr(result, "_hidden_params", {}) or {}
        )

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

        ## ADD OTHER HIDDEN PARAMS
        model_info = kwargs.get("model_info", {}) or {}
        model_id = model_info.get("id", None)
        new_params = {
            "litellm_call_id": getattr(logging_obj, "litellm_call_id", None),
            "api_base": get_api_base(model=model or "", optional_params=kwargs),
            "model_id": model_id,
            "response_cost": logging_obj._response_cost_calculator(
                result=self.result, litellm_model_name=model, router_model_id=model_id
            ),
            "additional_headers": process_response_headers(
                self._get_value_from_hidden_params("additional_headers") or {}
            ),
            "litellm_model_name": model,
        }
        self._update_hidden_params(new_params)

    def _update_hidden_params(self, new_params: dict) -> None:
        """
        Update hidden params - handles when self._hidden_params is a dict or HiddenParams object
        """
        # Handle both dict and HiddenParams cases
        if isinstance(self._hidden_params, dict):
            self._hidden_params.update(new_params)
        elif isinstance(self._hidden_params, HiddenParams):
            # For HiddenParams object, set attributes individually
            for key, value in new_params.items():
                setattr(self._hidden_params, key, value)

    def _get_value_from_hidden_params(self, key: str) -> Optional[Any]:
        """Get value from hidden params - handles when self._hidden_params is a dict or HiddenParams object"""
        if isinstance(self._hidden_params, dict):
            return self._hidden_params.get(key, None)
        elif isinstance(self._hidden_params, HiddenParams):
            return getattr(self._hidden_params, key, None)

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
            self._update_hidden_params(
                {
                    "litellm_overhead_time_ms": overhead_ms,
                    "_response_ms": total_response_time_ms,
                }
            )

    def apply(self) -> None:
        """Apply metadata to the response object"""
        if hasattr(self.result, "_hidden_params"):
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
