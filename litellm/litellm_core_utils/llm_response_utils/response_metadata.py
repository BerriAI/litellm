import datetime
from typing import Any, Optional, Union

from litellm.constants import LITELLM_DETAILED_TIMING
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

        #########################################################
        # 1. Add _response_ms total duration
        #########################################################
        self._update_hidden_params(
            {
                "_response_ms": total_response_time_ms,
            }
        )

        #########################################################
        # 2. Add LiteLLM overhead duration
        #########################################################
        llm_api_duration_ms = logging_obj.model_call_details.get("llm_api_duration_ms")
        if llm_api_duration_ms is not None:
            overhead_ms = round(total_response_time_ms - llm_api_duration_ms, 4)
            self._update_hidden_params(
                {
                    "litellm_overhead_time_ms": overhead_ms,
                }
            )

        #########################################################
        # 3. Add callback processing duration
        #########################################################
        callback_duration_ms = getattr(logging_obj, "callback_duration_ms", None)
        if callback_duration_ms is not None:
            self._update_hidden_params(
                {
                    "callback_duration_ms": round(callback_duration_ms, 4),
                }
            )

        #########################################################
        # 4. Add duration for reading from cache
        # In this case overhead from litellm is the difference between the cache read duration and the total response time
        #########################################################
        if (
            logging_obj.caching_details is not None
            and logging_obj.caching_details.get("cache_hit") is True
            and (
                cache_duration_ms := logging_obj.caching_details.get(
                    "cache_duration_ms"
                )
            )
            is not None
        ):
            overhead_ms = total_response_time_ms - cache_duration_ms
            self._update_hidden_params(
                {
                    "litellm_overhead_time_ms": overhead_ms,
                }
            )

        #########################################################
        # 5. Detailed per-phase timing (opt-in via env var)
        #########################################################
        if LITELLM_DETAILED_TIMING and llm_api_duration_ms is not None:
            detailed: dict = {
                "timing_llm_api_ms": round(llm_api_duration_ms, 4),
            }

            # message copy time from Logging.__init__()
            msg_copy_ms = getattr(logging_obj, "message_copy_duration_ms", None)
            if msg_copy_ms is not None:
                detailed["timing_message_copy_ms"] = round(msg_copy_ms, 4)

            # pre-processing = time from request start to LLM API call start
            api_call_start = logging_obj.model_call_details.get("api_call_start_time")
            if api_call_start is not None and start_time is not None:
                pre_ms = (api_call_start - start_time).total_seconds() * 1000
                detailed["timing_pre_processing_ms"] = round(pre_ms, 4)

                # post-processing = total - pre - llm_api
                post_ms = total_response_time_ms - pre_ms - llm_api_duration_ms
                detailed["timing_post_processing_ms"] = round(max(post_ms, 0), 4)

            self._update_hidden_params(detailed)

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
    Updates response metadata, adds the following:
        - response._hidden_params
        - response._hidden_params["litellm_overhead_time_ms"]
        - response.response_time_ms
    """
    if result is None:
        return

    metadata = ResponseMetadata(result)
    metadata.set_hidden_params(logging_obj, model, kwargs)
    metadata.set_timing_metrics(start_time, end_time, logging_obj)
    metadata.apply()
