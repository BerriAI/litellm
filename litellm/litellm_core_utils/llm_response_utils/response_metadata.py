import datetime
from collections.abc import Mapping
from typing import Any, Final

import httpx

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


def response_timing_metrics(
    start_time: datetime.datetime,
    end_time: datetime.datetime,
    logging_obj: LiteLLMLoggingObject,
    include_overhead: bool = True,
) -> Mapping[str, float]:
    """``_response_ms`` for the whole call, plus ``litellm_overhead_time_ms`` when it can be derived.

    On a cache hit the overhead is the total minus the cache read; otherwise it is the total minus
    the provider call (``llm_api_duration_ms``). It is omitted when neither duration was recorded,
    and when ``include_overhead`` is False because the two durations cover different windows.
    """
    total_response_time_ms: Final = (end_time - start_time).total_seconds() * 1000
    if not include_overhead:
        return {"_response_ms": total_response_time_ms}  # mutable-ok: read-only timing result
    caching_details: Final = logging_obj.caching_details
    cache_duration_ms: Final = (
        caching_details.get("cache_duration_ms")
        if caching_details is not None and caching_details.get("cache_hit") is True
        else None
    )
    llm_api_duration_ms: Final = logging_obj.model_call_details.get("llm_api_duration_ms")
    if cache_duration_ms is not None:
        overhead_ms: float | None = total_response_time_ms - cache_duration_ms
    elif llm_api_duration_ms is not None:
        overhead_ms = round(total_response_time_ms - llm_api_duration_ms, 4)
    else:
        overhead_ms = None
    if overhead_ms is None:
        return {"_response_ms": total_response_time_ms}
    return {"_response_ms": total_response_time_ms, "litellm_overhead_time_ms": overhead_ms}


class ResponseMetadata:
    """
    Handles setting and managing `_hidden_params`, `response_time_ms`, and `litellm_overhead_time_ms` for LiteLLM responses
    """

    def __init__(self, result: Any):
        self.result = result
        self._hidden_params: HiddenParams | dict = getattr(result, "_hidden_params", {}) or {}

    @property
    def supports_response_time(self) -> bool:
        """Check if response type supports timing metrics"""
        return isinstance(self.result, (ModelResponse, EmbeddingResponse, TranscriptionResponse))

    def set_hidden_params(self, logging_obj: LiteLLMLoggingObject, model: str | None, kwargs: dict) -> None:
        """Set hidden parameters on the response"""

        ## ADD OTHER HIDDEN PARAMS
        model_info: Final = kwargs.get("model_info", {}) or {}
        model_id: Final = model_info.get("id", None)
        new_params: Final = {
            "litellm_call_id": getattr(logging_obj, "litellm_call_id", None),
            "api_base": get_api_base(model=model or "", optional_params=kwargs),
            "model_id": model_id,
            "response_cost": logging_obj._response_cost_calculator(
                result=self.result, litellm_model_name=model, router_model_id=model_id
            ),
            "additional_headers": process_response_headers(
                self._get_additional_headers_from_hidden_params() or {},
                preserve_litellm_internal_headers=True,
            ),
            "litellm_model_name": model,
        }
        self._update_hidden_params(new_params)

    def _update_hidden_params(self, new_params: Mapping[str, object]) -> None:
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

    def _get_additional_headers_from_hidden_params(self) -> httpx.Headers | dict[str, str] | None:
        """Get `additional_headers` from hidden params - handles when self._hidden_params is a dict or HiddenParams object"""
        if isinstance(self._hidden_params, dict):
            return self._hidden_params.get("additional_headers", None)
        elif isinstance(self._hidden_params, HiddenParams):
            return getattr(self._hidden_params, "additional_headers", None)

    def set_timing_metrics(
        self,
        start_time: datetime.datetime,
        end_time: datetime.datetime,
        logging_obj: LiteLLMLoggingObject,
        include_overhead: bool = True,
    ) -> None:
        """Set response timing metrics"""
        timing_metrics: Final = response_timing_metrics(start_time, end_time, logging_obj, include_overhead)
        total_response_time_ms: Final = timing_metrics["_response_ms"]

        # Set total response time if supported
        if self.supports_response_time:
            self.result._response_ms = total_response_time_ms

        #########################################################
        # 1. Add _response_ms total duration and the LiteLLM overhead within it
        #    (total minus the cache read on a cache hit, else total minus the provider call)
        #########################################################
        self._update_hidden_params(timing_metrics)

        #########################################################
        # 2. Add callback processing duration
        #########################################################
        callback_duration_ms: Final[float | None] = getattr(logging_obj, "callback_duration_ms", None)
        if callback_duration_ms is not None:
            self._update_hidden_params(
                {
                    "callback_duration_ms": round(callback_duration_ms, 4),
                }
            )

        #########################################################
        # 3. Detailed per-phase timing (opt-in via env var)
        #########################################################
        llm_api_duration_ms: Final = logging_obj.model_call_details.get("llm_api_duration_ms")
        if LITELLM_DETAILED_TIMING and llm_api_duration_ms is not None:
            detailed: Final[dict[str, float]] = {
                "timing_llm_api_ms": round(llm_api_duration_ms, 4),
            }

            # message copy time from Logging.__init__()
            msg_copy_ms: Final[float | None] = getattr(logging_obj, "message_copy_duration_ms", None)
            if msg_copy_ms is not None:
                detailed["timing_message_copy_ms"] = round(msg_copy_ms, 4)

            # pre-processing = time from request start to LLM API call start
            api_call_start: Final[datetime.datetime | None] = logging_obj.model_call_details.get("api_call_start_time")
            if api_call_start is not None and start_time is not None:
                pre_ms: Final = (api_call_start - start_time).total_seconds() * 1000
                detailed["timing_pre_processing_ms"] = round(pre_ms, 4)

                # post-processing = total - pre - llm_api
                post_ms: Final = total_response_time_ms - pre_ms - llm_api_duration_ms
                detailed["timing_post_processing_ms"] = round(max(post_ms, 0), 4)

            self._update_hidden_params(detailed)

    def apply(self) -> None:
        """Apply metadata to the response object"""
        if hasattr(self.result, "_hidden_params"):
            self.result._hidden_params = self._hidden_params


def update_response_metadata(
    result: Any,
    logging_obj: LiteLLMLoggingObject,
    model: str | None,
    kwargs: dict,
    start_time: datetime.datetime,
    end_time: datetime.datetime,
    include_overhead: bool = True,
) -> None:
    """
    Updates response metadata including hidden params and timing metrics
    Updates response metadata, adds the following:
        - response._hidden_params
        - response._hidden_params["litellm_overhead_time_ms"]
        - response.response_time_ms
    A result that cannot hold ``_hidden_params`` gets its timing on ``logging_obj`` instead.
    Callers whose ``end_time`` covers more than the recorded provider call (a stream read to
    completion) pass ``include_overhead=False``, since the overhead cannot be derived there.
    """
    if result is None:
        return
    if not hasattr(result, "_hidden_params"):
        # /v1/messages returns a plain dict and the Anthropic / Responses bridge stream wrappers
        # cannot hold ``_hidden_params``: keep only the timing on the logging object (no cost
        # recompute) so the proxy headers and the standard logging payload can still read it.
        logging_obj.set_response_timing_metrics(
            response_timing_metrics(start_time, end_time, logging_obj, include_overhead)
        )
        return

    metadata: Final = ResponseMetadata(result)
    metadata.set_hidden_params(logging_obj, model, kwargs)
    metadata.set_timing_metrics(start_time, end_time, logging_obj, include_overhead)
    metadata.apply()
