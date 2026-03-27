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


def get_response_hidden_params(response: Any) -> Union[HiddenParams, dict]:
    """
    Read LiteLLM internal fields from ModelResponse/Streaming responses or from
    dict-shaped provider responses (e.g. Anthropic ``/v1/messages`` JSON bodies).

    Dict responses store timing/cost under the ``_hidden_params`` key; the proxy
    strips that key before returning JSON to clients.
    """
    if response is None:
        return {}
    hp = getattr(response, "_hidden_params", None)
    if hp is not None:
        return hp
    if isinstance(response, dict):
        inner = response.get("_hidden_params")
        if isinstance(inner, dict):
            return inner
    return {}


def hidden_params_to_plain_dict(hp: Any) -> dict:
    """
    Normalize ``get_response_hidden_params`` output (``dict`` or ``HiddenParams``) to a
    plain ``dict`` for merging into ``metadata['hidden_params']``.
    """
    if not hp:
        return {}
    if isinstance(hp, dict):
        return dict(hp)
    if hasattr(hp, "model_dump"):
        return hp.model_dump(exclude_none=True)
    return {}


def strip_litellm_internal_keys_from_dict_response(response: Any) -> None:
    """Remove internal keys from dict API responses before JSON serialization."""
    if isinstance(response, dict):
        response.pop("_hidden_params", None)


def merge_hidden_params_with_logging_timings(
    response: Any,
    logging_obj: Any,
    *,
    end_time: Optional[datetime.datetime] = None,
) -> dict:
    """
    Merge response hidden params with timing derived from the LiteLLM logging object.

    Used by the proxy when ``response`` does not expose ``_hidden_params`` (e.g. Anthropic
    ``/v1/messages`` streaming async generators) or when timing keys are missing, so
    ``x-litellm-overhead-duration-ms`` and spend metadata can still be populated.

    Mirrors :meth:`ResponseMetadata.set_timing_metrics` for overhead/cache/callback fields.
    """
    hp_raw = get_response_hidden_params(response)
    if isinstance(hp_raw, dict):
        out: dict = dict(hp_raw)
    elif hp_raw is not None and hasattr(hp_raw, "model_dump"):
        out = hp_raw.model_dump(exclude_none=True)
    else:
        out = {}

    if out.get("litellm_overhead_time_ms") is not None:
        return out

    if logging_obj is None or not hasattr(logging_obj, "model_call_details"):
        return out

    mcd = logging_obj.model_call_details
    start_time = getattr(logging_obj, "start_time", None)
    _end = end_time or datetime.datetime.now()

    if start_time is None:
        return out

    total_ms = (_end - start_time).total_seconds() * 1000
    out.setdefault("_response_ms", total_ms)

    llm_api_duration_ms = mcd.get("llm_api_duration_ms")
    if llm_api_duration_ms is not None:
        overhead_ms = round(total_ms - float(llm_api_duration_ms), 4)
        out["litellm_overhead_time_ms"] = max(overhead_ms, 0.0)
    else:
        caching_details = getattr(logging_obj, "caching_details", None)
        if (
            caching_details is not None
            and caching_details.get("cache_hit") is True
            and (cache_duration_ms := caching_details.get("cache_duration_ms"))
            is not None
        ):
            out["litellm_overhead_time_ms"] = max(
                total_ms - float(cache_duration_ms), 0.0
            )

    callback_duration_ms = getattr(logging_obj, "callback_duration_ms", None)
    if callback_duration_ms is not None:
        out.setdefault(
            "callback_duration_ms", round(float(callback_duration_ms), 4)
        )

    if LITELLM_DETAILED_TIMING and llm_api_duration_ms is not None:
        detailed: dict = {
            "timing_llm_api_ms": round(float(llm_api_duration_ms), 4),
        }
        msg_copy_ms = getattr(logging_obj, "message_copy_duration_ms", None)
        if msg_copy_ms is not None:
            detailed["timing_message_copy_ms"] = round(float(msg_copy_ms), 4)
        api_call_start = mcd.get("api_call_start_time")
        if api_call_start is not None and start_time is not None:
            pre_ms = (api_call_start - start_time).total_seconds() * 1000
            detailed["timing_pre_processing_ms"] = round(pre_ms, 4)
            post_ms = total_ms - pre_ms - float(llm_api_duration_ms)
            detailed["timing_post_processing_ms"] = round(max(post_ms, 0), 4)
        out.update(detailed)

    return out


class ResponseMetadata:
    """
    Handles setting and managing `_hidden_params`, `response_time_ms`, and `litellm_overhead_time_ms` for LiteLLM responses
    """

    def __init__(self, result: Any):
        self.result = result
        self._hidden_params: Union[HiddenParams, dict] = get_response_hidden_params(
            result
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
        elif isinstance(self.result, dict):
            # Dict-shaped responses (e.g. Anthropic Messages API) have no
            # attribute slot; use a private key stripped before HTTP response.
            if isinstance(self._hidden_params, dict):
                self.result["_hidden_params"] = self._hidden_params
            elif isinstance(self._hidden_params, HiddenParams):
                self.result["_hidden_params"] = self._hidden_params.model_dump(
                    exclude_none=True
                )


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
