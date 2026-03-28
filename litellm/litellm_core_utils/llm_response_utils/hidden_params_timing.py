"""
Hidden-params helpers and proxy timing merge.

Kept separate from ``response_metadata`` so the proxy can import timing merge logic
without pulling in ``response_metadata`` → ``logging_utils`` import chains that
CodeQL flags as cyclic with ``litellm.proxy.common_request_processing``.
"""

import datetime
from typing import Any, Optional, Union

from litellm.constants import LITELLM_DETAILED_TIMING
from litellm.types.utils import HiddenParams


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
