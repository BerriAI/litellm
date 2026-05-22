"""
Failure payload enricher.

Ensures spend log rows for failed requests carry the same identification,
upstream-attribution, and timing fields as successful rows by reading
directly from the Logging object's ``model_call_details``.

Without this, streaming-cancel rows in particular end up with empty
``request_tags``, ``model_group``, ``model_id``, ``api_base``,
``requester_ip_address``, and zero ``request_duration_ms``, which makes
failure debugging very hard.

The enricher is idempotent: it only fills fields that ``request_data`` does
not already have a real value for. Any failure inside the enricher is logged
and swallowed; enrichment must never block writing of the spend log row.
"""
from datetime import datetime
from typing import Any, Optional

from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.litellm_logging import (
    get_standard_logging_object_payload,
)

# Top-level keys read from ``logging_obj.model_call_details`` and promoted
# onto ``request_data`` so ``get_logging_payload`` and downstream readers
# see them on the failure path.
_TOP_LEVEL_FIELDS = (
    "model",
    "custom_llm_provider",
    "call_type",
    "stream",
    "completion_start_time",
    "litellm_call_id",
    "litellm_trace_id",
    "response_cost",
    "cache_hit",
    "cache_key",
)

# Keys read from ``logging_obj.model_call_details["litellm_params"]`` and
# promoted onto ``request_data["litellm_params"]``. ``api_base`` is the URL
# of the upstream deployment (e.g. the specific H200 vLLM hit) and is the
# single most important attribution field on a failure row.
_LITELLM_PARAMS_FIELDS = (
    "api_base",
    "model",
    "custom_llm_provider",
    "proxy_server_request",
)

# Keys read from ``logging_obj.model_call_details["litellm_params"]["metadata"]``
# and promoted onto ``request_data["litellm_params"]["metadata"]``. These drive
# the identification columns rendered on the spend logs UI: model_group,
# model_id (via model_info.id), requester_ip_address, and request_tags
# (rebuilt downstream from metadata.tags + proxy_server_request headers).
_METADATA_FIELDS = (
    "model_group",
    "model_info",
    "deployment",
    "requester_ip_address",
    "user_agent",
    "tags",
    "user_api_key",
    "user_api_key_alias",
    "user_api_key_hash",
    "user_api_key_user_id",
    "user_api_key_team_id",
    "user_api_key_org_id",
    "user_api_key_end_user_id",
    "headers",
    "endpoint",
)


def _promote_if_empty(target: dict, key: str, value: Any) -> None:
    """Set ``target[key] = value`` only if ``value`` is non-empty and ``target``
    does not already carry a real value at that key."""
    if value in (None, "", [], {}):
        return
    cur = target.get(key)
    if cur in (None, "", [], {}):
        target[key] = value


def enrich_failure_request_data(
    request_data: dict,
    litellm_logging_obj: Any,
    original_exception: Exception,
    end_time: Optional[datetime] = None,
) -> dict:
    """Mutate ``request_data`` in place so the spend log payload constructor
    sees the same identification fields it would see on a success.

    ``litellm_logging_obj.model_call_details`` is the authoritative source.
    The enricher only fills fields that ``request_data`` does not already
    have a real value for, so it is safe to call on any failure path.
    """
    if litellm_logging_obj is None:
        return request_data

    try:
        mcd = getattr(litellm_logging_obj, "model_call_details", {}) or {}
        lp = mcd.get("litellm_params") or {}
        md = lp.get("metadata") or {}

        for key in _TOP_LEVEL_FIELDS:
            _promote_if_empty(request_data, key, mcd.get(key))

        request_data.setdefault("litellm_params", {})
        rp_lp = request_data["litellm_params"]
        for key in _LITELLM_PARAMS_FIELDS:
            _promote_if_empty(rp_lp, key, lp.get(key))

        # Mirror metadata fields onto BOTH ``request_data["metadata"]`` (the
        # proxy-level top-level metadata) AND ``request_data["litellm_params"]
        # ["metadata"]``. The failure hook in ``proxy_track_cost_callback.py``
        # rebuilds ``litellm_params.metadata`` from ``request_data.metadata``
        # plus a small set of failure-context fields, replacing whatever was
        # there. Writing to the top-level metadata ensures these fields survive
        # that rebuild; writing to litellm_params.metadata as well keeps the
        # tag-preservation branch in the hook seeing the right data.
        request_data.setdefault("metadata", {})
        rp_top_md = request_data["metadata"]
        rp_md = rp_lp.setdefault("metadata", {})
        for key in _METADATA_FIELDS:
            value = md.get(key)
            _promote_if_empty(rp_top_md, key, value)
            _promote_if_empty(rp_md, key, value)

        _promote_if_empty(
            request_data,
            "litellm_trace_id",
            getattr(litellm_logging_obj, "litellm_trace_id", None),
        )

        # Build a best-effort StandardLoggingPayload if neither
        # ``request_data`` nor ``model_call_details`` carries one. This is what
        # unlocks request_tags, model_group, model_id, api_base, and the rich
        # metadata JSON column on the spend log row.
        if (
            request_data.get("standard_logging_object") is None
            and mcd.get("standard_logging_object") is None
        ):
            _maybe_build_standard_logging_object(
                request_data=request_data,
                litellm_logging_obj=litellm_logging_obj,
                mcd=mcd,
                original_exception=original_exception,
                end_time=end_time,
            )
    except Exception:
        verbose_proxy_logger.exception(
            "enrich_failure_request_data: enrichment failed; "
            "continuing with un-enriched payload"
        )
    return request_data


def _maybe_build_standard_logging_object(
    request_data: dict,
    litellm_logging_obj: Any,
    mcd: dict,
    original_exception: Exception,
    end_time: Optional[datetime],
) -> None:
    """Try to construct a ``StandardLoggingPayload`` from ``model_call_details``.

    Best-effort: any failure here is swallowed so we never make the failure
    row worse than it would have been without enrichment.
    """
    try:
        start_time = (
            getattr(litellm_logging_obj, "start_time", None)
            or mcd.get("start_time")
            or datetime.now()
        )
        _end_time = end_time or datetime.now()

        slp = get_standard_logging_object_payload(
            kwargs=mcd,
            init_response_obj=None,
            start_time=start_time,
            end_time=_end_time,
            logging_obj=litellm_logging_obj,
            status="failure",
            error_str=str(original_exception),
            original_exception=original_exception,
        )
        if slp:
            request_data["standard_logging_object"] = slp
    except Exception:
        verbose_proxy_logger.exception(
            "enrich_failure_request_data: SLO build failed; "
            "continuing without standard_logging_object"
        )


def resolve_failure_start_time(
    litellm_logging_obj: Any, default: Optional[datetime] = None
) -> datetime:
    """Pick the most accurate ``start_time`` for a failed request.

    Order of preference:

    1. ``litellm_logging_obj.start_time`` (attribute set at logging init)
    2. ``litellm_logging_obj.model_call_details["start_time"]``
    3. ``default`` argument
    4. ``datetime.now()``
    """
    if litellm_logging_obj is not None:
        obj_start = getattr(litellm_logging_obj, "start_time", None)
        if obj_start is not None:
            return obj_start
        mcd = getattr(litellm_logging_obj, "model_call_details", {}) or {}
        mcd_start = mcd.get("start_time")
        if mcd_start is not None:
            return mcd_start
    if default is not None:
        return default
    return datetime.now()
