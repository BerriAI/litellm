"""
`clickhouse` logging callback: one `spend_logs` row per LiteLLM request.

Agent LLM spans join to these rows on `otel_traces.LiteLLMRequestId = spend_logs.response_id`,
so `response_id` is always the raw provider response id (cache-hit suffix stripped).
"""

import json
import re
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, Final

import litellm
from litellm._logging import verbose_logger
from litellm.integrations.clickhouse.clickhouse_batch_logger import ClickHouseBatchLogger
from litellm.integrations.clickhouse.context import is_lens_analysis
from litellm.integrations.clickhouse.schema import SPEND_LOGS_TABLE
from litellm.tracing.types import SpendLogRecord
from litellm.types.utils import StandardLoggingPayload

# litellm_logging.py rewrites cache-hit ids as f"{id}_cache_hit{time.time()}"
MILLISECONDS_PER_SECOND: Final = 1000
_CACHE_HIT_SUFFIX: Final = re.compile(r"_cache_hit[0-9.]*$")
# W3C trace context: version-traceid-parentid-flags
_TRACEPARENT: Final = re.compile(r"^[0-9a-f]{2}-([0-9a-f]{32})-([0-9a-f]{16})-[0-9a-f]{2}$")
_INVALID_TRACE_ID: Final = "0" * 32
_INVALID_SPAN_ID: Final = "0" * 16
TRACE_INGEST_ROUTE: Final = "/v1/traces"


def strip_cache_hit_suffix(request_id: str) -> str:
    return _CACHE_HIT_SUFFIX.sub("", request_id)


def parse_traceparent(value: object) -> tuple[str, str]:
    """(trace_id, span_id) from a W3C `traceparent` header, or ("", "") if absent/invalid."""
    if not isinstance(value, str):
        return "", ""
    match = _TRACEPARENT.match(value.strip().lower())
    if match is None or match.group(1) == _INVALID_TRACE_ID or match.group(2) == _INVALID_SPAN_ID:
        return "", ""
    return match.group(1), match.group(2)


def _to_ms(seconds: object) -> int | None:
    return int(float(seconds) * MILLISECONDS_PER_SECOND) if isinstance(seconds, (int, float)) else None


def _int(value: object) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _json(value: object) -> str:
    if value is None or value == "":
        return ""
    return value if isinstance(value, str) else json.dumps(value, default=str)


def _json_mapping(value: Mapping[str, Any]) -> str:
    return _json(dict(value))


def _find_traceparent(metadata: Mapping[str, Any], kwargs: Mapping[str, Any]) -> tuple[str, str]:
    custom_headers = metadata.get("requester_custom_headers") or MappingProxyType({})
    proxy_request = (kwargs.get("litellm_params") or MappingProxyType({})).get(
        "proxy_server_request"
    ) or MappingProxyType({})
    request_headers = proxy_request.get("headers") or MappingProxyType({})
    for headers in (custom_headers, request_headers):
        for name, value in headers.items():
            if str(name).lower() == "traceparent":
                return parse_traceparent(value)
    return "", ""


def _cache_tokens(usage: Mapping[str, Any]) -> tuple[int, int]:
    """(cache_read, cache_write) from a Usage dict: OpenAI prompt_tokens_details first, Anthropic fields as fallback."""
    details = usage.get("prompt_tokens_details") or MappingProxyType({})
    cache_read = _int(details.get("cached_tokens")) or _int(usage.get("cache_read_input_tokens"))
    cache_write = (
        _int(details.get("cache_write_tokens"))
        or _int(details.get("cache_creation_tokens"))
        or _int(usage.get("cache_creation_input_tokens"))
    )
    return cache_read, cache_write


def _request_tags(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(tag) for tag in value]


def _session_id(payload: StandardLoggingPayload, kwargs: Mapping[str, Any]) -> str:
    """Mirrors proxy `_get_session_id_for_spend_log`: explicit session id, else the payload trace id."""
    request_metadata = (kwargs.get("litellm_params") or MappingProxyType({})).get("metadata") or MappingProxyType({})
    return str(payload.get("session_id") or request_metadata.get("session_id") or payload.get("trace_id") or "")


def _is_trace_ingest(payload: StandardLoggingPayload) -> bool:
    """OTLP exports to POST /v1/traces are not LLM requests; don't write them as spend rows."""
    return str(payload.get("call_type") or "").startswith(TRACE_INGEST_ROUTE)


def spend_log_row_from_payload(payload: StandardLoggingPayload, kwargs: Mapping[str, Any]) -> SpendLogRecord:
    metadata: Mapping[str, Any] = payload.get("metadata") or MappingProxyType({})
    hidden_params: Mapping[str, Any] = payload.get("hidden_params") or MappingProxyType({})
    usage: Mapping[str, Any] = metadata.get("usage_object") or hidden_params.get("usage_object") or MappingProxyType({})
    cache_read_tokens, cache_write_tokens = _cache_tokens(usage)
    trace_id, span_id = _find_traceparent(metadata, kwargs)
    request_id = str(payload.get("id") or "")
    redact = litellm.turn_off_message_logging is True
    completion_start_ms = _to_ms(payload.get("completionStartTime"))
    return SpendLogRecord(
        request_id=request_id,
        response_id=strip_cache_hit_suffix(request_id),
        call_type=payload.get("call_type") or "",
        api_key=metadata.get("user_api_key_hash") or "",
        key_alias=metadata.get("user_api_key_alias") or "",
        team_id=metadata.get("user_api_key_team_id") or metadata.get("team_id") or "",
        team_alias=metadata.get("user_api_key_team_alias") or metadata.get("team_alias") or "",
        organization_id=metadata.get("user_api_key_org_id") or "",
        user=metadata.get("user_api_key_user_id") or "",
        end_user=payload.get("end_user") or metadata.get("user_api_key_end_user_id") or "",
        model=payload.get("model") or "",
        model_group=payload.get("model_group") or "",
        model_id=payload.get("model_id") or "",
        custom_llm_provider=payload.get("custom_llm_provider") or "",
        api_base=payload.get("api_base") or "",
        spend=float(payload.get("response_cost") or 0.0),
        prompt_tokens=_int(payload.get("prompt_tokens")),
        completion_tokens=_int(payload.get("completion_tokens")),
        total_tokens=_int(payload.get("total_tokens")),
        cache_read_tokens=cache_read_tokens,
        cache_write_tokens=cache_write_tokens,
        start_time=_to_ms(payload.get("startTime")) or 0,
        end_time=_to_ms(payload.get("endTime")) or 0,
        completion_start_time=completion_start_ms or None,
        status=payload.get("status") or "",
        error_str=payload.get("error_str") or "",
        cache_hit=payload.get("cache_hit") is True,
        session_id=_session_id(payload, kwargs),
        trace_id=trace_id,
        span_id=span_id,
        request_tags=_request_tags(payload.get("request_tags")),
        metadata=_json_mapping(MappingProxyType({**metadata, "litellm_lens_internal": is_lens_analysis()})),
        messages="" if redact else _json(payload.get("messages")),
        response="" if redact else _json(payload.get("response")),
    )


class ClickHouseSpendLogger(ClickHouseBatchLogger):
    table = SPEND_LOGS_TABLE

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time) -> None:
        self._log(kwargs)

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time) -> None:
        self._log(kwargs)

    def _log(self, kwargs: Mapping[str, Any]) -> None:
        try:
            payload = kwargs.get("standard_logging_object")
            if payload is None or _is_trace_ingest(payload):
                return
            row: Final = spend_log_row_from_payload(payload, kwargs)
            self.enqueue([dict(row)])
        except Exception as e:
            verbose_logger.exception("ClickHouseSpendLogger: failed to log request: %s", e)
