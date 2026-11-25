# pyright: reportPrivateUsage=false
# The frozen ledger-off builder still reads the private helpers it shares with spend_tracking_utils.
"""Deprecated: the ledger-off SpendLogs row builder.

Frozen as it was before the ledger. With the ledger on, ``get_logging_payload`` builds every row in
Rust (``rust_spend_log_rows.spendlogs_payload_rust``), and nothing here runs. Delete this module
together with the ledger-off setting; do not add to it. ``record_goldens.py`` replays it to record the
Rust builder's goldens, which stay as recorded once it is gone.

Moved out of ``spend_tracking_utils`` unchanged. The helpers both builders share stay there.
"""

import json
from collections.abc import Mapping
from datetime import datetime, timezone
from functools import reduce
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    Any,  # noqa: TID251  # the legacy builder, moved out of spend_tracking_utils unchanged
    Final,
    Literal,
    cast,  # noqa: TID251  # the legacy builder, moved out of spend_tracking_utils unchanged
)

from pydantic import BaseModel, TypeAdapter

import litellm
from litellm.constants import (
    EMPTY_MAPPING,
    LITELLM_TRUNCATED_PAYLOAD_FIELD,
    LITELLM_TRUNCATION_DB_SAFEGUARD_NOTE,
    MAX_SPEND_LOG_MODEL_NAME_LENGTH,
    MCP_SPEND_LOG_MODEL_PREFIX,
    SESSION_ID_OMITTED_METADATA_KEY,
    UNKNOWN_MODEL_SPEND_LOG_MODEL,
)
from litellm.litellm_core_utils.batch_utils import BATCH_COST_REQUEST_ID_SUFFIX
from litellm.litellm_core_utils.classifier_logging import (
    CLASSIFIER_AUDIT_FIELDS,
    classifier_audit_fields,
    without_classifier_audit,
)
from litellm.litellm_core_utils.core_helpers import (
    as_str_mapping,
    claude_subscription_usage,
    get_litellm_metadata_from_kwargs,
    proxy_stamped_kubernetes_pod_routing,
    proxy_stamped_used_client_oauth_token,
    reconstruct_model_name,
)
from litellm.litellm_core_utils.db_truncation import TRUNCATION_MARKER
from litellm.litellm_core_utils.hidden_params import served_from_cache
from litellm.litellm_core_utils.internal_call_metadata import is_unbilled_non_inference_call
from litellm.litellm_core_utils.litellm_logging import StandardLoggingPayloadSetup
from litellm.litellm_core_utils.prompt_templates.mid_conversation_system import message_field, parts_of
from litellm.litellm_core_utils.ptu_pricing import azure_spillover
from litellm.litellm_core_utils.safe_json_dumps import safe_dumps, strip_null_bytes
from litellm.llms.anthropic.common_utils import resolve_used_client_oauth_token
from litellm.proxy._types import SpendLogsPayload, SpendLogsRouterMetadata
from litellm.proxy.route_llm_request import ProxyModelNotFoundError
from litellm.proxy.spend_tracking import spend_tracking_utils
from litellm.proxy.spend_tracking.spend_log_error_logger import spend_log_error
from litellm.proxy.spend_tracking.spend_tracking_utils import (
    _CUSTOMER_METADATA_ADAPTER,
    _OBJECT_LIST,
    _OBJECT_MAPPING,
    _REQUEST_BODY_ADAPTER,
    _REQUEST_METADATA_KEYS,
    _RESPONSES_CALL_TYPES,
    _SCRUBBED_ERROR_TEXT_FIELDS,
    _SPEND_LOGS_PRIVACY_METADATA_ADAPTER,
    _TOOL_OUTPUT_BLOCK_TYPES,
    _adapter_route_call_type,
    _block_type,
    _compaction_summary_for_db,
    _content_block_text,
    _error_information_without_ip_addresses,
    _extract_usage_for_ocr_call,
    _get_session_id_for_spend_log,
    _get_spend_logs_metadata,
    _is_configured_model_group,
    _is_non_secret_key_value,
    _masked_ip_text,
    _model_group_provider,
    _sanitize_request_body_for_spend_logs_payload,
    _snapshotless_cost_breakdown,
    _string_or_empty,
    _without_tokens,
    get_spend_logs_id,
    redact_logged_api_key,
    requester_ip_address_logging_disabled,
    sanitize_error_information_for_spend_logs,
    stores_latest_turn_only,
)
from litellm.responses.litellm_completion_transformation.compaction_items import (
    compaction_summary,
    summary_only_compaction_content,
)
from litellm.types.utils import (
    CallTypes,
    LiteLLMBatch,
    StandardLoggingMCPToolCall,
    StandardLoggingPayload,
    StandardLoggingPayloadErrorInformation,
    batch_chat_usage,
)
from litellm.utils import get_end_user_id_for_cost_tracking

if TYPE_CHECKING:
    from litellm.router import Router


def _get_router_metadata_for_spend_log(
    metadata: Mapping[str, object] | None,
    requested_model: str | None,
    selected_model: str | None,
    selected_provider: str | None,
    router_correlation_id: str | None,
) -> SpendLogsRouterMetadata | None:
    model_info: Final = metadata.get("model_info") if metadata is not None else None
    if not isinstance(model_info, Mapping) or model_info.get("internal_router_model") is not True:
        return None
    return SpendLogsRouterMetadata(
        requested_model=requested_model or None,
        selected_model=selected_model or None,
        selected_provider=selected_provider or None,
        router_correlation_id=router_correlation_id,
    )


def _sl_attribution_fallback(
    standard_logging_payload: StandardLoggingPayload | None,
    field: Literal["model_id", "model_group", "api_base", "custom_llm_provider"],
) -> str:
    if standard_logging_payload is None:
        return ""
    return _string_or_empty(standard_logging_payload.get(field))


def _looks_like_model_name(model: str) -> bool:
    candidate: Final = model.removeprefix(MCP_SPEND_LOG_MODEL_PREFIX)
    return len(candidate) <= MAX_SPEND_LOG_MODEL_NAME_LENGTH and not any(char.isspace() for char in candidate)


def _raw_model_spellings(raw_model: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys((raw_model, repr(raw_model)[1:-1], json.dumps(raw_model)[1:-1])))


def _overlap_at_end(text: str, spelling: str) -> int:
    lengths: Final = range(min(len(text), len(spelling) - 1), 0, -1)
    return next((length for length in lengths if text.endswith(spelling[:length])), 0)


def _overlap_at_start(text: str, spelling: str) -> int:
    lengths: Final = range(min(len(text), len(spelling) - 1), 0, -1)
    return next((length for length in lengths if text.startswith(spelling[-length:])), 0)


def _scrub_raw_model_split_by_truncation(text: str, spellings: tuple[str, ...]) -> str:
    marker: Final = TRUNCATION_MARKER.search(text)
    if marker is None:
        return text
    head: Final = text[: marker.start()]
    tail: Final = text[marker.end() :]
    head_cut: Final = max(_overlap_at_end(head, spelling) for spelling in spellings)
    tail_cut: Final = max(_overlap_at_start(tail, spelling) for spelling in spellings)
    return "".join(
        (
            head[: len(head) - head_cut],
            UNKNOWN_MODEL_SPEND_LOG_MODEL if head_cut else "",
            marker.group(0),
            UNKNOWN_MODEL_SPEND_LOG_MODEL if tail_cut else "",
            tail[tail_cut:],
        )
    )


def _scrub_raw_model_from_error_text(text: str, spellings: tuple[str, ...]) -> str:
    whole_occurrences_scrubbed: Final = reduce(
        lambda scrubbed, spelling: scrubbed.replace(spelling, UNKNOWN_MODEL_SPEND_LOG_MODEL), spellings, text
    )
    return _scrub_raw_model_split_by_truncation(whole_occurrences_scrubbed, spellings)


def _scrub_raw_model_from_error_information(
    error_information: StandardLoggingPayloadErrorInformation | None, raw_model: str
) -> StandardLoggingPayloadErrorInformation | None:
    if error_information is None or not raw_model:
        return error_information
    spellings: Final = _raw_model_spellings(raw_model)
    return cast(
        StandardLoggingPayloadErrorInformation,
        {
            key: _scrub_raw_model_from_error_text(value, spellings)
            if key in _SCRUBBED_ERROR_TEXT_FIELDS and isinstance(value, str)
            else value
            for key, value in error_information.items()
        },
    )


def _omits_session_id_when_missing(metadata: Mapping[str, object] | None) -> bool:
    """The pre-call stamp pins `omit` on for the requests that carry it, so a config reload between pre-call and spend
    logging cannot fabricate a session. `apply_missing_session_id_policy` drops any client-supplied copy of the key
    from both metadata buckets before stamping, which the merge of `litellm_metadata` into `metadata` makes
    necessary, so a caller cannot forge it. Requests that never reach the pre-call helper, router-model
    passthrough among them, carry no stamp, so they fall back to the configured policy and `omit` still covers their
    spend logs."""
    if metadata is not None and metadata.get(SESSION_ID_OMITTED_METADATA_KEY):
        return True

    from litellm.proxy.proxy_server import general_settings

    return general_settings.get("missing_session_id") == "omit"


_BATCH_TRACE_CALL_TYPES: Final = frozenset(
    {
        CallTypes.create_batch.value,
        CallTypes.acreate_batch.value,
        CallTypes.retrieve_batch.value,
        CallTypes.aretrieve_batch.value,
    }
)


def _get_batch_trace_session_id(call_type: str | None, request_id: str | None) -> str | None:
    """A batch's create row and its poller-written cost row both derive their request id
    from the same batch id (the cost row appends BATCH_COST_REQUEST_ID_SUFFIX), so using
    that id as the session groups the batch lifecycle into one trace on the logs UI. The
    poller builds its own logging context, so per-request trace ids can never link them.
    """
    if call_type not in _BATCH_TRACE_CALL_TYPES or not request_id:
        return None
    return request_id.removesuffix(BATCH_COST_REQUEST_ID_SUFFIX)


def _get_request_duration_ms(start_time: datetime, end_time: datetime) -> int | None:
    """Compute request duration in milliseconds from start and end times."""
    try:
        return int((end_time - start_time).total_seconds() * 1000)
    except Exception:
        return None


def _ensure_datetime_utc(timestamp: datetime) -> datetime:
    """Helper to ensure datetime is in UTC"""
    timestamp = timestamp.astimezone(timezone.utc)
    return timestamp


def _get_messages_for_spend_logs_payload(
    standard_logging_payload: StandardLoggingPayload | None,
    metadata: dict | None = None,
) -> str:
    if spend_tracking_utils.should_store_prompts_and_responses_in_spend_logs(
        _SPEND_LOGS_PRIVACY_METADATA_ADAPTER.validate_python(metadata)
    ):
        if standard_logging_payload is not None:
            call_type: Final = standard_logging_payload.get("call_type", "")
            if call_type == "_arealtime":
                messages: Final = standard_logging_payload.get("messages")
                if messages is not None:
                    try:
                        return _without_tokens(safe_dumps(messages, non_finite_as_null=True))
                    except Exception:
                        return "{}"
    return "{}"


_STATEFUL_REQUEST_KEYS: Final = frozenset({"previous_response_id", "conversation"})


_LATEST_TURN_KEPT_KEYS: Final = frozenset({"model"})


_TURNS_KEYS: Final = ("messages", "input")


def _is_tool_output_block(block: object) -> bool:
    return _block_type(block) in _TOOL_OUTPUT_BLOCK_TYPES


def _user_content_text(content: object) -> str | None:
    if isinstance(content, str):
        return content or None
    blocks: Final = parts_of(content)
    if any(_is_tool_output_block(block) for block in blocks):
        return None
    texts: Final = tuple(text for text in map(_content_block_text, blocks) if text)
    return "\n".join(texts) or None


def _latest_user_message_text(turns: object) -> str | None:
    if isinstance(turns, str):
        return turns or None
    latest: Final = next(reversed(parts_of(turns)), None)
    if message_field(latest, "role") != "user":
        return None
    return _user_content_text(message_field(latest, "content"))


def _continues_stored_conversation(request_body: Mapping[str, object], call_type: object) -> bool:
    """Only `/v1/responses` resumes a stored conversation, so the same keys on a chat body do not exempt it."""
    return call_type in _RESPONSES_CALL_TYPES and any(request_body.get(key) for key in _STATEFUL_REQUEST_KEYS)


def _replayed_by_responses_bridge(standard_payload: Mapping[str, object] | None) -> bool:
    """A `/v1/responses` turn served through the chat completions bridge logs a chat-shaped response.

    The bridge rebuilds a `previous_response_id` chain from these rows, so they keep their full request.
    """
    if standard_payload is None or standard_payload.get("call_type") not in _RESPONSES_CALL_TYPES:
        return False
    response: Final = standard_payload.get("response")
    return isinstance(response, Mapping) and "choices" in response


def _trimmed_turn(request_body: Mapping[str, object], call_type: object) -> Mapping[str, object]:
    if _continues_stored_conversation(request_body, call_type):
        return request_body
    kept: Final = {key: value for key, value in request_body.items() if key in _LATEST_TURN_KEPT_KEYS}
    turns_key: Final = next((key for key in _TURNS_KEYS if key in request_body), None)
    text: Final = None if turns_key is None else _latest_user_message_text(request_body[turns_key])
    if turns_key is None or text is None:
        return MappingProxyType(kept)
    return MappingProxyType({**kept, turns_key: [{"role": "user", "content": text}]})


def _latest_turn_body(
    request_body: Mapping[str, object], call_type: object, originating: object
) -> Mapping[str, object]:
    """A classifier row stores the trimmed originating request, since the composed classifier message quotes the caller's history.

    `originating` comes from the logging payload the classifier wrote, never from the client's body.
    """
    if not isinstance(originating, Mapping):
        return _trimmed_turn(request_body, call_type)
    originating_body: Final = _REQUEST_BODY_ADAPTER.validate_python(originating)
    model: Final = {key: value for key, value in request_body.items() if key == "model"}
    return MappingProxyType(
        {
            **_trimmed_turn(originating_body, None),
            **model,
            "originating_request_masked": _trimmed_turn(originating_body, None),
        }
    )


def _stored_request_body(
    request_body: Mapping[str, object], standard_payload: Mapping[str, object] | None
) -> Mapping[str, object]:
    if not stores_latest_turn_only() or _replayed_by_responses_bridge(standard_payload):
        return request_body
    if standard_payload is None:
        return _latest_turn_body(request_body, None, None)
    return _latest_turn_body(
        request_body, standard_payload.get("call_type"), standard_payload.get("originating_request_masked")
    )


def _redacted_when_message_logging_is_off(
    request_body: Mapping[str, object], litellm_params: Mapping[str, object], kwargs: Mapping[str, object] | None
) -> Mapping[str, object]:
    """Redaction runs on the body that is about to be stored, after any latest_turn selection."""
    if kwargs is None:
        return request_body
    from litellm.litellm_core_utils.redact_messages import perform_redaction, should_redact_message_logging

    model_call_details: Final = {
        "litellm_params": litellm_params,
        "standard_callback_dynamic_params": kwargs.get("standard_callback_dynamic_params"),
    }
    if not should_redact_message_logging(model_call_details=model_call_details):
        return request_body
    redacted: Final = _convert_mapping_to_json_serializable(without_classifier_audit(request_body))
    perform_redaction(model_call_details=redacted, result=None)
    return redacted


def _convert_to_json_serializable_dict(obj: object, visited: set[int] | None = None, max_depth: int = 20) -> object:
    """
    Convert object to JSON-serializable dict, handling Pydantic models safely.

    This avoids pickle-based deepcopy which fails on Pydantic v2 models
    containing _thread.RLock objects.

    Args:
        obj: Object to convert (dict, list, Pydantic model, or primitive)
        visited: Set of object IDs to track circular references
        max_depth: Maximum recursion depth to prevent infinite recursion

    Returns:
        JSON-serializable version of the object
    """
    if max_depth <= 0:
        # Return a placeholder if max depth is exceeded
        return "<max_depth_exceeded>"

    if visited is None:
        visited = set()

    # Get the object's memory address to track visited objects
    obj_id: Final = id(obj)
    if obj_id in visited:
        # Circular reference detected, return placeholder
        return "<circular_reference>"

    # Only track mutable objects (dict, list, objects with __dict__)
    if isinstance(obj, (dict, list)) or hasattr(obj, "__dict__"):
        visited.add(obj_id)

    try:
        if isinstance(obj, BaseModel):
            # Use Pydantic's model_dump() instead of pickle
            result: Final = obj.model_dump()
            # Recursively process the dumped dict
            return _convert_to_json_serializable_dict(result, visited, max_depth - 1)
        elif isinstance(obj, dict):
            return {k: _convert_to_json_serializable_dict(v, visited, max_depth - 1) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [_convert_to_json_serializable_dict(item, visited, max_depth - 1) for item in obj]
        elif hasattr(obj, "__dict__"):
            # Handle objects with __dict__ attribute
            return _convert_to_json_serializable_dict(obj.__dict__, visited, max_depth - 1)
        else:
            # Primitives (str, int, float, bool, None) pass through
            return obj
    finally:
        # Remove from visited set when done processing this object
        if obj_id in visited:
            visited.remove(obj_id)


def _convert_mapping_to_json_serializable(obj: Mapping[str, object]) -> dict[str, object]:
    converted: Final = _convert_to_json_serializable_dict(obj)
    if isinstance(converted, dict):
        return converted
    return dict(obj)


def _placeholder_stored_request_body_metadata(
    request_body: Mapping[str, object], persisted_model_group: str, raw_model: str
) -> Mapping[str, object]:
    body_metadata: Final = request_body.get("metadata")
    if not isinstance(body_metadata, Mapping):
        return request_body
    error_information: Final = body_metadata.get("error_information")
    placeholdered_fields: Final = MappingProxyType(
        {
            "model_group": persisted_model_group,
            "error_information": _scrub_raw_model_from_error_information(
                cast(StandardLoggingPayloadErrorInformation, error_information), raw_model
            )
            if isinstance(error_information, Mapping)
            else error_information,
        }
    )
    return MappingProxyType(
        {
            **request_body,
            "metadata": MappingProxyType(
                {key: placeholdered_fields.get(key, value) for key, value in body_metadata.items()}
            ),
        }
    )


def _placeholder_stored_request_body(
    litellm_params: Mapping[str, object], persisted_model_group: str, raw_model: str
) -> Mapping[str, object]:
    proxy_server_request: Final = litellm_params.get("proxy_server_request")
    if not isinstance(proxy_server_request, Mapping):
        return litellm_params
    request_body: Final = proxy_server_request.get("body")
    if not isinstance(request_body, Mapping):
        return litellm_params
    model_placeholdered: Final = (
        MappingProxyType({**request_body, "model": UNKNOWN_MODEL_SPEND_LOG_MODEL})
        if "model" in request_body
        else request_body
    )
    return MappingProxyType(
        {
            **litellm_params,
            "proxy_server_request": MappingProxyType(
                {
                    **proxy_server_request,
                    "body": _placeholder_stored_request_body_metadata(
                        model_placeholdered, persisted_model_group, raw_model
                    ),
                }
            ),
        }
    )


_CLIENT_IP_HEADERS: Final = frozenset(
    {
        "x-forwarded-for",
        "x-real-ip",
        "forwarded",
        "true-client-ip",
        "cf-connecting-ip",
        "x-client-ip",
        "x-cluster-client-ip",
        "fastly-client-ip",
        "x-original-forwarded-for",
        "x-envoy-external-address",
        "x-azure-clientip",
        "cloudfront-viewer-address",
        "fly-client-ip",
        "cf-connecting-ipv6",
        "cf-pseudo-ipv4",
        "x-appengine-user-ip",
        "x-vercel-forwarded-for",
        "x-arr-clientip",
    }
)


def _stored_value_without_requester_ip_address(key: object, value: object) -> object:
    if key == "requester_ip_address":
        return None
    if key == "error_information" and isinstance(value, Mapping):
        return _error_information_without_ip_addresses(_OBJECT_MAPPING.validate_python(value))
    return _without_requester_ip_address(value, headers=key == "headers")


def _without_requester_ip_address(value: object, *, headers: bool = False) -> object:
    if isinstance(value, Mapping):
        return {
            key: _masked_ip_text(item)
            if headers and isinstance(item, str)
            else _stored_value_without_requester_ip_address(key, item)
            for key, item in _OBJECT_MAPPING.validate_python(value).items()
            if not (headers and isinstance(key, str) and key.lower() in _CLIENT_IP_HEADERS)
        }
    if isinstance(value, list):
        return [_without_requester_ip_address(item) for item in _OBJECT_LIST.validate_python(value)]
    return value


def _request_metadata_without_requester_ip_address(request_body: Mapping[object, object]) -> Mapping[object, object]:
    return {
        key: _without_requester_ip_address(value) if key in _REQUEST_METADATA_KEYS else value
        for key, value in request_body.items()
    }


def _stored_request_without_requester_ip_address(request_body: Mapping[object, object]) -> Mapping[object, object]:
    """The request body, and the classifier input and originating request a classifier row stores in it, with
    `requester_ip_address` set to null, client-IP forwarding headers dropped and every IP address in the other header
    values and in the error text masked inside their metadata."""
    return {
        key: _request_metadata_without_requester_ip_address(_OBJECT_MAPPING.validate_python(value))
        if key in CLASSIFIER_AUDIT_FIELDS and isinstance(value, Mapping)
        else value
        for key, value in _request_metadata_without_requester_ip_address(request_body).items()
    }


def _with_bounded_compaction_summary(item: object, stored_item: object) -> object:
    summary: Final = compaction_summary(item)
    if summary is None or not isinstance(stored_item, Mapping):
        return stored_item
    return {
        **_OBJECT_MAPPING.validate_python(stored_item),
        "encrypted_content": summary_only_compaction_content(_compaction_summary_for_db(summary)),
    }


def _sanitize_proxy_server_request_body(request_body: Mapping[str, object]) -> Mapping[object, object]:
    sanitized: Final = _OBJECT_MAPPING.validate_python(_sanitize_request_body_for_spend_logs_payload(request_body))
    items: Final = request_body.get("input")
    stored_items: Final = sanitized.get("input")
    if not isinstance(items, list) or not isinstance(stored_items, list):
        return sanitized
    return {
        **sanitized,
        "input": [
            _with_bounded_compaction_summary(item, stored_item)
            for item, stored_item in zip(
                _OBJECT_LIST.validate_python(items), _OBJECT_LIST.validate_python(stored_items)
            )
        ],
    }


def _get_proxy_server_request_for_spend_logs_payload(
    metadata: Mapping[str, object],
    litellm_params: Mapping[str, object],
    kwargs: Mapping[str, object] | None = None,
    drop_requester_ip_address: bool = False,
) -> str:
    """
    Only store if should_store_prompts_and_responses_in_spend_logs() is True

    If turn_off_message_logging is enabled, redact messages in the request body.
    """
    if spend_tracking_utils.should_store_prompts_and_responses_in_spend_logs(
        _SPEND_LOGS_PRIVACY_METADATA_ADAPTER.validate_python(metadata)
    ):
        _proxy_server_request: Final = cast(dict | None, litellm_params.get("proxy_server_request", EMPTY_MAPPING))
        if _proxy_server_request is not None:
            _request_body = _proxy_server_request.get("body", EMPTY_MAPPING) or EMPTY_MAPPING

            standard_payload: Final = (kwargs or EMPTY_MAPPING).get("standard_logging_object")
            if isinstance(standard_payload, Mapping):
                _request_body = MappingProxyType({**_request_body, **classifier_audit_fields(standard_payload)})

            if kwargs is not None:
                realtime_tools: Final = kwargs.get("realtime_tools")
                if realtime_tools:
                    _request_body = dict(_request_body)
                    _request_body["tools"] = realtime_tools

            selected_request_body: Final = _stored_request_body(
                _request_body,
                _REQUEST_BODY_ADAPTER.validate_python(standard_payload)
                if isinstance(standard_payload, Mapping)
                else None,
            )
            stored_request_body: Final = _sanitize_proxy_server_request_body(
                _redacted_when_message_logging_is_off(selected_request_body, litellm_params, kwargs)
            )
            _request_body_json_str: Final = safe_dumps(
                _stored_request_without_requester_ip_address(stored_request_body)
                if drop_requester_ip_address
                else stored_request_body,
                non_finite_as_null=True,
            )
            if LITELLM_TRUNCATED_PAYLOAD_FIELD in _request_body_json_str:
                spend_tracking_utils.verbose_proxy_logger.info(
                    "Spend Log: request body was truncated before storing in DB. %s",
                    LITELLM_TRUNCATION_DB_SAFEGUARD_NOTE,
                )
            return _request_body_json_str
    return "{}"


def _get_response_for_spend_logs_payload(
    payload: StandardLoggingPayload | None,
    kwargs: dict[str, object] | None = None,
) -> str:
    if payload is None:
        return "{}"
    if spend_tracking_utils.should_store_prompts_and_responses_in_spend_logs(
        _SPEND_LOGS_PRIVACY_METADATA_ADAPTER.validate_python(get_litellm_metadata_from_kwargs(kwargs or {}))
    ):
        response_obj: object = payload.get("response")
        if response_obj is None:
            return "{}"

        if kwargs is not None:
            realtime_tool_calls: Final = kwargs.get("realtime_tool_calls")
            if realtime_tool_calls and isinstance(response_obj, dict):
                response_obj = dict(response_obj)
                response_obj["tool_calls"] = realtime_tool_calls

        # Apply message redaction if turn_off_message_logging is enabled
        from litellm.litellm_core_utils.redact_messages import (
            perform_redaction,
            should_redact_message_logging,
        )

        redacted: Final = kwargs is not None and should_redact_message_logging(
            model_call_details={
                "litellm_params": kwargs.get("litellm_params", {}),
                "standard_callback_dynamic_params": kwargs.get("standard_callback_dynamic_params"),
            }
        )
        # If redaction is enabled, convert to serializable dict before redacting
        if redacted:
            response_obj = _convert_to_json_serializable_dict(response_obj)
            response_obj = perform_redaction(model_call_details={}, result=response_obj)

        sanitized_wrapper: Final = _sanitize_request_body_for_spend_logs_payload(
            {"response": response_obj},
            keep_compaction_blocks=True,
            keep_provider_signed_fields=not redacted and _replayed_by_responses_bridge(payload),
        )

        sanitized_response: Final = sanitized_wrapper.get("response", response_obj)

        if sanitized_response is None:
            return "{}"
        if isinstance(sanitized_response, str):
            result_str = strip_null_bytes(sanitized_response)
        else:
            result_str = safe_dumps(sanitized_response, non_finite_as_null=True)
        if LITELLM_TRUNCATED_PAYLOAD_FIELD in result_str:
            spend_tracking_utils.verbose_proxy_logger.info(
                "Spend Log: response was truncated before storing in DB. %s",
                LITELLM_TRUNCATION_DB_SAFEGUARD_NOTE,
            )
        return result_str
    return "{}"


def _get_status_for_spend_log(
    metadata: dict,
) -> Literal["success", "failure"]:
    """
    Get the status for the spend log.

    It's only a failure if metadata.get("status") is "failure"
    """
    _status: Final[str | None] = metadata.get("status", None)
    if _status == "failure":
        return "failure"
    return "success"


def logging_payload(
    kwargs: dict[str, Any],  # mutable-ok: the logging kwargs as litellm hands them over, untyped as they were
    response_obj: object,
    start_time: datetime,
    end_time: datetime,
    llm_router: "Router | None" = None,
) -> SpendLogsPayload:
    """The ledger-off ``get_logging_payload``, frozen as it was before the Rust row builder."""
    row_kwargs: Final = _CUSTOMER_METADATA_ADAPTER.validate_python(kwargs)
    subscription_usage: Final = claude_subscription_usage(as_str_mapping(row_kwargs) or {}, response_obj)
    rejected_as_unknown_model: Final = isinstance(response_obj, ProxyModelNotFoundError)
    failure_exception: Final = response_obj if isinstance(response_obj, BaseException) else None
    response_served_from_cache: Final = served_from_cache(response_obj)
    if response_obj is None:
        response_obj = {}
    elif not isinstance(response_obj, BaseModel) and not isinstance(response_obj, dict):
        response_obj = {"result": str(response_obj)}
    # standardize this function to be used across, s3, dynamoDB, langfuse logging
    litellm_params: Final = kwargs.get("litellm_params", {})
    metadata: Final = get_litellm_metadata_from_kwargs(kwargs)
    routing_litellm_params: Final[Mapping[str, object] | None] = (
        litellm_params if isinstance(litellm_params, Mapping) else None
    )
    routing_metadata: Final[object] = (
        routing_litellm_params.get("metadata") if routing_litellm_params is not None else None
    )
    kubernetes_pod_routing: Final = proxy_stamped_kubernetes_pod_routing(
        routing_metadata,
        routing_litellm_params,
    )
    completion_start_time: Final = TypeAdapter[datetime | None](datetime | None).validate_python(
        kwargs.get("completion_start_time", end_time)
    )
    call_type: Final = kwargs.get("call_type")
    cache_hit: Final = kwargs.get("cache_hit", False)

    # Convert response_obj to dict first
    if isinstance(response_obj, dict):
        response_obj_dict = response_obj
    elif isinstance(response_obj, BaseModel):
        response_obj_dict = response_obj.model_dump()
    else:
        response_obj_dict = {}

    # Handle OCR responses which use usage_info instead of usage
    usage: dict = {}
    if call_type in ["ocr", "aocr"]:
        usage = _extract_usage_for_ocr_call(response_obj, response_obj_dict)
    elif not is_unbilled_non_inference_call(call_type, metadata, response_obj_dict):
        # Use response_obj_dict instead of response_obj to avoid calling .get() on Pydantic models
        chat_usage_of_batch: Final = batch_chat_usage(response_obj) if isinstance(response_obj, LiteLLMBatch) else None
        _usage: Final = (
            chat_usage_of_batch.model_dump()
            if chat_usage_of_batch is not None
            else StandardLoggingPayloadSetup.get_usage_as_dict(response_obj_dict)
        )
        if isinstance(_usage, litellm.Usage):
            usage = dict(_usage)
        elif isinstance(_usage, dict):
            usage = _usage

    # A request that failed mid-stream has no usable response_obj usage, and a stream
    # continued on a fallback or a success that bills a failed follow-up only has its own.
    # combined_usage_object holds the usage of every call the row bills, so it wins in each case.
    _combined_usage: Final = kwargs.get("combined_usage_object")
    if (
        not usage or "continued_partial_usage" in kwargs or "failed_attempts_billed_on_success" in kwargs
    ) and isinstance(_combined_usage, litellm.Usage):
        usage = _combined_usage.model_dump()

    id = get_spend_logs_id(call_type or "acompletion", response_obj_dict, kwargs, metadata)
    standard_logging_payload: Final = cast(StandardLoggingPayload | None, kwargs.get("standard_logging_object", None))

    end_user_id = get_end_user_id_for_cost_tracking(
        dict(litellm_params) if isinstance(litellm_params, Mapping) else litellm_params
    )

    api_key = metadata.get("user_api_key", "")

    standard_logging_prompt_tokens: int = 0
    standard_logging_completion_tokens: int = 0
    standard_logging_total_tokens: int = 0
    if standard_logging_payload is not None:
        standard_logging_prompt_tokens = standard_logging_payload.get("prompt_tokens", 0)
        standard_logging_completion_tokens = standard_logging_payload.get("completion_tokens", 0)
        standard_logging_total_tokens = standard_logging_payload.get("total_tokens", 0)
    _trusted_hash = metadata.get("user_api_key_hash")
    _key_alias = metadata.get("user_api_key_alias")
    _key_already_redacted = (
        isinstance(_trusted_hash, str)
        and _is_non_secret_key_value(_trusted_hash, key_alias=_key_alias)
        and _trusted_hash == api_key
    )
    api_key = redact_logged_api_key(api_key, already_redacted=_key_already_redacted, key_alias=_key_alias) or ""

    if (
        standard_logging_payload is not None
    ):  # [TODO] migrate completely to sl payload. currently missing pass-through endpoint data
        api_key = (
            api_key
            or redact_logged_api_key(
                standard_logging_payload["metadata"].get("user_api_key_hash"),
                already_redacted=True,
                key_alias=standard_logging_payload["metadata"].get("user_api_key_alias"),
            )
            or ""
        )
        if not litellm.disable_end_user_cost_tracking:
            end_user_id = end_user_id or standard_logging_payload["metadata"].get("user_api_key_end_user_id")
    request_tags = safe_dumps(metadata.get("tags", [])) if isinstance(metadata.get("tags", []), list) else "[]"
    if (
        standard_logging_payload is not None and standard_logging_payload.get("request_tags") is not None
    ):  # use 'tags' from standard logging payload instead
        request_tags = safe_dumps(standard_logging_payload["request_tags"])

    _model_id: Final = metadata.get("model_info", {}).get("id", "") or _sl_attribution_fallback(
        standard_logging_payload, "model_id"
    )
    _model_group: Final = _string_or_empty(metadata.get("model_group")) or _sl_attribution_fallback(
        standard_logging_payload, "model_group"
    )
    _api_base: Final = litellm_params.get("api_base", "") or _sl_attribution_fallback(
        standard_logging_payload, "api_base"
    )

    # Extract overhead from hidden_params if available
    litellm_overhead_time_ms = None
    if standard_logging_payload is not None:
        hidden_params: Final = standard_logging_payload.get("hidden_params", {})
        litellm_overhead_time_ms = hidden_params.get("litellm_overhead_time_ms")
    upstream_provider_value: Final[object] = (
        standard_logging_payload["hidden_params"]["upstream_provider"]
        if standard_logging_payload is not None
        and "hidden_params" in standard_logging_payload
        and "upstream_provider" in standard_logging_payload["hidden_params"]
        else None
    )
    upstream_provider: Final = upstream_provider_value if isinstance(upstream_provider_value, str) else None

    logged_provider: Final = (
        kwargs.get("custom_llm_provider")
        or _sl_attribution_fallback(standard_logging_payload, "custom_llm_provider")
        or None
    )
    custom_llm_provider: Final = logged_provider or _model_group_provider(_model_group, llm_router)
    requested_model: Final = cast(object, kwargs.get("model"))
    raw_model: Final = requested_model if isinstance(requested_model, str) else ""
    model_is_malformed: Final = requested_model is not None and not isinstance(requested_model, str)
    logged_model: Final = standard_logging_payload.get("model") if standard_logging_payload is not None else None
    resolved_model: Final = (logged_model if isinstance(logged_model, str) else None) or reconstruct_model_name(
        raw_model, logged_provider, metadata or {}
    )
    failed_with_prompt_shaped_model: Final = (
        _get_status_for_spend_log(metadata=metadata) == "failure"
        and not _model_id
        and not _looks_like_model_name(resolved_model)
        and not _is_configured_model_group(_model_group, llm_router)
    )
    model_name: Final = (
        UNKNOWN_MODEL_SPEND_LOG_MODEL
        if rejected_as_unknown_model or failed_with_prompt_shaped_model or model_is_malformed
        else resolved_model
    )
    model_is_placeholdered: Final = model_name == UNKNOWN_MODEL_SPEND_LOG_MODEL
    persisted_model_group: Final = (
        ""
        if model_is_placeholdered and _model_group == raw_model and not _looks_like_model_name(raw_model)
        else _model_group
    )
    route_call_type: Final = _adapter_route_call_type(
        call_type if isinstance(call_type, str) else None,
        cast("Mapping[str, object]", metadata),  # cast-ok: request metadata is an untyped dict here
    )
    persisted_metadata: Final = (
        {
            **metadata,
            "error_information": _scrub_raw_model_from_error_information(metadata.get("error_information"), raw_model),
        }
        if model_is_placeholdered
        else metadata
    )
    litellm_call_id: Final = cast(
        str | None,
        kwargs.get("litellm_call_id") or litellm_params.get("litellm_call_id"),
    )

    no_log: Final = spend_tracking_utils.is_no_log_call(kwargs)
    # clean up litellm metadata
    clean_metadata = _get_spend_logs_metadata(
        persisted_metadata,
        no_log=no_log,
        is_silent_experiment=(
            standard_logging_payload is not None
            and "is_silent_experiment" in standard_logging_payload["metadata"]
            and standard_logging_payload["metadata"]["is_silent_experiment"]
        ),
        applied_guardrails=(
            standard_logging_payload["metadata"].get("applied_guardrails", None)
            if standard_logging_payload is not None
            else None
        ),
        batch_models=(
            standard_logging_payload.get("hidden_params", {}).get("batch_models", None)
            if standard_logging_payload is not None
            else None
        ),
        batch_successful_requests=(
            standard_logging_payload.get("hidden_params", {}).get("batch_successful_requests", None)
            if standard_logging_payload is not None
            else None
        ),
        batch_failed_requests=(
            standard_logging_payload.get("hidden_params", {}).get("batch_failed_requests", None)
            if standard_logging_payload is not None
            else None
        ),
        mcp_tool_call_metadata=(
            standard_logging_payload["metadata"].get("mcp_tool_call_metadata", None)
            if standard_logging_payload is not None
            else None
        ),
        vector_store_request_metadata=(
            standard_logging_payload["metadata"].get("vector_store_request_metadata", None)
            if standard_logging_payload is not None
            else None
        ),
        usage_object=(
            standard_logging_payload["metadata"].get("usage_object", None)
            if standard_logging_payload is not None
            else None
        ),
        model_map_information=(
            standard_logging_payload["model_map_information"] if standard_logging_payload is not None else None
        ),
        guardrail_information=(
            standard_logging_payload.get("guardrail_information", None)
            if standard_logging_payload is not None
            else (metadata.get("standard_logging_guardrail_information", None) if metadata is not None else None)
        ),
        cold_storage_object_key=(
            standard_logging_payload["metadata"].get("cold_storage_object_key", None)
            if standard_logging_payload is not None
            else None
        ),
        litellm_overhead_time_ms=litellm_overhead_time_ms,
        upstream_provider=upstream_provider,
        cost_breakdown=(
            standard_logging_payload.get("cost_breakdown", None)
            if standard_logging_payload is not None
            else _snapshotless_cost_breakdown(row_kwargs)
        ),
        autorouter_savings=(
            standard_logging_payload.get("autorouter_savings", None) if standard_logging_payload is not None else None
        ),
        autorouter_savings_estimate=(
            standard_logging_payload.get("autorouter_savings_estimate")
            if standard_logging_payload is not None
            else None
        ),
        autorouter_baseline_observation=(
            standard_logging_payload.get("autorouter_baseline_observation")
            if standard_logging_payload is not None
            else None
        ),
        litellm_call_id=litellm_call_id,
        router_metadata=_get_router_metadata_for_spend_log(
            metadata=metadata,
            requested_model=persisted_model_group,
            selected_model=model_name,
            selected_provider=custom_llm_provider,
            router_correlation_id=litellm_call_id,
        ),
        used_client_oauth_token=resolve_used_client_oauth_token(
            proxy_stamped_used_client_oauth_token(litellm_params.get("metadata"), litellm_params), custom_llm_provider
        ),
        subscription_usage=(
            standard_logging_payload["metadata"]["claude_subscription_usage"]
            if standard_logging_payload is not None
            and "claude_subscription_usage" in standard_logging_payload["metadata"]
            else subscription_usage
        ),
        kubernetes_pod_routing=kubernetes_pod_routing,
        route_call_type=route_call_type,
        azure_spillover=azure_spillover(
            response_headers=kwargs.get("response_headers")
            if isinstance(kwargs.get("response_headers"), Mapping)
            else None,
            additional_headers=standard_logging_payload["hidden_params"].get("additional_headers")
            if standard_logging_payload is not None
            and isinstance(standard_logging_payload.get("hidden_params"), Mapping)
            else None,
        ),
    )

    special_usage_fields: Final = ["completion_tokens", "prompt_tokens", "total_tokens"]
    additional_usage_values = {}
    for k, v in usage.items():
        if k not in special_usage_fields:
            if isinstance(v, BaseModel):
                v = v.model_dump()
            additional_usage_values.update({k: v})
    prompt_tokens_details = additional_usage_values.get("prompt_tokens_details")
    if not isinstance(prompt_tokens_details, dict):
        usage_object: Final = clean_metadata.get("usage_object")
        if isinstance(usage_object, dict):
            prompt_tokens_details = usage_object.get("prompt_tokens_details")
    if isinstance(prompt_tokens_details, dict):
        if "cache_read_input_tokens" not in additional_usage_values:
            cached_tokens: Final = prompt_tokens_details.get("cached_tokens")
            if isinstance(cached_tokens, int) and cached_tokens > 0:
                additional_usage_values["cache_read_input_tokens"] = cached_tokens
        if "cache_creation_input_tokens" not in additional_usage_values:
            cache_write_tokens: Final = prompt_tokens_details.get("cache_write_tokens") or prompt_tokens_details.get(
                "cache_creation_tokens"
            )
            if isinstance(cache_write_tokens, int) and cache_write_tokens > 0:
                additional_usage_values["cache_creation_input_tokens"] = cache_write_tokens
    clean_metadata["additional_usage_values"] = additional_usage_values
    drop_requester_ip_address: Final = requester_ip_address_logging_disabled()
    if drop_requester_ip_address:
        clean_metadata["requester_ip_address"] = None
    clean_metadata["error_information"] = sanitize_error_information_for_spend_logs(
        clean_metadata["error_information"],
        original_exception=failure_exception,
        metadata=_SPEND_LOGS_PRIVACY_METADATA_ADAPTER.validate_python(metadata),
        no_log=no_log,
    )

    if litellm.cache is None:
        cache_key = "Cache OFF"
    elif litellm_params.get("preset_cache_key") is not None:
        cache_key = litellm_params["preset_cache_key"]
    else:
        cache_key = litellm.cache.get_cache_key(**kwargs)
    if cache_hit is True or response_served_from_cache:
        import time

        id = f"{id}_cache_hit{time.time()}"  # SpendLogs does not allow duplicate request_id

    mcp_namespaced_tool_name = None
    mcp_tool_call_metadata: Final[StandardLoggingMCPToolCall | None] = clean_metadata.get("mcp_tool_call_metadata")
    if mcp_tool_call_metadata is not None:
        mcp_namespaced_tool_name = mcp_tool_call_metadata.get("namespaced_tool_name", None)

    # Extract agent_id for A2A requests (set directly on model_call_details)
    agent_id: Final[str | None] = kwargs.get("agent_id") or metadata.get("agent_id")

    try:
        payload: Final[SpendLogsPayload] = SpendLogsPayload(
            request_id=str(id),
            call_type=call_type or "",
            api_key=str(api_key),
            cache_hit=str(cache_hit),
            startTime=_ensure_datetime_utc(start_time),
            endTime=_ensure_datetime_utc(end_time),
            completionStartTime=(
                _ensure_datetime_utc(completion_start_time) if completion_start_time is not None else None
            ),
            model=model_name,
            user=metadata.get("user_api_key_user_id", "") or "",
            team_id=metadata.get("user_api_key_team_id", "") or "",
            organization_id=metadata.get("user_api_key_org_id") or "",
            metadata=safe_dumps(clean_metadata, non_finite_as_null=True),
            cache_key=cache_key,
            spend=kwargs.get("response_cost", 0),
            total_tokens=usage.get("total_tokens", standard_logging_total_tokens),
            prompt_tokens=usage.get("prompt_tokens", standard_logging_prompt_tokens),
            completion_tokens=usage.get("completion_tokens", standard_logging_completion_tokens),
            request_tags=request_tags,
            end_user=end_user_id or "",
            api_base=_api_base,
            model_group=persisted_model_group,
            model_id=_model_id,
            mcp_namespaced_tool_name=mcp_namespaced_tool_name,
            agent_id=agent_id,
            billing_agent_id=clean_metadata.get("billing_agent_id"),
            requester_ip_address=clean_metadata.get("requester_ip_address", None),
            custom_llm_provider=custom_llm_provider or "",
            messages="{}"
            if no_log
            else _get_messages_for_spend_logs_payload(
                standard_logging_payload=standard_logging_payload, metadata=metadata
            ),
            response="{}"
            if no_log
            else _get_response_for_spend_logs_payload(payload=standard_logging_payload, kwargs=kwargs),
            proxy_server_request="{}"
            if no_log
            else _get_proxy_server_request_for_spend_logs_payload(
                metadata=metadata,
                litellm_params=(
                    _placeholder_stored_request_body(litellm_params, persisted_model_group, raw_model)
                    if model_is_placeholdered
                    else litellm_params
                ),
                kwargs=kwargs,
                drop_requester_ip_address=drop_requester_ip_address,
            ),
            session_id=_get_session_id_for_spend_log(
                kwargs=kwargs,
                metadata=metadata,
                standard_logging_payload=standard_logging_payload,
                omit_when_missing=_omits_session_id_when_missing(metadata),
                batch_trace_session_id=_get_batch_trace_session_id(call_type=call_type, request_id=id),
            ),
            request_duration_ms=_get_request_duration_ms(start_time, end_time),
            status=_get_status_for_spend_log(
                metadata=metadata,
            ),
            litellm_call_id=litellm_call_id,
        )

        spend_tracking_utils.verbose_proxy_logger.debug(
            "SpendTable: created payload - request_id: %s, model: %s, spend: %s",
            payload.get("request_id"),
            payload.get("model"),
            payload.get("spend"),
        )

        # Explicitly clear large intermediate objects to reduce memory pressure
        del response_obj_dict, usage, clean_metadata, additional_usage_values

        return payload
    except Exception as e:
        spend_log_error("Error creating spendlogs object - %s", str(e), exc=e)
        raise e
