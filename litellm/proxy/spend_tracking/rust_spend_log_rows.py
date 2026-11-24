import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from typing import (
    TYPE_CHECKING,
    Final,
    NamedTuple,
    cast,  # noqa: TID251  # narrows the untyped request body walk and the row spend
)

from pydantic import BaseModel, TypeAdapter
from typing_extensions import TypeIs  # noqa: TID251  # TypeIs reaches typing only on 3.13

import litellm
from litellm._uuid import uuid
from litellm.constants import (
    CALLER_SESSION_ID_METADATA_KEY,
    DEFAULT_MAX_RECURSE_DEPTH,
    SESSION_ID_GENERATED_METADATA_KEY,
    SESSION_ID_INHERITED_METADATA_KEY,
)
from litellm.litellm_core_utils.core_helpers import (
    claude_subscription_usage,
    get_spend_log_metadata_from_kwargs,
)
from litellm.litellm_core_utils.hidden_params import served_from_cache
from litellm.litellm_core_utils.internal_call_metadata import is_unbilled_non_inference_call
from litellm.litellm_core_utils.litellm_logging import StandardLoggingPayloadSetup
from litellm.litellm_core_utils.ptu_pricing import AZURE_SPILLOVER_FROM_HEADER, AZURE_SPILLOVER_HEADER
from litellm.litellm_core_utils.redact_messages import MESSAGE_REDACTION_ENABLE_HEADERS
from litellm.litellm_core_utils.safe_json_dumps import safe_json_structure
from litellm.litellm_core_utils.secret_redaction import (
    _token_redactor,  # pyright: ignore[reportPrivateUsage]  # the row builder shares the process's compiled token redactor
)
from litellm.proxy._types import SpendLogsMetadata, SpendLogsPayload
from litellm.proxy.route_llm_request import ProxyModelNotFoundError
from litellm.proxy.spend_tracking.spend_logs_ledger_setting import spend_logs_ledger_enabled
from litellm.rust_bridge import _native
from litellm.types.proxy.spend_log_rows import LedgerColumns, NativeSpendLogRow
from litellm.types.utils import LiteLLMBatch, batch_chat_usage

if TYPE_CHECKING:
    from litellm.router import Router

_COMPLETION_START_TIME: Final[TypeAdapter[datetime | None]] = TypeAdapter(datetime | None)
_ROW_METADATA: Final = TypeAdapter(dict[str, object])
_METADATA_KEYS: Final = frozenset(
    {
        *SpendLogsMetadata.__annotations__,
        "model_group",
        "tags",
        "user_api_key_hash",
        "user_api_key_end_user_id",
        "status",
        "session_id",
        "litellm_session_id_omitted",
        "agent_id",
        "deployment",
        "standard_logging_guardrail_information",
        CALLER_SESSION_ID_METADATA_KEY,
    }
)
_MODEL_INFO_KEYS: Final = frozenset({"id", "internal_router_model"})
_REQUEST_METADATA_KEYS: Final = frozenset(
    {
        "used_client_oauth_token",
        "kubernetes_pod_routing",
        "session_id",
        "trace_id",
        SESSION_ID_GENERATED_METADATA_KEY,
        SESSION_ID_INHERITED_METADATA_KEY,
    }
)
_MESSAGE_REDACTION_HEADERS: Final = frozenset({"litellm-disable-message-redaction", *MESSAGE_REDACTION_ENABLE_HEADERS})
_PAYLOAD_KEYS: Final = frozenset(
    {
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "request_tags",
        "model_id",
        "model_group",
        "api_base",
        "custom_llm_provider",
        "model",
        "model_map_information",
        "guardrail_information",
        "cost_breakdown",
        "autorouter_savings",
        "autorouter_savings_estimate",
        "autorouter_baseline_observation",
        "call_type",
        "trace_id",
        "classifier_input",
        "originating_request_masked",
    }
)
_STORED_PROMPT_PAYLOAD_KEYS: Final = frozenset({"response"})
_PAYLOAD_METADATA_KEYS: Final = frozenset(
    {
        "user_api_key_hash",
        "user_api_key_alias",
        "user_api_key_end_user_id",
        "is_silent_experiment",
        "applied_guardrails",
        "mcp_tool_call_metadata",
        "vector_store_request_metadata",
        "usage_object",
        "cold_storage_object_key",
    }
)
_HIDDEN_PARAMS_KEYS: Final = frozenset(
    {
        "litellm_overhead_time_ms",
        "upstream_provider",
        "batch_models",
        "batch_successful_requests",
        "batch_failed_requests",
    }
)
_LITELLM_PARAMS_KEYS: Final = frozenset(
    {
        "api_base",
        "litellm_call_id",
        "litellm_session_id",
        "user_api_key_end_user_id",
    }
)
_KWARGS_KEYS: Final = frozenset(
    {
        "model",
        "custom_llm_provider",
        "response_cost",
        "agent_id",
        "litellm_call_id",
        "litellm_trace_id",
    }
)
_STORED_PROMPT_KWARGS_KEYS: Final = frozenset({"realtime_tools", "realtime_tool_calls"})
_SPILLOVER_HEADERS: Final = (AZURE_SPILLOVER_HEADER, AZURE_SPILLOVER_FROM_HEADER)
_PROVIDER_SPILLOVER_HEADERS: Final = tuple(f"llm_provider-{header}" for header in _SPILLOVER_HEADERS)
PYTHON_OBJECT_KEY: Final = "\x00python_object"


def _is_str_mapping(value: object) -> TypeIs[Mapping[str, object]]:  # guard-ok: logging mappings are str-keyed
    return isinstance(value, Mapping)


def _is_dict(value: object) -> TypeIs[dict[str, object]]:  # guard-ok: str-keyed  # mutable-ok: main's dict
    return isinstance(value, dict)


def _str_mapping(value: object) -> Mapping[str, object] | None:
    return value if _is_str_mapping(value) else None


def _bounded(value: object) -> object:
    return safe_json_structure(value)


_UNSTRIPPED_SESSION_KEYS: Final = frozenset({"litellm_session_id", CALLER_SESSION_ID_METADATA_KEY})


def _bounded_field(key: str, value: object) -> object:
    """``value`` bounded, except a caller session string, which keeps its NUL so Rust can fall back to the trace id."""
    return value if key in _UNSTRIPPED_SESSION_KEYS and isinstance(value, str) else _bounded(value)


def _projected(source: Mapping[str, object], keys: frozenset[str]) -> Mapping[str, object]:
    return {key: _bounded_field(key, value) for key, value in source.items() if key in keys}


def _projected_headers(headers: object, names: Sequence[str] | frozenset[str]) -> Mapping[str, object] | None:
    found: Final = _str_mapping(headers)
    if found is None:
        return None
    return {name: _bounded(value) for name in names if (value := found.get(name)) is not None}


def _request_metadata(litellm_params: Mapping[str, object]) -> Mapping[str, object]:
    return {
        key: {
            **_projected(metadata, _REQUEST_METADATA_KEYS),
            **(
                {"headers": _projected_headers(metadata.get("headers"), _MESSAGE_REDACTION_HEADERS)}
                if "headers" in metadata
                else {}
            ),
        }
        if (metadata := _str_mapping(litellm_params[key])) is not None
        else None
        for key in ("metadata", "litellm_metadata")
        if key in litellm_params
    }


def _timestamp(value: datetime) -> Mapping[str, object]:
    return {"raw": value.isoformat(), "utc": value.astimezone(timezone.utc).isoformat()}


def _metadata(metadata: Mapping[str, object]) -> Mapping[str, object]:
    model_info: Final = _str_mapping(metadata.get("model_info"))
    return {
        **_projected(metadata, _METADATA_KEYS),
        **({"model_info": _projected(model_info, _MODEL_INFO_KEYS)} if model_info is not None else {}),
    }


def _request_body(value: object, path: frozenset[int] = frozenset(), depth: int = 0) -> object:
    identity: Final = id(value)
    if isinstance(value, BaseModel):
        return {PYTHON_OBJECT_KEY: safe_json_structure(value, DEFAULT_MAX_RECURSE_DEPTH - depth)}
    if not isinstance(value, dict | list):
        return safe_json_structure(value, DEFAULT_MAX_RECURSE_DEPTH - depth)
    if depth > DEFAULT_MAX_RECURSE_DEPTH:
        return "MaxDepthExceeded"
    if identity in path:
        return "CircularReference Detected"
    inner: Final = path | {identity}
    if isinstance(value, list):
        items: Final = cast(list[object], value)  # cast-ok: narrowed to a list
        return [_request_body(item, inner, depth + 1) for item in items]
    return {
        key.replace("\x00", ""): _request_body(item, inner, depth + 1)
        for key, item in cast(dict[object, object], value).items()  # cast-ok: narrowed to a dict
        if isinstance(key, str)
    }


def _proxy_server_request(litellm_params: Mapping[str, object]) -> Mapping[str, object]:
    if "proxy_server_request" not in litellm_params:
        return {}
    request: Final = litellm_params["proxy_server_request"]
    found: Final = _str_mapping(request)
    return {
        "proxy_server_request": _bounded(request)
        if found is None
        else {key: _request_body(value) if key == "body" else _bounded(value) for key, value in found.items()}
    }


def _usage(
    call_type: object,
    metadata: Mapping[str, object],
    response_obj: object,
    response_obj_dict: dict[str, object],  # mutable-ok: forwarded to the OCR usage extractor, which takes a dict
) -> Mapping[str, object]:
    from litellm.proxy.spend_tracking.spend_tracking_utils import (
        _extract_usage_for_ocr_call,  # pyright: ignore[reportPrivateUsage, reportUnknownVariableType]  # the row input reuses the logging payload's own OCR usage extraction
    )

    if call_type in ["ocr", "aocr"]:
        return _str_mapping(_extract_usage_for_ocr_call(response_obj, response_obj_dict)) or {}  # pyright: ignore[reportUnknownArgumentType]  # the OCR extractor returns an untyped dict
    if is_unbilled_non_inference_call(call_type, metadata, response_obj_dict):  # pyright: ignore[reportArgumentType]  # call_type is the logged kwargs value, passed as main passes it
        return {}
    chat_usage_of_batch: Final = batch_chat_usage(response_obj) if isinstance(response_obj, LiteLLMBatch) else None
    usage: Final = (
        chat_usage_of_batch.model_dump()
        if chat_usage_of_batch is not None
        else StandardLoggingPayloadSetup.get_usage_as_dict(response_obj_dict)
    )
    if isinstance(usage, litellm.Usage):
        return dict(usage)
    return usage


def _final_usage(kwargs: Mapping[str, object], usage: Mapping[str, object]) -> Mapping[str, object]:
    combined: Final = kwargs.get("combined_usage_object")
    replaced: Final = not usage or "continued_partial_usage" in kwargs or "failed_attempts_billed_on_success" in kwargs
    if replaced and isinstance(combined, litellm.Usage):
        return combined.model_dump()
    return usage


class SpendLogRowConfig(NamedTuple):
    latest_turn_only: bool
    redact_tokens: bool
    drop_requester_ip_address: bool
    omit_missing_session_id: bool
    max_string_length_prompt_in_db: int
    disable_end_user_cost_tracking: bool
    request_correlation_in_logs: bool
    turn_off_message_logging: bool


def spend_log_row_config() -> SpendLogRowConfig:
    from litellm.proxy.proxy_server import general_settings_view
    from litellm.proxy.spend_tracking.spend_tracking_utils import (
        _get_max_string_length_prompt_in_db,  # pyright: ignore[reportPrivateUsage]  # the row reads the limit the Python row reads
        requester_ip_address_logging_disabled,
        stores_latest_turn_only,
    )

    settings: Final = general_settings_view()
    return SpendLogRowConfig(
        latest_turn_only=stores_latest_turn_only(),
        redact_tokens=settings.get("redact_tokens_in_spend_logs") is True,
        drop_requester_ip_address=requester_ip_address_logging_disabled(),
        omit_missing_session_id=settings.get("missing_session_id") == "omit",
        max_string_length_prompt_in_db=_get_max_string_length_prompt_in_db(),
        disable_end_user_cost_tracking=bool(litellm.disable_end_user_cost_tracking),
        request_correlation_in_logs=bool(litellm.request_correlation_in_logs),
        turn_off_message_logging=litellm.turn_off_message_logging is True,
    )


def spend_log_row_input(
    kwargs: dict[str, object],  # mutable-ok: the logging kwargs as litellm hands them over, only read here
    response_obj: object,
    start_time: datetime,
    end_time: datetime,
    llm_router: "Router | None",
    now: float,
    fallback_session_id: str,
) -> Mapping[str, object]:
    from litellm.proxy.spend_tracking.spend_tracking_utils import (
        _adapter_route_call_type,  # pyright: ignore[reportPrivateUsage]  # the route table lookup stays with the route table
        _is_configured_model_group,  # pyright: ignore[reportPrivateUsage]  # router registry lookups stay with the router
        _model_group_provider,  # pyright: ignore[reportPrivateUsage]  # router registry lookups stay with the router
        _snapshotless_cost_breakdown,  # pyright: ignore[reportPrivateUsage]  # validated against the CostBreakdown type
        _string_or_empty,  # pyright: ignore[reportPrivateUsage]  # the router lookup key is read the logging payload's way
        get_spend_logs_id,
        should_store_prompts_and_responses_in_spend_logs,
    )

    response_obj_dict: Final[dict[str, object]] = (  # mutable-ok: the response dict main's extractors take
        {}
        if response_obj is None
        else response_obj
        if _is_dict(response_obj)
        else response_obj.model_dump()
        if isinstance(response_obj, BaseModel)
        else {"result": str(response_obj)}
    )
    response: Final = response_obj if isinstance(response_obj, BaseModel) else response_obj_dict
    litellm_params: Final = _str_mapping(kwargs.get("litellm_params", {})) or {}
    metadata: Final = get_spend_log_metadata_from_kwargs(kwargs)
    completion_start_time: Final = _COMPLETION_START_TIME.validate_python(kwargs.get("completion_start_time", end_time))
    call_type: Final = kwargs.get("call_type")
    usage: Final = _final_usage(kwargs, _usage(call_type, metadata, response, response_obj_dict))
    payload: Final = _str_mapping(kwargs.get("standard_logging_object", None))
    payload_metadata: Final = _str_mapping(payload.get("metadata")) if payload is not None else None
    hidden_params: Final = _str_mapping(payload.get("hidden_params")) if payload is not None else None
    dynamic_params: Final = _str_mapping(kwargs.get("standard_callback_dynamic_params"))
    stores_prompts: Final = should_store_prompts_and_responses_in_spend_logs(metadata)

    model_group: Final = _string_or_empty(metadata.get("model_group")) or (
        _string_or_empty(payload.get("model_group")) if payload is not None else ""
    )
    logged_provider: Final = kwargs.get("custom_llm_provider") or (
        _string_or_empty(payload.get("custom_llm_provider")) if payload is not None else ""
    )
    cache_key: Final = (
        "Cache OFF"
        if litellm.cache is None
        else litellm_params["preset_cache_key"]
        if litellm_params.get("preset_cache_key") is not None
        else litellm.cache.get_cache_key(**kwargs)
    )

    return {
        "request_id": get_spend_logs_id(
            call_type or "acompletion",  # pyright: ignore[reportArgumentType]  # the logged call type, passed as main passes it
            response_obj_dict,
            kwargs,
            metadata,
        ),
        "call_type": _bounded(call_type),
        "cache_hit": _bounded(kwargs.get("cache_hit", False)),
        "served_from_cache": served_from_cache(response_obj),
        "stores_prompts": stores_prompts,
        "rejected_as_unknown_model": isinstance(response_obj, ProxyModelNotFoundError),
        "unknown_model_error_message": _bounded(response_obj.spend_log_error_message)
        if isinstance(response_obj, ProxyModelNotFoundError)
        else None,
        "now": now,
        "fallback_session_id": fallback_session_id,
        "start_time": _timestamp(start_time),
        "end_time": _timestamp(end_time),
        "completion_start_time": _timestamp(completion_start_time) if completion_start_time is not None else None,
        "usage": _bounded(usage),
        "metadata": _metadata(metadata),
        "litellm_params": {
            **_projected(litellm_params, _LITELLM_PARAMS_KEYS),
            **(_proxy_server_request(litellm_params) if stores_prompts else {}),
            **_request_metadata(litellm_params),
        },
        "kwargs": {
            **_projected(kwargs, _KWARGS_KEYS | _STORED_PROMPT_KWARGS_KEYS if stores_prompts else _KWARGS_KEYS),
            "response_headers": _projected_headers(kwargs.get("response_headers"), _SPILLOVER_HEADERS),
            "turn_off_message_logging": _bounded(dynamic_params.get("turn_off_message_logging"))
            if dynamic_params
            else None,
        },
        "standard_logging_object": None
        if payload is None
        else {
            **_projected(payload, _PAYLOAD_KEYS | _STORED_PROMPT_PAYLOAD_KEYS if stores_prompts else _PAYLOAD_KEYS),
            **(
                {"messages": _bounded(payload.get("messages"))}
                if stores_prompts and "messages" in payload and payload.get("call_type") == "_arealtime"
                else {}
            ),
            **(
                {"metadata": _projected(payload_metadata, _PAYLOAD_METADATA_KEYS)}
                if payload_metadata is not None
                else {}
            ),
            **(
                {
                    "hidden_params": {
                        **_projected(hidden_params, _HIDDEN_PARAMS_KEYS),
                        "additional_headers": _projected_headers(
                            hidden_params.get("additional_headers"), _PROVIDER_SPILLOVER_HEADERS
                        ),
                    }
                }
                if hidden_params is not None
                else {}
            ),
        },
        "cache_key": _bounded(cache_key),
        "model_group_provider": None if logged_provider else _model_group_provider(model_group, llm_router),
        "is_configured_model_group": _is_configured_model_group(model_group, llm_router)
        if metadata.get("status") == "failure"
        else False,
        "route_call_type": _adapter_route_call_type(call_type if isinstance(call_type, str) else None, metadata),
        "snapshotless_cost_breakdown": _bounded(
            _snapshotless_cost_breakdown(kwargs)  # pyright: ignore[reportArgumentType]  # main types the kwargs keys as object
        )
        if payload is None
        else None,
        "subscription_usage": _bounded(
            payload_metadata["claude_subscription_usage"]
            if payload_metadata is not None and "claude_subscription_usage" in payload_metadata
            else claude_subscription_usage(kwargs, response_obj)
        ),
    }


def current_spend_log_row_input(
    kwargs: dict[str, object],  # mutable-ok: the logging kwargs as litellm hands them over, only read here
    response_obj: object,
    start_time: datetime,
    end_time: datetime,
    llm_router: "Router | None",
) -> Mapping[str, object]:
    return spend_log_row_input(kwargs, response_obj, start_time, end_time, llm_router, time.time(), str(uuid.uuid4()))


def encode_row_input(row_input: Mapping[str, object]) -> bytes:
    return json.dumps(row_input, ensure_ascii=False).encode("utf-8", "replace")


def _spend_logs_payload(row: NativeSpendLogRow) -> SpendLogsPayload:
    """The builder's row as the payload the writer and loggers take, its timestamps as datetimes."""
    completion_start_time: Final = row["completionStartTime"]
    return SpendLogsPayload(
        request_id=row["request_id"],
        call_type=row["call_type"],
        api_key=row["api_key"],
        cache_hit=row["cache_hit"],
        startTime=datetime.fromisoformat(row["startTime"]),
        endTime=datetime.fromisoformat(row["endTime"]),
        completionStartTime=(
            datetime.fromisoformat(completion_start_time) if completion_start_time is not None else None
        ),
        model=row["model"],
        user=row["user"],
        team_id=row["team_id"],
        organization_id=row["organization_id"],
        metadata=row["metadata"],
        cache_key=row["cache_key"],
        spend=cast(float, row["spend"]),  # cast-ok: Python's builder also passed a missing cost as None
        total_tokens=row["total_tokens"],
        prompt_tokens=row["prompt_tokens"],
        completion_tokens=row["completion_tokens"],
        request_tags=row["request_tags"],
        end_user=row["end_user"],
        api_base=row["api_base"],
        model_group=row["model_group"],
        model_id=row["model_id"],
        mcp_namespaced_tool_name=row["mcp_namespaced_tool_name"],
        agent_id=row["agent_id"],
        billing_agent_id=row["billing_agent_id"],
        requester_ip_address=row["requester_ip_address"],
        custom_llm_provider=row["custom_llm_provider"],
        messages=row["messages"],
        response=row["response"],
        proxy_server_request=row["proxy_server_request"],
        session_id=row["session_id"],
        request_duration_ms=row["request_duration_ms"],
        status=row["status"],
        litellm_call_id=row["litellm_call_id"],
    )


def _iso_text(value: object) -> object:
    return value.isoformat() if isinstance(value, datetime) else value


@dataclass(frozen=True, slots=True)
class LedgerRowInput:
    billable_cost: float | None
    delivery_id: str
    entities: Mapping[str, str | None]
    model_access_groups: Sequence[str]
    windows: Sequence[Mapping[str, object]]
    router: "Callable[[], Router | None]"


def uses_spend_logs_ledger() -> bool:
    from litellm.proxy.proxy_server import general_settings_view

    return spend_logs_ledger_enabled(general_settings_view())


@lru_cache(maxsize=8)
def _row_builder(config: SpendLogRowConfig) -> _native.RustSpendLogRowBuilder:
    return _native.RustSpendLogRowBuilder(
        token_redactor=_token_redactor(),
        latest_turn_only=config.latest_turn_only,
        redact_tokens=config.redact_tokens,
        drop_requester_ip_address=config.drop_requester_ip_address,
        omit_missing_session_id=config.omit_missing_session_id,
        max_string_length_prompt_in_db=config.max_string_length_prompt_in_db,
        disable_end_user_cost_tracking=config.disable_end_user_cost_tracking,
        request_correlation_in_logs=config.request_correlation_in_logs,
        turn_off_message_logging=config.turn_off_message_logging,
    )


def spendlogs_payload_rust(
    kwargs: dict[str, object],  # mutable-ok: the logging kwargs as litellm hands them over, only read here
    response_obj: object,
    start_time: datetime,
    end_time: datetime,
    llm_router: "Router | None",
) -> SpendLogsPayload:
    row_input: Final = current_spend_log_row_input(kwargs, response_obj, start_time, end_time, llm_router)
    return _spend_logs_payload(_row_builder(spend_log_row_config()).build_row(encode_row_input(row_input)))


def ledger_columns(payload: SpendLogsPayload, ledger: LedgerRowInput) -> LedgerColumns:
    from litellm.accounting.snapshot import (
        _savings_or_zero,  # pyright: ignore[reportPrivateUsage]  # savings pricing stays in Python and feeds the Rust row
    )

    return _native.RustSpendLogRowBuilder.ledger_columns(
        {key: _iso_text(value) for key, value in payload.items()},
        billable_cost=ledger.billable_cost,
        org_id=None,
        team_id=None,
        delivery_id=ledger.delivery_id,
        entities=list(ledger.entities.items()),
        model_access_groups=list(ledger.model_access_groups),
        windows=[dict(window) for window in ledger.windows],
        savings=_savings_or_zero(payload, _ROW_METADATA.validate_json(payload["metadata"]), ledger.router)._asdict(),
    )
