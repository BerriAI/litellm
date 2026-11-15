from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

from pydantic import TypeAdapter

from litellm.constants import INTERNAL_CALL_ORIGIN_METADATA_KEY, RESPONSES_WEBSOCKET_CALL_ID_METADATA_KEY
from litellm.integrations.custom_logger_v2.enums import Field
from litellm.integrations.custom_logger_v2.field_sources import FieldSource, HookArgs, derived, mapped
from litellm.types.integrations.custom_logger_v2 import (
    AgentPolicy,
    AttemptTiming,
    BudgetPlan,
    BudgetReservation,
    Cost,
    Deployment,
    EstimatedOutputTokens,
    GuardrailEntry,
    InputSequenceLength,
    KeyIdentity,
    KeyLimits,
    LimiterRequest,
    LoggedKeyIdentity,
    LoggedModelGroup,
    LoggedRequest,
    McpToolCall,
    MetadataKind,
    MetadataLabel,
    ModelInfo,
    Outcome,
    OutputCap,
    RateLimitState,
    RateLimitStatuses,
    ReservationEntry,
    ScalarItem,
    Stringified,
    Usage,
    UsageRecord,
    coerce_record,
    coerce_value,
)

_ABSENT: Final = object()
_TAGS: Final[TypeAdapter[tuple[str | int | float | bool | Stringified | None, ...]]] = TypeAdapter(
    tuple[ScalarItem, ...]
)
_DERIVED_RESERVATION_MEMBERS: Final = frozenset({"rejected", "entries_is_list", "not_a_mapping"})
_LIMITER_HEADER_PREFIX: Final = "x-ratelimit-"


def member(source: object, name: str) -> object:
    if isinstance(source, Mapping):
        return source.get(name)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  # untyped legacy mapping
    return getattr(source, name, None)


def mapping(source: object) -> Mapping[str, object] | None:
    return source if isinstance(source, Mapping) else None  # pyright: ignore[reportUnknownVariableType]  # untyped legacy mapping


def auth_of(args: HookArgs) -> object:
    return args.auth if args.auth is not None else getattr(args.logging_obj, "v2_auth", None)


def _metadata_value(source: object, name: str) -> object:
    return member(mapping(source) or {}, name)


def _text(value: object) -> str | None:
    return value if isinstance(value, str) else None


def key_identity(args: HookArgs) -> KeyIdentity | None:
    auth: Final = auth_of(args)
    if auth is None:
        return None
    team_metadata: Final = getattr(auth, "team_metadata", None)
    return coerce_record(
        KeyIdentity,
        {
            "key_hash": getattr(auth, "token", None),
            "api_key": getattr(auth, "api_key", None),
            "key_alias": getattr(auth, "key_alias", None),
            "key_name": getattr(auth, "key_name", None),
            "user_id": getattr(auth, "user_id", None),
            "user_email": getattr(auth, "user_email", None),
            "team_id": getattr(auth, "team_id", None),
            "team_alias": getattr(auth, "team_alias", None),
            "org_id": getattr(auth, "org_id", None),
            "org_alias": getattr(auth, "organization_alias", None),
            "project_id": getattr(auth, "project_id", None),
            "project_alias": getattr(auth, "project_alias", None),
            "end_user_id": getattr(auth, "end_user_id", None),
            "agent_id": getattr(auth, "agent_id", None),
            "invoked_agent_id": getattr(auth, "invoked_agent_id", None),
            "team_priority": _metadata_value(team_metadata, "priority"),
            "key_priority": _metadata_value(getattr(auth, "metadata", None), "priority"),
        },
    )


def _metadata_kind(payload: Mapping[str, object]) -> MetadataKind:
    if "metadata" not in payload:
        return MetadataKind.missing
    metadata: Final = payload["metadata"]
    if metadata is None:
        return MetadataKind.null
    return MetadataKind.mapping if isinstance(metadata, Mapping) else MetadataKind.other


_LOGGED_KEY_MEMBERS: Final = MappingProxyType(
    {
        "key_hash": "user_api_key_hash",
        "key_alias": "user_api_key_alias",
        "key_spend": "user_api_key_spend",
        "key_max_budget": "user_api_key_max_budget",
        "key_budget_reset_at": "user_api_key_budget_reset_at",
        "user_id": "user_api_key_user_id",
        "user_email": "user_api_key_user_email",
        "user_spend": "user_api_key_user_spend",
        "user_max_budget": "user_api_key_user_max_budget",
        "team_id": "user_api_key_team_id",
        "team_alias": "user_api_key_team_alias",
        "team_spend": "user_api_key_team_spend",
        "team_max_budget": "user_api_key_team_max_budget",
        "org_id": "user_api_key_org_id",
        "org_alias": "user_api_key_org_alias",
        "project_id": "user_api_key_project_id",
        "project_alias": "user_api_key_project_alias",
        "end_user_id": "user_api_key_end_user_id",
        "request_route": "user_api_key_request_route",
        "agent_id": "agent_id",
    }
)


def _cost_tracking_end_user_id(params: Mapping[str, object]) -> object:
    litellm_bucket: Final = mapping(params.get("litellm_metadata")) or {}
    metadata_bucket: Final = mapping(params.get("metadata")) or {}
    spend_keys: Final = {key: value for key, value in metadata_bucket.items() if "user_api_key" in key}
    bucket: Final = (
        {**litellm_bucket, **spend_keys} if litellm_bucket and metadata_bucket else litellm_bucket or metadata_bucket
    )
    return params.get("user_api_key_end_user_id") or bucket.get("user_api_key_end_user_id")


_STAMPED_KEYS: Final = (
    *(key for key in _LOGGED_KEY_MEMBERS.values() if key.startswith("user_api_key_")),
    "user_api_key_auth_metadata",
)


def _missing_keys(metadata: Mapping[str, object] | None) -> tuple[str, ...] | None:
    if metadata is None:
        return None
    return tuple(key for key in _STAMPED_KEYS if key not in metadata) or None


def logged_key_identity(args: HookArgs) -> LoggedKeyIdentity | None:
    payload: Final = payload_of(args)
    if payload is None:
        return None
    raw_metadata: Final = payload.get("metadata")
    metadata: Final = mapping(raw_metadata) or {}
    auth_metadata: Final = metadata.get("user_api_key_auth_metadata")
    auth_metadata_mapping: Final = mapping(auth_metadata)
    return coerce_record(
        LoggedKeyIdentity,
        {
            "metadata_kind": _metadata_kind(payload),
            "missing_keys": _missing_keys(mapping(raw_metadata)),
            **{name: metadata.get(key) for name, key in _LOGGED_KEY_MEMBERS.items()},
            "auth_metadata_priority": member(auth_metadata_mapping, "priority"),
            "request_user": member(args.request, "user"),
            "cost_tracking_end_user_id": _cost_tracking_end_user_id(litellm_params(args)),
        },
        rejected=("auth_metadata",) if auth_metadata is not None and auth_metadata_mapping is None else (),
    )


def _estimates(metadata: object, model: str | None) -> tuple[int | None, int | None]:
    from litellm.proxy.auth.auth_utils import (
        ESTIMATED_OUTPUT_TOKENS_FIELD,
        ESTIMATED_OUTPUT_TOKENS_METADATA_FIELDS,
        ESTIMATED_OUTPUT_TOKENS_PER_MODEL_FIELD,
        _validated_output_token_estimate,  # pyright: ignore[reportPrivateUsage]  # the limiter's own parse of the value
        _validated_output_token_estimates_per_model,  # pyright: ignore[reportPrivateUsage]  # the limiter's own parse of the map
    )

    members: Final = mapping(metadata)
    if not members or ESTIMATED_OUTPUT_TOKENS_METADATA_FIELDS.isdisjoint(members):
        return None, None
    per_model: Final = (
        None
        if model is None
        else _validated_output_token_estimates_per_model(members.get(ESTIMATED_OUTPUT_TOKENS_PER_MODEL_FIELD))
    )
    return (
        None if per_model is None or model is None else per_model.get(model),
        _validated_output_token_estimate(members.get(ESTIMATED_OUTPUT_TOKENS_FIELD)),
    )


def _estimated_output_tokens(auth: object, model: str | None) -> EstimatedOutputTokens:
    key_per_model, key_default = _estimates(getattr(auth, "metadata", None), model)
    team_per_model, team_default = _estimates(getattr(auth, "team_metadata", None), model)
    return coerce_record(
        EstimatedOutputTokens,
        {
            "key_per_model": key_per_model,
            "key_default": key_default,
            "team_per_model": team_per_model,
            "team_default": team_default,
        },
    )


_LIMIT_ATTRIBUTES: Final = (
    "rpm_limit",
    "tpm_limit",
    "max_parallel_requests",
    "budget_throttle_pct",
    "user_rpm_limit",
    "user_tpm_limit",
    "team_rpm_limit",
    "team_tpm_limit",
    "team_member_rpm_limit",
    "team_member_tpm_limit",
    "end_user_rpm_limit",
    "end_user_tpm_limit",
    "organization_rpm_limit",
    "organization_tpm_limit",
)


_AGENT_POLICY_LIMITS: Final = ("rpm_limit", "tpm_limit", "session_rpm_limit", "session_tpm_limit")


def _agent_policy(policy: object) -> AgentPolicy | None:
    if policy is None:
        return None
    return coerce_record(AgentPolicy, {name: getattr(policy, name, None) for name in _AGENT_POLICY_LIMITS})


def key_limits(args: HookArgs) -> KeyLimits | None:
    auth: Final = auth_of(args)
    if auth is None:
        return None
    metadata: Final = getattr(auth, "metadata", None)
    model: Final = member(args.request, "model")
    return coerce_record(
        KeyLimits,
        {
            **{name: getattr(auth, name, None) for name in _LIMIT_ATTRIBUTES},
            "rpm_limit_type": _text(_metadata_value(metadata, "rpm_limit_type")),
            "tpm_limit_type": _text(_metadata_value(metadata, "tpm_limit_type")),
            "estimated_output_tokens": _estimated_output_tokens(auth, model if isinstance(model, str) else None),
            "managed_agent_policy": _agent_policy(getattr(auth, "managed_agent_policy", None)),
            "invoked_agent_policy": _agent_policy(getattr(auth, "invoked_agent_policy", None)),
        },
    )


def _plan(raw: object) -> BudgetPlan | None:
    if isinstance(raw, BudgetPlan):
        return raw
    members: Final = mapping(raw)
    if members is None:
        return coerce_value(BudgetPlan.__name__, TypeAdapter(BudgetPlan), raw)
    return coerce_record(BudgetPlan, members)


def claim_budget_plan(args: HookArgs) -> BudgetPlan | None:
    auth: Final = args.auth
    raw: Final[object] = getattr(auth, "budget_plan", None)
    if raw is None:
        return None
    setattr(auth, "budget_plan", None)  # noqa: B010  # claim once: the fallback loop runs pre_call again with this auth
    return _plan(raw)


def _reservation_entry(raw: object) -> ReservationEntry | None:
    members: Final = mapping(raw)
    return None if members is None else coerce_record(ReservationEntry, members)


def _iterated(raw: object) -> tuple[object, ...] | None:
    try:
        return tuple(raw)  # pyright: ignore[reportArgumentType, reportUnknownArgumentType]  # any iterable a caller stored
    except (TypeError, ValueError):
        return None


def _budget_entries(entries: object) -> tuple[object, ...] | None:
    try:
        return tuple(entries or ())  # pyright: ignore[reportArgumentType, reportUnknownArgumentType]  # any iterable a caller stored
    except (TypeError, ValueError):
        return None


def _entries_members(entries: object) -> Mapping[str, object]:
    iterated: Final = None if entries is None else _budget_entries(entries)
    if iterated is None:
        return {"entries": entries, "entries_is_list": None}
    return {"entries": tuple(map(_reservation_entry, iterated)), "entries_is_list": isinstance(entries, list)}


def budget_reservation(args: HookArgs) -> BudgetReservation | None:
    raw: Final[object] = getattr(args.auth, "budget_reservation", None)
    if raw is None:
        return None
    members: Final = mapping(raw)
    if members is None:
        return coerce_record(BudgetReservation, {"not_a_mapping": raw})
    return coerce_record(BudgetReservation, {**members, **_entries_members(members.get("entries"))})


def _written_entry(entry: object) -> object:
    members: Final = mapping(entry)
    if members is None:
        return entry
    return {key: item for key, item in members.items() if item is not None and key != "rejected"}


def _written(key: str, item: object) -> object:
    if key == "entries" and isinstance(item, list | tuple):
        return list(map(_written_entry, item))  # pyright: ignore[reportUnknownArgumentType]  # entries a callback wrote
    return item


def _record_members(value: object) -> Mapping[str, object]:
    members: Final = value.model_dump(mode="python") if isinstance(value, BudgetReservation) else mapping(value)
    if members is None:
        raise TypeError("a budget_reservation write must be a mapping of reservation members")
    return MappingProxyType(
        {
            key: _written(key, item)
            for key, item in members.items()
            if item is not None and key not in _DERIVED_RESERVATION_MEMBERS
        }
    )


def write_budget_reservation(args: HookArgs, value: object) -> None:
    members: Final = _record_members(value)
    current: Final = getattr(args.auth, "budget_reservation", None)
    if isinstance(current, dict):
        current.update(members)  # pyright: ignore[reportUnknownMemberType]  # the shared reservation dict
        return
    setattr(args.auth, "budget_reservation", dict(members))  # noqa: B010  # the auth model types it as a dict


def usage_record(source: object) -> UsageRecord | None:
    if source is None:
        return None
    prompt_details: Final = member(source, "prompt_tokens_details")
    completion_details: Final = member(source, "completion_tokens_details")
    return coerce_record(
        UsageRecord,
        {
            "prompt_tokens": member(source, "prompt_tokens"),
            "completion_tokens": member(source, "completion_tokens"),
            "total_tokens": member(source, "total_tokens"),
            "cached_tokens": member(prompt_details, "cached_tokens"),
            "prompt_audio_tokens": member(prompt_details, "audio_tokens"),
            "cache_write_tokens": member(prompt_details, "cache_write_tokens"),
            "cache_creation_tokens": member(prompt_details, "cache_creation_tokens"),
            "reasoning_tokens": member(completion_details, "reasoning_tokens"),
            "completion_audio_tokens": member(completion_details, "audio_tokens"),
            "cache_read_input_tokens": member(source, "cache_read_input_tokens"),
            "cache_creation_input_tokens": member(source, "cache_creation_input_tokens"),
            "duration_seconds": member(source, "duration_seconds"),
            "output_image_count": member(source, "output_image_count"),
        },
    )


def payload_of(args: HookArgs) -> Mapping[str, object] | None:
    return mapping(member(args.request, "standard_logging_object"))


def payload_metadata(args: HookArgs) -> Mapping[str, object]:
    return mapping(member(payload_of(args), "metadata")) or {}


def _logged_usage(payload: Mapping[str, object] | None) -> UsageRecord | None:
    if payload is None:
        return None
    details: Final = mapping(member(mapping(payload.get("metadata")), "usage_object")) or {}
    return usage_record(
        {**details, **{name: payload.get(name) for name in ("prompt_tokens", "completion_tokens", "total_tokens")}}
    )


def _socket_session(args: HookArgs) -> UsageRecord | None:
    if member(args.request, "call_type") != "aresponses_websocket":
        return None
    from litellm.proxy.hooks.rate_limiter_utils import socket_session_usage

    return usage_record(socket_session_usage(mapping(args.request) or {}, args.result))


def usage(args: HookArgs) -> Usage | None:
    request: Final = args.request
    streamed: Final = member(request, "async_complete_streaming_response")
    return coerce_record(
        Usage,
        {
            "response": usage_record(member(args.result, "usage")),
            "streamed": usage_record(member(streamed, "usage")),
            "combined": usage_record(member(request, "combined_usage_object")),
            "upstream": usage_record(member(request, "_litellm_upstream_reported_usage")),
            "socket_session": _socket_session(args),
            "logged": _logged_usage(payload_of(args)),
        },
    )


def combined_usage(args: HookArgs) -> Usage | None:
    combined: Final = usage_record(member(args.request, "combined_usage_object"))
    return None if combined is None else Usage(combined=combined)


def outcome(args: HookArgs) -> Outcome | None:
    return args.outcome


def _zero_cost(diagnostic: object) -> object:
    members: Final = mapping(diagnostic)
    return diagnostic if members is None else {"reason": members.get("reason"), "reason_present": "reason" in members}


def cost(args: HookArgs) -> Cost | None:
    payload: Final = payload_of(args)
    if payload is None:
        return None
    return coerce_record(
        Cost,
        {
            "response_cost": payload.get("response_cost"),
            "zero_cost": _zero_cost(payload.get("zero_cost_diagnostic")),
        },
    )


def request_tags(args: HookArgs) -> tuple[str | int | float | bool | Stringified | None, ...] | None:
    raw: Final = member(payload_of(args), "request_tags")
    if raw is None:
        return None
    iterated: Final = _iterated(raw)
    return coerce_value("request_tags", _TAGS, raw if iterated is None else iterated)


def _guardrail_modes(mode: object) -> object:
    return mode if isinstance(mode, list) else (mode,)  # pyright: ignore[reportUnknownVariableType]  # untyped legacy modes


def _guardrail_entry(entry: object) -> GuardrailEntry:
    entry_mapping: Final = mapping(entry)
    if entry_mapping is None:
        return GuardrailEntry(mapping=False)
    mode: Final = entry_mapping.get("guardrail_mode")
    return coerce_record(
        GuardrailEntry,
        {
            "mapping": True,
            "modes": _guardrail_modes(mode),
            "duration": entry_mapping.get("duration"),
        },
    )


def guardrail_info(args: HookArgs) -> tuple[GuardrailEntry, ...] | None:
    raw: Final = member(payload_of(args), "guardrail_information")
    if not raw:
        return None
    single: Final = mapping(raw)
    if single is not None:
        return (_guardrail_entry(single),)
    if isinstance(raw, list | tuple):
        return tuple(map(_guardrail_entry, raw))  # pyright: ignore[reportUnknownArgumentType]  # untyped legacy entries
    return (GuardrailEntry(mapping=False),)


def litellm_params(args: HookArgs) -> Mapping[str, object]:
    return mapping(member(args.request, "litellm_params")) or {}


def litellm_metadata(args: HookArgs) -> Mapping[str, object]:
    params: Final = litellm_params(args)
    return mapping(params.get("litellm_metadata")) or mapping(params.get("metadata")) or {}


def _model_info(raw: object) -> object:
    if not raw:
        return None
    members: Final = mapping(raw)
    return (
        raw if members is None else coerce_record(ModelInfo, {name: members.get(name) for name in ("id", "tpm", "rpm")})
    )


def deployment(args: HookArgs) -> Deployment | None:
    params: Final = litellm_params(args)
    metadata: Final = litellm_metadata(args)
    model_info: Final = metadata.get("model_info")
    original: Final = {**metadata, **(mapping(params.get("metadata")) or {})}.get("original_model_group")
    return coerce_record(
        Deployment,
        {
            "model_info": _model_info(model_info),
            "provider": params.get("custom_llm_provider"),
            "tpm": params.get("tpm"),
            "rpm": params.get("rpm"),
            "original_model_group": original if isinstance(original, str) else None,
            "params_present": bool(params),
        },
    )


def attempt_timing(args: HookArgs) -> AttemptTiming | None:
    request: Final = mapping(args.request) or {}
    hidden: Final = mapping(member(payload_of(args), "hidden_params")) or {}
    return coerce_record(
        AttemptTiming,
        {
            "start": args.start_time,
            "end": args.end_time,
            "api_call_start": request.get("api_call_start_time"),
            "completion_start": request.get("completion_start_time"),
            "completion_start_present": "completion_start_time" in request,
            "queue_seconds": _metadata_value(litellm_params(args).get("metadata"), "queue_time_seconds"),
            "overhead_ms": hidden.get("litellm_overhead_time_ms"),
            "logged_start": request.get("start_time"),
            "logged_end": request.get("end_time") or None,
        },
    )


def model_group(args: HookArgs) -> object:
    params: Final = litellm_params(args)
    name: Final = "litellm_metadata" if params.get("litellm_metadata") is not None else "metadata"
    return _metadata_value(params.get(name), "model_group")


def _handler_model_group(request: Mapping[str, object] | None) -> tuple[object, Exception | None]:
    if request is None:
        return None, None
    from litellm.proxy.common_utils.callback_utils import (
        get_model_group_from_litellm_kwargs,  # pyright: ignore[reportUnknownVariableType]  # the helper takes an untyped dict
    )

    try:
        return get_model_group_from_litellm_kwargs(request), None  # pyright: ignore[reportArgumentType]  # the logging handlers pass model_call_details
    except Exception as error:  # noqa: BLE001  # the port reproduces what the logging handlers do when the helper raises
        return None, error


def logged_model_group(args: HookArgs) -> LoggedModelGroup | None:
    request: Final = mapping(args.request)
    if request is None:
        return None
    group, error = _handler_model_group(request)
    if error is not None:
        return LoggedModelGroup(error_type=type(error).__name__, error_text=str(error))
    return coerce_record(LoggedModelGroup, {"value": group})


def rate_limit_state(args: HookArgs) -> RateLimitState | None:
    group, error = _handler_model_group(mapping(args.request))
    raised: Final = error is not None
    metadata: Final = litellm_params(args).get("metadata")
    hidden: Final = member(payload_of(args), "hidden_params")
    headers: Final = member(mapping(hidden), "additional_headers")
    header_map: Final = mapping(headers)
    return coerce_record(
        RateLimitState,
        {
            "key_model_group": None if raised else group,
            "key_remaining_requests": None
            if raised
            else _metadata_value(metadata, f"litellm-key-remaining-requests-{group}"),
            "key_remaining_tokens": None
            if raised
            else _metadata_value(metadata, f"litellm-key-remaining-tokens-{group}"),
            "limiter_headers": None
            if header_map is None
            else {key: value for key, value in header_map.items() if key.startswith(_LIMITER_HEADER_PREFIX)},
            "provider_remaining_requests": member(header_map, "x_ratelimit_remaining_requests"),
            "provider_remaining_tokens": member(header_map, "x_ratelimit_remaining_tokens"),
            "hidden_params_present": isinstance(hidden, Mapping),
            "additional_headers_present": header_map is not None,
        },
    )


def logged_request(args: HookArgs) -> LoggedRequest | None:
    payload: Final = payload_of(args)
    if payload is None:
        return None
    from litellm.litellm_core_utils.service_tier_utils import get_service_tier_from_standard_logging_payload

    metadata: Final = payload_metadata(args)
    return coerce_record(
        LoggedRequest,
        {
            "client_ip": metadata.get("requester_ip_address"),
            "user_agent": metadata.get("user_agent"),
            "service_tier": get_service_tier_from_standard_logging_payload(payload),  # pyright: ignore[reportArgumentType]  # the payload is a TypedDict at runtime
            "stream": payload.get("stream"),
        },
    )


def mcp_tool_call(args: HookArgs) -> McpToolCall | None:
    call: Final = mapping(payload_metadata(args).get("mcp_tool_call_metadata"))
    if call is None:
        return None
    return coerce_record(McpToolCall, {"name": call.get("name"), "server_name": call.get("mcp_server_name")})


def log_id(args: HookArgs) -> str | None:
    return _text(member(payload_of(args), "id"))


def cold_storage_object_key(args: HookArgs) -> str | None:
    return _text(payload_metadata(args).get("cold_storage_object_key"))


def metadata_labels(args: HookArgs) -> tuple[MetadataLabel, ...]:
    from litellm.integrations.prometheus import (
        _get_combined_custom_metadata_from_standard_logging_payload,  # pyright: ignore[reportPrivateUsage, reportUnknownVariableType]  # the metadata prometheus labels from
        get_custom_labels_from_metadata,  # pyright: ignore[reportUnknownVariableType]  # the helper takes an untyped dict
    )

    payload: Final = member(args.request, "standard_logging_object")
    logged: Final = payload if isinstance(payload, dict) else None  # pyright: ignore[reportUnknownVariableType]  # the payload prometheus reads
    combined: Final = _get_combined_custom_metadata_from_standard_logging_payload(logged)  # pyright: ignore[reportUnknownArgumentType]  # the payload prometheus reads
    return tuple(
        MetadataLabel(name=name, value=value)
        for name, value in get_custom_labels_from_metadata(combined).items()  # pyright: ignore[reportArgumentType]  # prometheus passes its combined dict
    )


def input_sequence_length(args: HookArgs) -> InputSequenceLength | None:
    payload: Final = payload_of(args)
    if payload is None:
        return None
    from litellm.integrations.prometheus import PrometheusLogger

    try:
        bucket: Final = PrometheusLogger._get_input_sequence_length(payload, mapping(args.request) or {}, args.result)  # pyright: ignore[reportPrivateUsage, reportArgumentType]  # prometheus's own bucket rule over the payload it reads
    except Exception as error:  # noqa: BLE001  # the port reproduces what prometheus does when the rule raises
        return InputSequenceLength(error_type=type(error).__name__, error_text=str(error))
    return InputSequenceLength(bucket=bucket)


def payload_copy(
    args: HookArgs,
) -> dict[str, object] | None:  # mutable-ok: the shallow copy handed to Rust by reference
    payload: Final = payload_of(args)
    return None if payload is None else dict(payload)


RECORD_TYPES: Final[Mapping[Field, object]] = MappingProxyType(
    {
        Field.key_identity: KeyIdentity,
        Field.deployment: Deployment,
        Field.request_tags: tuple[ScalarItem, ...],
        Field.guardrail_info: tuple[GuardrailEntry, ...],
        Field.outcome: Outcome,
        Field.usage: Usage,
        Field.cost: Cost,
        Field.key_limits: KeyLimits,
        Field.budget_plan: BudgetPlan,
        Field.budget_reservation: BudgetReservation,
        Field.rate_limit_statuses: RateLimitStatuses,
        Field.output_cap: OutputCap,
        Field.attempt_timing: AttemptTiming,
        Field.rate_limit_state: RateLimitState,
        Field.logged_request: LoggedRequest,
        Field.mcp_tool_call: McpToolCall,
        Field.log_id: str,
        Field.cold_storage_object_key: str,
        Field.logged_key_identity: LoggedKeyIdentity,
        Field.logged_model_group: LoggedModelGroup,
        Field.metadata_labels: tuple[MetadataLabel, ...],
        Field.input_sequence_length: InputSequenceLength,
        Field.limiter_request: LimiterRequest,
    }
)

DERIVED_LOGGING_FIELDS: Final[Mapping[Field, FieldSource]] = MappingProxyType(
    {
        Field.logged_key_identity: derived(logged_key_identity),
        Field.logged_model_group: derived(logged_model_group),
        Field.metadata_labels: derived(metadata_labels),
        Field.deployment: derived(deployment),
        Field.request_tags: derived(request_tags),
        Field.guardrail_info: derived(guardrail_info),
        Field.usage: derived(usage),
        Field.cost: derived(cost),
        Field.attempt_timing: derived(attempt_timing),
        Field.rate_limit_state: derived(rate_limit_state),
        Field.logged_request: derived(logged_request),
        Field.mcp_tool_call: derived(mcp_tool_call),
        Field.log_id: derived(log_id),
        Field.cold_storage_object_key: derived(cold_storage_object_key),
    }
)
DERIVED_ATTEMPT_FIELDS: Final[Mapping[Field, FieldSource]] = MappingProxyType(
    {
        **DERIVED_LOGGING_FIELDS,
        Field.outcome: derived(outcome),
        Field.standard_logging_payload_ref: derived(payload_copy),
        Field.input_sequence_length: derived(input_sequence_length),
    }
)
DERIVED_AUTH_FIELDS: Final[Mapping[Field, FieldSource]] = MappingProxyType({Field.key_identity: derived(key_identity)})
DERIVED_POST_CALL_FAILURE_FIELDS: Final[Mapping[Field, FieldSource]] = MappingProxyType(
    {**DERIVED_AUTH_FIELDS, Field.usage: derived(combined_usage)}
)
PRE_CALL_AUTH_SOURCES: Final[Mapping[Field, FieldSource]] = MappingProxyType(
    {
        **DERIVED_AUTH_FIELDS,
        Field.key_limits: derived(key_limits),
        Field.budget_plan: derived(claim_budget_plan),
        Field.budget_reservation: mapped(budget_reservation, write_budget_reservation),
    }
)


def call_internal_origin(args: HookArgs) -> str | None:
    origin: Final = litellm_metadata(args).get(INTERNAL_CALL_ORIGIN_METADATA_KEY)
    return origin if isinstance(origin, str) else None


def call_connection_id(args: HookArgs) -> str | None:
    connection: Final = _metadata_value(
        litellm_params(args).get("litellm_metadata"), RESPONSES_WEBSOCKET_CALL_ID_METADATA_KEY
    )
    return connection if isinstance(connection, str) else None


__all__ = [
    "DERIVED_ATTEMPT_FIELDS",
    "DERIVED_AUTH_FIELDS",
    "DERIVED_LOGGING_FIELDS",
    "DERIVED_POST_CALL_FAILURE_FIELDS",
    "PRE_CALL_AUTH_SOURCES",
    "RECORD_TYPES",
    "attempt_timing",
    "auth_of",
    "budget_reservation",
    "call_connection_id",
    "call_internal_origin",
    "claim_budget_plan",
    "combined_usage",
    "cost",
    "deployment",
    "guardrail_info",
    "input_sequence_length",
    "key_identity",
    "key_limits",
    "log_id",
    "logged_key_identity",
    "logged_model_group",
    "logged_request",
    "mapping",
    "mcp_tool_call",
    "member",
    "metadata_labels",
    "model_group",
    "payload_copy",
    "payload_of",
    "rate_limit_state",
    "request_tags",
    "usage",
    "usage_record",
    "write_budget_reservation",
]
