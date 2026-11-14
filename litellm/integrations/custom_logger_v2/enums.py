from enum import Enum, IntEnum
from typing import Final


class Hook(str, Enum):
    prompt_resolve = "prompt_resolve"
    pre_route = "pre_route"
    filter_deployments = "filter_deployments"
    pre_call = "pre_call"
    during_call = "during_call"
    pre_deployment_call = "pre_deployment_call"
    post_deployment_success = "post_deployment_success"
    post_deployment_failure = "post_deployment_failure"
    post_call_success = "post_call_success"
    post_call_failure = "post_call_failure"
    response_headers = "response_headers"
    stream_chunk = "stream_chunk"
    list_models = "list_models"
    mcp_tool_result = "mcp_tool_result"
    log_prepare = "log_prepare"
    provider_request_sent = "provider_request_sent"
    provider_response_received = "provider_response_received"
    attempt_ended = "attempt_ended"
    call_ended = "call_ended"
    router_event = "router_event"
    audit_event = "audit_event"


class FieldTier(IntEnum):
    small_record = 1
    content = 2
    wire = 3
    compatibility = 4


class Field(str, Enum):
    _value_: str
    bit: int
    tier: FieldTier

    def __new__(cls, value: str, bit: int, tier: FieldTier) -> "Field":
        member: Final = str.__new__(cls, value)
        member._value_ = value
        member.bit = bit
        member.tier = tier
        return member

    model = ("model", 0, FieldTier.small_record)
    key_identity = ("key_identity", 1, FieldTier.small_record)
    deployment = ("deployment", 2, FieldTier.small_record)
    candidate_deployments = ("candidate_deployments", 3, FieldTier.small_record)
    request_tags = ("request_tags", 4, FieldTier.small_record)
    dynamic_params = ("dynamic_params", 5, FieldTier.small_record)
    guardrail_info = ("guardrail_info", 6, FieldTier.small_record)
    outcome = ("outcome", 7, FieldTier.small_record)
    error = ("error", 8, FieldTier.small_record)
    usage = ("usage", 9, FieldTier.small_record)
    cost = ("cost", 10, FieldTier.small_record)
    metadata = ("metadata", 11, FieldTier.content)
    messages = ("messages", 12, FieldTier.content)
    input = ("input", 13, FieldTier.content)
    tools = ("tools", 14, FieldTier.content)
    optional_params = ("optional_params", 15, FieldTier.content)
    request_headers = ("request_headers", 16, FieldTier.content)
    response = ("response", 17, FieldTier.content)
    response_headers = ("response_headers", 18, FieldTier.content)
    stream_chunk = ("stream_chunk", 19, FieldTier.content)
    assembled_stream = ("assembled_stream", 20, FieldTier.content)
    tool_result = ("tool_result", 21, FieldTier.content)
    model_list = ("model_list", 22, FieldTier.content)
    request_body = ("request_body", 23, FieldTier.wire)
    provider_request_body = ("provider_request_body", 24, FieldTier.wire)
    provider_response_raw = ("provider_response_raw", 25, FieldTier.wire)
    provider_response_headers = ("provider_response_headers", 26, FieldTier.wire)
    standard_logging_payload = ("standard_logging_payload", 27, FieldTier.compatibility)
    deprecated_kwargs = ("deprecated_kwargs", 28, FieldTier.compatibility)
    deprecated_response_obj = ("deprecated_response_obj", 29, FieldTier.compatibility)
    key_limits = ("key_limits", 30, FieldTier.small_record)
    budget_plan = ("budget_plan", 31, FieldTier.small_record)
    budget_reservation = ("budget_reservation", 32, FieldTier.small_record)
    rate_limit_statuses = ("rate_limit_statuses", 33, FieldTier.small_record)
    output_cap = ("output_cap", 34, FieldTier.small_record)
    attempt_timing = ("attempt_timing", 35, FieldTier.small_record)
    rate_limit_state = ("rate_limit_state", 36, FieldTier.small_record)
    logged_request = ("logged_request", 37, FieldTier.small_record)
    mcp_tool_call = ("mcp_tool_call", 38, FieldTier.small_record)
    log_id = ("log_id", 39, FieldTier.small_record)
    cold_storage_object_key = ("cold_storage_object_key", 40, FieldTier.small_record)
    standard_logging_payload_ref = ("standard_logging_payload_ref", 41, FieldTier.compatibility)
    logged_key_identity = ("logged_key_identity", 42, FieldTier.small_record)
    logged_model_group = ("logged_model_group", 43, FieldTier.small_record)
    metadata_labels = ("metadata_labels", 44, FieldTier.small_record)
    input_sequence_length = ("input_sequence_length", 45, FieldTier.small_record)
    limiter_request = ("limiter_request", 46, FieldTier.small_record)


class Activation(str, Enum):
    required = "required"
    default_on = "default_on"
    opt_in = "opt_in"


class OptOutBy(str, Enum):
    admin = "admin"
    caller = "caller"


DERIVED_ONLY: Final = frozenset(
    {
        Field.key_identity,
        Field.deployment,
        Field.request_tags,
        Field.outcome,
        Field.usage,
        Field.cost,
        Field.key_limits,
        Field.budget_plan,
        Field.output_cap,
        Field.attempt_timing,
        Field.rate_limit_state,
        Field.logged_request,
        Field.mcp_tool_call,
        Field.log_id,
        Field.cold_storage_object_key,
        Field.standard_logging_payload_ref,
        Field.logged_key_identity,
        Field.logged_model_group,
        Field.metadata_labels,
        Field.input_sequence_length,
        Field.limiter_request,
    }
)
BY_REFERENCE: Final = frozenset({Field.standard_logging_payload_ref})

ALL_FIELDS_MASK: Final = sum(1 << field.bit for field in Field)


def field_mask(fields: frozenset[Field]) -> int:
    if Field.deprecated_kwargs in fields:
        return ALL_FIELDS_MASK
    return sum(1 << field.bit for field in fields)
