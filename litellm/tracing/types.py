from collections.abc import Sequence

from typing_extensions import ReadOnly, TypedDict


class SpendLogRecord(TypedDict):
    """One LiteLLM request, as written by the `clickhouse` logging callback."""

    request_id: ReadOnly[str]
    response_id: ReadOnly[str]
    provider_request_id: ReadOnly[str]
    litellm_call_id: ReadOnly[str]
    call_type: ReadOnly[str]
    api_key: ReadOnly[str]
    key_alias: ReadOnly[str]
    team_id: ReadOnly[str]
    team_alias: ReadOnly[str]
    organization_id: ReadOnly[str]
    user: ReadOnly[str]
    end_user: ReadOnly[str]
    model: ReadOnly[str]
    model_group: ReadOnly[str]
    model_id: ReadOnly[str]
    custom_llm_provider: ReadOnly[str]
    api_base: ReadOnly[str]
    spend: ReadOnly[float | None]
    prompt_tokens: ReadOnly[int]
    completion_tokens: ReadOnly[int]
    total_tokens: ReadOnly[int]
    cache_read_tokens: ReadOnly[int]
    cache_write_tokens: ReadOnly[int]
    start_time: ReadOnly[int]  # unix ms
    end_time: ReadOnly[int]  # unix ms
    completion_start_time: ReadOnly[int | None]
    status: ReadOnly[str]
    error_str: ReadOnly[str]
    cache_hit: ReadOnly[bool]
    session_id: ReadOnly[str]
    trace_id: ReadOnly[str]  # from an incoming W3C traceparent, if any
    span_id: ReadOnly[str]
    request_tags: ReadOnly[Sequence[str]]
    metadata: ReadOnly[str]
    messages: ReadOnly[str]
    response: ReadOnly[str]
