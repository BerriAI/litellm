from collections.abc import Mapping, Sequence
from datetime import datetime

from pydantic import ConfigDict, Field
from typing_extensions import NotRequired, ReadOnly, TypedDict

from litellm.types.llms.base import LiteLLMBaseModel


class TraceAgent(LiteLLMBaseModel):
    """One agent seen in the caller's traces, for picking which agent's runs to look at."""

    model_config = ConfigDict(frozen=True)

    name: str
    runs: int = Field(ge=0)
    failed_runs: int = Field(ge=0)
    last_seen: datetime
    frameworks: tuple[str, ...] = ()


class TraceAgentList(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    agents: tuple[TraceAgent, ...]


class SpendLogPayload(TypedDict, total=False):
    id: ReadOnly[str | None]
    litellm_call_id: ReadOnly[str | None]
    call_type: ReadOnly[str | None]
    metadata: ReadOnly[Mapping[str, object] | None]
    hidden_params: ReadOnly[Mapping[str, object] | None]
    end_user: ReadOnly[str | None]
    model: ReadOnly[str | None]
    model_group: ReadOnly[str | None]
    model_id: ReadOnly[str | None]
    custom_llm_provider: ReadOnly[str | None]
    api_base: ReadOnly[str | None]
    response_cost: ReadOnly[float | None]
    prompt_tokens: ReadOnly[int | None]
    completion_tokens: ReadOnly[int | None]
    total_tokens: ReadOnly[int | None]
    startTime: ReadOnly[float | None]
    endTime: ReadOnly[float | None]
    completionStartTime: ReadOnly[float | None]
    status: ReadOnly[str | None]
    error_str: ReadOnly[str | None]
    cache_hit: ReadOnly[bool | None]
    session_id: ReadOnly[str | None]
    trace_id: ReadOnly[str | None]
    request_tags: ReadOnly[Sequence[object] | None]
    messages: ReadOnly[object]
    response: ReadOnly[object]


class SpendLogRecord(TypedDict):
    """One LiteLLM request, as written by the `clickhouse` logging callback."""

    request_id: ReadOnly[str]
    response_id: ReadOnly[str]
    provider_request_id: NotRequired[ReadOnly[str]]
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
