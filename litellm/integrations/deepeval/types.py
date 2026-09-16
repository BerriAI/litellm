# Duplicate -> https://github.com/confident-ai/deepeval/blob/main/deepeval/tracing/api.py
from enum import Enum
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field


class SpanApiType(Enum):
    BASE = "base"
    AGENT = "agent"
    LLM = "llm"
    RETRIEVER = "retriever"
    TOOL = "tool"


span_api_type_literals = Literal["base", "agent", "llm", "retriever", "tool"]


class TraceSpanApiStatus(Enum):
    SUCCESS = "SUCCESS"
    ERRORED = "ERRORED"


class BaseApiSpan(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(use_enum_values=True)

    uuid: str
    name: str | None = None
    status: TraceSpanApiStatus
    type: SpanApiType
    trace_uuid: str = Field(alias="traceUuid")
    parent_uuid: str | None = Field(None, alias="parentUuid")
    start_time: str = Field(alias="startTime")
    end_time: str = Field(alias="endTime")
    input: dict | list | str | None = None
    output: dict | list | str | None = None
    error: str | None = None

    # llm
    model: str | None = None
    input_token_count: int | None = Field(None, alias="inputTokenCount")
    output_token_count: int | None = Field(None, alias="outputTokenCount")
    cost_per_input_token: float | None = Field(None, alias="costPerInputToken")
    cost_per_output_token: float | None = Field(None, alias="costPerOutputToken")


class TraceApi(BaseModel):
    uuid: str
    base_spans: list[BaseApiSpan] = Field(alias="baseSpans")
    agent_spans: list[BaseApiSpan] = Field(alias="agentSpans")
    llm_spans: list[BaseApiSpan] = Field(alias="llmSpans")
    retriever_spans: list[BaseApiSpan] = Field(alias="retrieverSpans")
    tool_spans: list[BaseApiSpan] = Field(alias="toolSpans")
    start_time: str = Field(alias="startTime")
    end_time: str = Field(alias="endTime")
    metadata: dict[str, Any] | None = Field(None)
    tags: list[str] | None = Field(None)
    environment: str | None = Field(None)


class Environment(Enum):
    PRODUCTION = "production"
    DEVELOPMENT = "development"
    STAGING = "staging"
