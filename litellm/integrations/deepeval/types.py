# Duplicate -> https://github.com/confident-ai/deepeval/blob/main/deepeval/tracing/api.py
from enum import Enum
from typing import Any, ClassVar, Dict, List, Optional, Union, Literal
from pydantic import BaseModel, Field, ConfigDict


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
    name: Optional[str] = None
    status: TraceSpanApiStatus
    type: SpanApiType
    trace_uuid: str = Field(alias="traceUuid")
    parent_uuid: Optional[str] = Field(None, alias="parentUuid")
    start_time: str = Field(alias="startTime")
    end_time: str = Field(alias="endTime")
    input: Optional[Union[Dict, list, str]] = None
    output: Optional[Union[Dict, list, str]] = None
    error: Optional[str] = None

    # llm
    model: Optional[str] = None
    input_token_count: Optional[int] = Field(None, alias="inputTokenCount")
    output_token_count: Optional[int] = Field(None, alias="outputTokenCount")
    cost_per_input_token: Optional[float] = Field(None, alias="costPerInputToken")
    cost_per_output_token: Optional[float] = Field(None, alias="costPerOutputToken")


class TraceApi(BaseModel):
    uuid: str
    base_spans: List[BaseApiSpan] = Field(alias="baseSpans")
    agent_spans: List[BaseApiSpan] = Field(alias="agentSpans")
    llm_spans: List[BaseApiSpan] = Field(alias="llmSpans")
    retriever_spans: List[BaseApiSpan] = Field(alias="retrieverSpans")
    tool_spans: List[BaseApiSpan] = Field(alias="toolSpans")
    start_time: str = Field(alias="startTime")
    end_time: str = Field(alias="endTime")
    metadata: Optional[Dict[str, Any]] = Field(None)
    tags: Optional[List[str]] = Field(None)
    environment: Optional[str] = Field(None)


class Environment(Enum):
    PRODUCTION = "production"
    DEVELOPMENT = "development"
    STAGING = "staging"
