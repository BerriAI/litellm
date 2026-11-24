"""
Pydantic models for Tool Policy management endpoints.
"""

from datetime import datetime
from typing import Literal

from pydantic import Field

from litellm.types.llms.base import LiteLLMBaseModel

ToolCallPolicy = Literal["trusted", "untrusted", "dual_llm", "blocked"]

ToolInputPolicy = Literal["trusted", "untrusted", "blocked"]
ToolOutputPolicy = Literal["trusted", "untrusted"]


class ToolDiscoveryUser(LiteLLMBaseModel):
    user_id: str
    user_email: str | None = None
    user_alias: str | None = None


class LiteLLM_ToolTableRow(LiteLLMBaseModel):
    tool_id: str
    tool_name: str
    origin: str | None = None
    input_policy: ToolInputPolicy = "untrusted"
    output_policy: ToolOutputPolicy = "untrusted"
    call_count: int = 0
    assignments: dict | None = None
    key_hash: str | None = None
    team_id: str | None = None
    key_alias: str | None = None
    user_agent: str | None = None
    user: ToolDiscoveryUser | None = None
    last_used_at: datetime | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    created_by: str | None = None
    updated_by: str | None = None


class ToolListResponse(LiteLLMBaseModel):
    tools: list[LiteLLM_ToolTableRow]
    total: int


class ToolPolicyUpdateRequest(LiteLLMBaseModel):
    tool_name: str
    input_policy: ToolInputPolicy | None = None
    output_policy: ToolOutputPolicy | None = None
    team_id: str | None = None
    key_hash: str | None = None
    key_alias: str | None = None


class ToolPolicyUpdateResponse(LiteLLMBaseModel):
    tool_name: str
    input_policy: ToolInputPolicy | None = None
    output_policy: ToolOutputPolicy | None = None
    updated: bool
    team_id: str | None = None
    key_hash: str | None = None


class ToolPolicyOverrideRow(LiteLLMBaseModel):
    override_id: str
    tool_name: str
    team_id: str | None = None
    key_hash: str | None = None
    input_policy: ToolInputPolicy = "blocked"
    key_alias: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


class ToolPolicyOption(LiteLLMBaseModel):
    value: str
    label: str
    description: str


class ToolPolicyOptionsResponse(LiteLLMBaseModel):
    input_policies: list[ToolPolicyOption]
    output_policies: list[ToolPolicyOption]


class ToolDetailResponse(LiteLLMBaseModel):
    tool: LiteLLM_ToolTableRow
    overrides: list[ToolPolicyOverrideRow] = Field(default_factory=list)


class ToolUsageLogEntry(LiteLLMBaseModel):
    """One spend log row for a tool call (for UI "recent logs" table)."""

    id: str  # request_id
    timestamp: str
    model: str | None = None
    spend: float | None = None
    total_tokens: int | None = None
    input_snippet: str | None = None


class ToolUsageLogsResponse(LiteLLMBaseModel):
    logs: list[ToolUsageLogEntry]
    total: int
    page: int
    page_size: int


class ToolSpendEntry(LiteLLMBaseModel):
    """Total spend attributed to one tool over the requested window."""

    tool_name: str
    spend: float = Field(
        0.0,
        description="Attributed spend: a request that used several tools counts its full spend toward each of them",
    )
    call_count: int = 0
    total_tokens: int = 0


class ToolSpendDailyEntry(LiteLLMBaseModel):
    """Spend attributed to one tool on one UTC day."""

    date: str
    tool_name: str
    spend: float = 0.0
    call_count: int = 0


class ToolSpendResponse(LiteLLMBaseModel):
    by_tool: list[ToolSpendEntry] = Field(default_factory=list)
    daily: list[ToolSpendDailyEntry] = Field(default_factory=list)
    start_date: str | None = None
    end_date: str | None = None
