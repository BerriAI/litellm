"""
Pydantic models for Tool Policy management endpoints.
"""

from datetime import datetime
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

ToolCallPolicy = Literal["trusted", "untrusted", "dual_llm", "blocked"]

ToolInputPolicy = Literal["trusted", "untrusted", "blocked"]
ToolOutputPolicy = Literal["trusted", "untrusted"]


class LiteLLM_ToolTableRow(BaseModel):
    tool_id: str
    tool_name: str
    origin: Optional[str] = None
    input_policy: ToolInputPolicy = "untrusted"
    output_policy: ToolOutputPolicy = "untrusted"
    call_count: int = 0
    assignments: Optional[Dict] = None
    key_hash: Optional[str] = None
    team_id: Optional[str] = None
    key_alias: Optional[str] = None
    user_agent: Optional[str] = None
    agent_id: Optional[str] = None
    last_used_at: Optional[datetime] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    created_by: Optional[str] = None
    updated_by: Optional[str] = None


class ToolListResponse(BaseModel):
    tools: List[LiteLLM_ToolTableRow]
    total: int


class ToolPolicyUpdateRequest(BaseModel):
    tool_name: str
    input_policy: Optional[ToolInputPolicy] = None
    output_policy: Optional[ToolOutputPolicy] = None
    team_id: Optional[str] = None
    key_hash: Optional[str] = None
    key_alias: Optional[str] = None


class ToolPolicyUpdateResponse(BaseModel):
    tool_name: str
    input_policy: Optional[ToolInputPolicy] = None
    output_policy: Optional[ToolOutputPolicy] = None
    updated: bool
    team_id: Optional[str] = None
    key_hash: Optional[str] = None


class ToolPolicyOverrideRow(BaseModel):
    override_id: str
    tool_name: str
    team_id: Optional[str] = None
    key_hash: Optional[str] = None
    input_policy: ToolInputPolicy = "blocked"
    key_alias: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class ToolPolicyOption(BaseModel):
    value: str
    label: str
    description: str


class ToolPolicyOptionsResponse(BaseModel):
    input_policies: List[ToolPolicyOption]
    output_policies: List[ToolPolicyOption]


class ToolDetailResponse(BaseModel):
    tool: LiteLLM_ToolTableRow
    overrides: List[ToolPolicyOverrideRow] = Field(default_factory=list)


class ToolUsageLogEntry(BaseModel):
    """One spend log row for a tool call (for UI "recent logs" table)."""

    id: str  # request_id
    timestamp: str
    model: Optional[str] = None
    spend: Optional[float] = None
    total_tokens: Optional[int] = None
    input_snippet: Optional[str] = None


class ToolUsageLogsResponse(BaseModel):
    logs: List[ToolUsageLogEntry]
    total: int
    page: int
    page_size: int
