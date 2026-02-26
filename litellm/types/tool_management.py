"""
Pydantic models for Tool Policy management endpoints.
"""

from datetime import datetime
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

ToolCallPolicy = Literal["trusted", "untrusted", "dual_llm", "blocked"]


class LiteLLM_ToolTableRow(BaseModel):
    tool_id: str
    tool_name: str
    origin: Optional[str] = None
    call_policy: ToolCallPolicy = "untrusted"
    call_count: int = 0
    assignments: Optional[Dict] = None
    key_hash: Optional[str] = None
    team_id: Optional[str] = None
    key_alias: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    created_by: Optional[str] = None
    updated_by: Optional[str] = None


class ToolListResponse(BaseModel):
    tools: List[LiteLLM_ToolTableRow]
    total: int


class ToolPolicyUpdateRequest(BaseModel):
    tool_name: str
    call_policy: ToolCallPolicy
    team_id: Optional[str] = None  # if set, create/update override for this team
    key_hash: Optional[str] = None  # if set, create/update override for this key
    key_alias: Optional[str] = None  # human-readable key alias for UI


class ToolPolicyUpdateResponse(BaseModel):
    tool_name: str
    call_policy: ToolCallPolicy
    updated: bool
    team_id: Optional[str] = None
    key_hash: Optional[str] = None


class ToolPolicyOverrideRow(BaseModel):
    override_id: str
    tool_name: str
    team_id: Optional[str] = None
    key_hash: Optional[str] = None
    call_policy: ToolCallPolicy
    key_alias: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


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
