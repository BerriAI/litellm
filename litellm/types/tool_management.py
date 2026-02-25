"""
Pydantic models for Tool Policy management endpoints.
"""

from datetime import datetime
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel

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


class ToolPolicyUpdateResponse(BaseModel):
    tool_name: str
    call_policy: ToolCallPolicy
    updated: bool


class ToolCallLogRow(BaseModel):
    id: str
    tool_name: str
    request_id: Optional[str] = None
    key_hash: Optional[str] = None
    team_id: Optional[str] = None
    created_at: Optional[datetime] = None
    # Joined from LiteLLM_SpendLogs (present only when prompt logging is enabled)
    request: Optional[str] = None
    response: Optional[str] = None


class ToolCallLogsResponse(BaseModel):
    calls: List[ToolCallLogRow]
    total: int
    tool_name: str


class ToolSettingsResponse(BaseModel):
    store_tool_call_logs: bool


class ToolSettingsUpdateRequest(BaseModel):
    store_tool_call_logs: bool
