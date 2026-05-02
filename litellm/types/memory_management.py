"""
Pydantic models for Memory management endpoints.
"""

from datetime import datetime
from typing import Any, List, Optional

from pydantic import BaseModel, Field


class LiteLLM_MemoryRow(BaseModel):
    memory_id: str
    key: str
    value: str
    metadata: Optional[Any] = None
    user_id: Optional[str] = None
    team_id: Optional[str] = None
    created_at: Optional[datetime] = None
    created_by: Optional[str] = None
    updated_at: Optional[datetime] = None
    updated_by: Optional[str] = None


class MemoryCreateRequest(BaseModel):
    key: str = Field(..., description="Memory key (acts as the namespace in the URL).")
    value: str = Field(
        ..., description="Memory content. Typically markdown/text for LLM context."
    )
    metadata: Optional[Any] = Field(
        default=None,
        description="Optional JSON metadata (tags, structured fields).",
    )
    user_id: Optional[str] = Field(
        default=None,
        description="Scope to this user. Defaults to the caller's user_id.",
    )
    team_id: Optional[str] = Field(
        default=None,
        description="Scope to this team. Defaults to the caller's team_id.",
    )


class MemoryUpdateRequest(BaseModel):
    value: Optional[str] = None
    metadata: Optional[Any] = None
    # Only honored on create (when the row doesn't yet exist) and only for
    # PROXY_ADMIN callers — mirrors MemoryCreateRequest so admins can bootstrap
    # rows scoped to another user/team via PUT, not just POST.
    user_id: Optional[str] = None
    team_id: Optional[str] = None


class MemoryListResponse(BaseModel):
    memories: List[LiteLLM_MemoryRow]
    total: int


class MemoryDeleteResponse(BaseModel):
    key: str
    deleted: bool
