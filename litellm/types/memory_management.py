"""
Pydantic models for Memory management endpoints.
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class LiteLLM_MemoryRow(BaseModel):
    memory_id: str
    key: str
    value: str
    metadata: Any | None = None
    user_id: str | None = None
    team_id: str | None = None
    created_at: datetime | None = None
    created_by: str | None = None
    updated_at: datetime | None = None
    updated_by: str | None = None


class MemoryCreateRequest(BaseModel):
    key: str = Field(..., description="Memory key (acts as the namespace in the URL).")
    value: str = Field(..., description="Memory content. Typically markdown/text for LLM context.")
    metadata: Any | None = Field(
        default=None,
        description="Optional JSON metadata (tags, structured fields).",
    )
    user_id: str | None = Field(
        default=None,
        description="Scope to this user. Defaults to the caller's user_id.",
    )
    team_id: str | None = Field(
        default=None,
        description="Scope to this team. Defaults to the caller's team_id.",
    )


class MemoryUpdateRequest(BaseModel):
    value: str | None = None
    metadata: Any | None = None
    # Only honored on create (when the row doesn't yet exist) and only for
    # PROXY_ADMIN callers — mirrors MemoryCreateRequest so admins can bootstrap
    # rows scoped to another user/team via PUT, not just POST.
    user_id: str | None = None
    team_id: str | None = None


class MemoryListResponse(BaseModel):
    memories: list[LiteLLM_MemoryRow]
    total: int


class MemoryDeleteResponse(BaseModel):
    key: str
    deleted: bool
