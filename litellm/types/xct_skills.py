"""Pydantic types for xct-native skills (S2-03).

These live alongside the Anthropic skill types in
``litellm.types.llms.anthropic_skills`` — same DB table, different shape.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class XCTSkillCreate(BaseModel):
    """Payload for ``POST /v1/xct-skills`` (JSON or multipart-form fields)."""

    display_title: str
    description: Optional[str] = None
    instructions: Optional[str] = None
    system_prompt_template: Optional[str] = None
    tool_schema: Optional[Dict[str, Any]] = None
    category: Optional[str] = None
    team_id: Optional[str] = None
    is_public: bool = False
    version: Optional[str] = "1"
    xct_metadata: Dict[str, Any] = Field(default_factory=dict)


class XCTSkillPatch(BaseModel):
    """Partial update payload for ``PATCH /v1/xct-skills/{id}``."""

    display_title: Optional[str] = None
    description: Optional[str] = None
    instructions: Optional[str] = None
    system_prompt_template: Optional[str] = None
    tool_schema: Optional[Dict[str, Any]] = None
    is_public: Optional[bool] = None
    version: Optional[str] = None
    xct_metadata: Optional[Dict[str, Any]] = None


class XCTSkill(BaseModel):
    """Full read shape returned by ``GET /v1/xct-skills/{id}``."""

    skill_id: str
    display_title: Optional[str] = None
    description: Optional[str] = None
    instructions: Optional[str] = None
    system_prompt_template: Optional[str] = None
    tool_schema: Optional[Dict[str, Any]] = None
    source: str = "custom"
    version: Optional[str] = None
    is_public: bool = False
    team_id: Optional[str] = None
    user_id: Optional[str] = None
    created_by: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    xct_metadata: Dict[str, Any] = Field(default_factory=dict)


class XCTSkillListResponse(BaseModel):
    """Cursor-paginated list response for ``GET /v1/xct-skills``."""

    data: List[XCTSkill]
    has_more: bool = False
    next_cursor: Optional[str] = None
