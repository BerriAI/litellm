from typing import Any, Dict, Optional

from pydantic import BaseModel


class SkillBase(BaseModel):
    name: str
    description: Optional[str] = None


class SkillNewRequest(SkillBase):
    """Request body for POST /skill/new."""

    content: str  # Full raw SKILL.md (YAML frontmatter + markdown body)


class SkillUpdateRequest(SkillBase):
    """Request body for POST /skill/update."""

    content: Optional[str] = None  # If provided, replaces the full content


class SkillDeleteRequest(BaseModel):
    """Request body for POST /skill/delete."""

    name: str


class SkillConfig(SkillBase):
    """List response - frontmatter only, no body."""

    metadata: Optional[Dict[str, Any]] = None
    created_at: str
    updated_at: str
    created_by: Optional[str] = None


class SkillFullConfig(SkillConfig):
    """Single-skill response - includes full content."""

    content: str
