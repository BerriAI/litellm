"""
Type definitions for OpenAI Skills API

Reference: https://developers.openai.com/api/reference/resources/skills
"""

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field
from typing_extensions import Required, TypedDict


# ──────────────────────────────────────────────
# Request Types
# ──────────────────────────────────────────────


class OpenAICreateSkillRequest(TypedDict, total=False):
    """Request parameters for creating an OpenAI skill (multipart upload)."""

    files: Optional[List[Any]]
    """Files to upload — directory parts or a single .zip containing a top-level folder with SKILL.md."""


class OpenAIListSkillsParams(TypedDict, total=False):
    """Query parameters for listing OpenAI skills."""

    limit: Optional[int]
    """Number of results to return. Default 20, max 100."""

    after: Optional[str]
    """Cursor for forward pagination (ID of the last item from previous page)."""

    before: Optional[str]
    """Cursor for backward pagination (ID of the first item from previous page)."""


class OpenAIUpdateSkillRequest(TypedDict, total=False):
    """Request body for updating an OpenAI skill (e.g. set default_version)."""

    default_version: Optional[int]
    """The version number to set as the new default."""


class OpenAICreateSkillVersionRequest(TypedDict, total=False):
    """Request parameters for creating a new skill version (multipart upload)."""

    files: Optional[List[Any]]
    """Files to upload for the new version."""


class OpenAIListSkillVersionsParams(TypedDict, total=False):
    """Query parameters for listing skill versions."""

    limit: Optional[int]
    """Number of results to return. Default 20, max 100."""

    after: Optional[str]
    """Cursor for forward pagination."""

    before: Optional[str]
    """Cursor for backward pagination."""


# ──────────────────────────────────────────────
# Response Types — Skills
# ──────────────────────────────────────────────


class OpenAISkill(BaseModel):
    """Represents an OpenAI Skill resource."""

    id: str
    """Unique identifier for the skill."""

    created_at: int
    """Unix timestamp (seconds) for when the skill was created."""

    default_version: Optional[int] = None
    """Default version for the skill."""

    description: Optional[str] = None
    """Description of the skill (from SKILL.md front matter)."""

    latest_version: Optional[int] = None
    """Latest version for the skill."""

    name: Optional[str] = None
    """Name of the skill (from SKILL.md front matter)."""

    object: str = "skill"
    """The object type, always 'skill'."""


class OpenAISkillList(BaseModel):
    """Paginated list of OpenAI skills."""

    data: List[OpenAISkill]
    """A list of skill items."""

    first_id: Optional[str] = None
    """The ID of the first item in the list."""

    last_id: Optional[str] = None
    """The ID of the last item in the list."""

    has_more: bool = False
    """Whether there are more items available."""

    object: str = "list"
    """The type of object returned, always 'list'."""


class OpenAIDeletedSkill(BaseModel):
    """Response from deleting an OpenAI skill."""

    id: str
    """The ID of the deleted skill."""

    deleted: bool
    """Whether the skill was successfully deleted."""

    object: str = "skill.deleted"
    """The object type, always 'skill.deleted'."""


# ──────────────────────────────────────────────
# Response Types — Skill Versions
# ──────────────────────────────────────────────


class OpenAISkillVersion(BaseModel):
    """Represents an OpenAI Skill Version resource."""

    id: str
    """Unique identifier for the skill version."""

    created_at: int
    """Unix timestamp (seconds) for when the version was created."""

    description: Optional[str] = None
    """Description of the skill version."""

    name: Optional[str] = None
    """Name of the skill version."""

    object: str = "skill.version"
    """The object type, always 'skill.version'."""

    skill_id: str
    """Identifier of the skill for this version."""

    version: int
    """Version number for this skill."""


class OpenAISkillVersionList(BaseModel):
    """Paginated list of skill versions."""

    data: List[OpenAISkillVersion]
    """A list of skill version items."""

    first_id: Optional[str] = None
    """The ID of the first item in the list."""

    last_id: Optional[str] = None
    """The ID of the last item in the list."""

    has_more: bool = False
    """Whether there are more items available."""

    object: str = "list"
    """The type of object returned, always 'list'."""


class OpenAIDeletedSkillVersion(BaseModel):
    """Response from deleting an OpenAI skill version."""

    id: str
    """The ID of the deleted version."""

    deleted: bool
    """Whether the version was successfully deleted."""

    object: str = "skill.version.deleted"
    """The object type, always 'skill.version.deleted'."""

    version: int
    """The deleted skill version number."""
