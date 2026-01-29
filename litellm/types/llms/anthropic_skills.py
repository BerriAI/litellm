"""
Type definitions for Anthropic Skills API
"""

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field
from typing_extensions import Required, TypedDict


# Skills API Request Types
class CreateSkillRequest(TypedDict, total=False):
    """Request parameters for creating a skill"""

    display_title: Optional[str]
    """Display title for the skill (optional)"""

    files: Optional[List[Any]]
    """Files to upload for the skill. All files must be in the same top-level directory and must include a SKILL.md file at the root."""


class ListSkillsParams(TypedDict, total=False):
    """Query parameters for listing skills"""

    limit: Optional[int]
    """Number of results to return per page. Maximum value is 100. Defaults to 20."""

    page: Optional[str]
    """Pagination token for fetching a specific page of results"""

    source: Optional[str]
    """Filter skills by source ('custom' or 'anthropic')"""


# Skills API Response Types
class Skill(BaseModel):
    """Represents a skill from the Anthropic Skills API"""

    id: str
    """Unique identifier for the skill"""

    created_at: str
    """ISO 8601 timestamp of when the skill was created"""

    display_title: Optional[str] = None
    """Display title for the skill"""

    latest_version: Optional[str] = None
    """The latest version identifier for the skill"""

    source: str
    """Source of the skill (custom or anthropic)"""

    type: str = "skill"
    """Object type, always 'skill'"""

    updated_at: str
    """ISO 8601 timestamp of when the skill was last updated"""


class ListSkillsResponse(BaseModel):
    """Response from listing skills"""

    data: List[Skill]
    """List of skills"""

    next_page: Optional[str] = None
    """Pagination token for the next page"""

    has_more: bool = False
    """Whether there are more skills available"""


class DeleteSkillResponse(BaseModel):
    """Response from deleting a skill"""

    id: str
    """The ID of the deleted skill"""

    type: str = "skill_deleted"
    """Deleted object type, always 'skill_deleted'"""


# Skill Version Types
class CreateSkillVersionRequest(TypedDict, total=False):
    """Request parameters for creating a skill version"""

    display_title: Optional[str]
    """Display title for this version"""

    description: Optional[str]
    """Description of this version"""

    instructions: Optional[str]
    """Instructions for this version"""

    metadata: Optional[Dict[str, Any]]
    """Additional metadata"""


class SkillVersion(BaseModel):
    """Represents a skill version"""

    id: str
    """Unique identifier for the version"""

    skill_id: str
    """ID of the parent skill"""

    created_at: str
    """ISO 8601 timestamp of when the version was created"""

    display_title: Optional[str] = None
    """Display title for this version"""

    description: Optional[str] = None
    """Description of this version"""

    instructions: Optional[str] = None
    """Instructions for this version"""

    metadata: Optional[Dict[str, Any]] = None
    """Additional metadata"""

    type: str = "skill.version"
    """Object type"""


class ListSkillVersionsResponse(BaseModel):
    """Response from listing skill versions"""

    object: str = "list"
    """Object type, always 'list'"""

    data: List[SkillVersion]
    """List of skill versions"""

    first_id: Optional[str] = None
    """ID of the first version in the list"""

    last_id: Optional[str] = None
    """ID of the last version in the list"""

    has_more: bool = False
    """Whether there are more versions available"""


class DeleteSkillVersionResponse(BaseModel):
    """Response from deleting a skill version"""

    id: str
    """The ID of the deleted version"""

    object: str = "skill.version.deleted"
    """Object type"""

    deleted: bool
    """Whether the version was successfully deleted"""

