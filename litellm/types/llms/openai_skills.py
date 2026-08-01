"""Type definitions for the OpenAI Skills API."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict
from typing_extensions import TypedDict


class CreateOpenAISkillRequest(TypedDict, total=False):
    files: list[Any]


class CreateOpenAISkillVersionRequest(TypedDict, total=False):
    files: list[Any]
    default: bool


class UpdateOpenAISkillRequest(TypedDict, total=False):
    default_version: str


class ListOpenAISkillsParams(TypedDict, total=False):
    after: str
    limit: int
    order: Literal["asc", "desc"]


class OpenAISkill(BaseModel):
    id: str
    created_at: int
    default_version: str | None = None
    description: str | None = None
    latest_version: str | None = None
    name: str | None = None
    object: str = "skill"

    model_config = ConfigDict(extra="allow")


class OpenAISkillList(BaseModel):
    data: list[OpenAISkill]
    first_id: str | None = None
    has_more: bool = False
    last_id: str | None = None
    object: str = "list"

    model_config = ConfigDict(extra="allow")


class OpenAIDeletedSkill(BaseModel):
    id: str
    deleted: bool = True
    object: str = "skill.deleted"

    model_config = ConfigDict(extra="allow")


class OpenAISkillVersion(BaseModel):
    id: str
    created_at: int
    description: str | None = None
    name: str | None = None
    object: str = "skill.version"
    skill_id: str
    version: str

    model_config = ConfigDict(extra="allow")


class OpenAISkillVersionList(BaseModel):
    data: list[OpenAISkillVersion]
    first_id: str | None = None
    has_more: bool = False
    last_id: str | None = None
    object: str = "list"

    model_config = ConfigDict(extra="allow")


class OpenAIDeletedSkillVersion(BaseModel):
    id: str
    deleted: bool = True
    object: str = "skill.version.deleted"
    version: str

    model_config = ConfigDict(extra="allow")
